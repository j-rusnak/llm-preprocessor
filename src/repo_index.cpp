#include "repo_index.hpp"

#include "bm25_index.hpp"
#include "file_watcher.hpp"
#include "hybrid_retriever.hpp"
#include "i_embedding_engine.hpp"
#include "vector_store.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace preprocessor {

namespace {

std::string read_file_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string lower_ext(const fs::path& p) {
    std::string e = p.extension().string();
    for (auto& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return e;
}

} // namespace

RepoIndex::RepoIndex(std::shared_ptr<IEmbeddingEngine> embedder,
                     std::shared_ptr<IChunker> chunker,
                     RepoIndexConfig config)
    : embedder_(std::move(embedder)),
      chunker_(std::move(chunker)),
      config_(std::move(config)) {
    if (!embedder_) throw std::invalid_argument("RepoIndex: embedder is null");
    if (!chunker_) throw std::invalid_argument("RepoIndex: chunker is null");
    if (config_.embedding_dim == 0) throw std::invalid_argument("embedding_dim must be > 0");

    vectors_ = std::make_unique<VectorStore>(config_.embedding_dim,
                                             config_.max_chunks,
                                             std::string{},
                                             config_.metric);
    keywords_ = std::make_unique<BM25Index>();
    retriever_ = std::make_unique<HybridRetriever>(*vectors_, *keywords_, config_.rrf_k);
    if (config_.watch_for_changes) {
        watcher_ = std::make_unique<FileWatcher>();
    }
}

RepoIndex::~RepoIndex() = default;

bool RepoIndex::accepts_path(const std::string& path) const {
    if (config_.include_extensions.empty()) return true;
    std::string e = lower_ext(fs::path(path));
    for (const auto& want : config_.include_extensions) {
        if (e == want) return true;
    }
    return false;
}

void RepoIndex::index_path(const std::string& root) {
    fs::path root_p(root);
    if (!fs::exists(root_p)) {
        throw std::runtime_error("RepoIndex::index_path: not found: " + root);
    }
    if (fs::is_regular_file(root_p)) {
        std::lock_guard<std::mutex> lock(mu_);
        index_file_locked(fs::absolute(root_p).string());
        return;
    }

    std::vector<std::string> to_index;
    for (auto it = fs::recursive_directory_iterator(
                 root_p, fs::directory_options::skip_permission_denied);
         it != fs::recursive_directory_iterator(); ++it) {
        const auto& entry = *it;
        if (entry.is_directory()) {
            std::string dname = entry.path().filename().string();
            bool skip = !dname.empty() && dname.front() == '.';
            if (!skip) {
                for (const auto& s : config_.skip_dirs) {
                    if (dname == s) { skip = true; break; }
                }
            }
            if (skip) {
                it.disable_recursion_pending();
            }
            continue;
        }
        if (!entry.is_regular_file()) continue;
        std::string abs = fs::absolute(entry.path()).string();
        if (!accepts_path(abs)) continue;
        to_index.push_back(std::move(abs));
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto& f : to_index) {
            index_file_locked(f);
        }
    }

    if (watcher_) {
        std::string abs_root = fs::absolute(root_p).string();
        std::lock_guard<std::mutex> lock(mu_);
        if (!watch_ids_by_root_.count(abs_root)) {
            long id = watcher_->add_watch(abs_root, [this](const FileEvent& ev) {
                if (!accepts_path(ev.path)) return;
                switch (ev.type) {
                case FileEventType::Added:
                case FileEventType::Modified:
                    this->reindex_file(ev.path);
                    break;
                case FileEventType::Removed:
                    this->forget_file(ev.path);
                    break;
                case FileEventType::Renamed:
                    if (!ev.old_path.empty()) this->forget_file(ev.old_path);
                    this->reindex_file(ev.path);
                    break;
                }
            });
            if (id >= 0) watch_ids_by_root_[abs_root] = id;
        }
    }
}

void RepoIndex::index_file_locked(const std::string& file_path) {
    // Always drop existing entries first so a content change replaces them.
    forget_file_locked(file_path);

    std::string text = read_file_text(file_path);
    if (text.empty()) return;

    auto chunks = chunker_->chunk(file_path, text);
    if (chunks.empty()) return;

    std::vector<std::string> texts;
    texts.reserve(chunks.size());
    for (const auto& c : chunks) texts.push_back(c.text);
    auto embeds = embedder_->generate_embeddings(texts);
    if (embeds.size() != chunks.size()) {
        throw std::runtime_error("RepoIndex: embedder returned mismatched batch size");
    }

    auto& bucket = ids_by_file_[file_path];
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        // Skip duplicate ids (e.g. identical chunks across files).
        if (chunks_by_id_.count(chunks[i].id)) {
            bucket.insert(chunks[i].id);
            continue;
        }
        vectors_->add(chunks[i].id, embeds[i]);
        keywords_->add(chunks[i].id, chunks[i].text);
        bucket.insert(chunks[i].id);
        chunks_by_id_.emplace(chunks[i].id, std::move(chunks[i]));
    }
}

void RepoIndex::reindex_file(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mu_);
    index_file_locked(file_path);
}

void RepoIndex::forget_file(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mu_);
    forget_file_locked(file_path);
}

void RepoIndex::forget_file_locked(const std::string& file_path) {
    auto it = ids_by_file_.find(file_path);
    if (it == ids_by_file_.end()) return;
    for (auto id : it->second) {
        vectors_->remove(id);
        keywords_->remove(id);
        chunks_by_id_.erase(id);
    }
    ids_by_file_.erase(it);
}

std::vector<RetrievedChunk> RepoIndex::search(const std::string& query, std::size_t k) const {
    if (k == 0) return {};
    auto q_emb = embedder_->generate_embedding(query);
    if (q_emb.size() != config_.embedding_dim) {
        throw std::runtime_error("RepoIndex::search: query embedding dim mismatch");
    }
    std::vector<HybridHit> hits;
    {
        std::lock_guard<std::mutex> lock(mu_);
        hits = retriever_->search(query, q_emb, k);
    }
    std::vector<RetrievedChunk> out;
    out.reserve(hits.size());
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto& h : hits) {
            auto it = chunks_by_id_.find(h.id);
            if (it == chunks_by_id_.end()) continue;
            out.push_back({it->second, h.score});
        }
    }
    return out;
}

std::size_t RepoIndex::chunk_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return chunks_by_id_.size();
}

std::size_t RepoIndex::file_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return ids_by_file_.size();
}

std::vector<CodeChunk> RepoIndex::snapshot_chunks() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<CodeChunk> out;
    out.reserve(chunks_by_id_.size());
    for (const auto& kv : chunks_by_id_) out.push_back(kv.second);
    return out;
}

} // namespace preprocessor
