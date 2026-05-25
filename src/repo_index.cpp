#include "repo_index.hpp"

#include "bm25_index.hpp"
#include "file_watcher.hpp"
#include "hybrid_retriever.hpp"
#include "i_embedding_engine.hpp"
#include "symbol_graph.hpp"
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
        const std::uint64_t id = chunks[i].id;
        auto& refs = chunk_refs_by_id_[id];
        refs[file_path] = chunks[i];
        bucket.insert(id);

        // Keep one indexed representative per content-addressed chunk id.
        if (chunks_by_id_.count(id)) {
            continue;
        }
        vectors_->add(id, embeds[i]);
        keywords_->add(id, chunks[i].text);
        if (symbol_graph_ && symbol_extractor_) {
            symbol_graph_->update_chunk(chunks[i],
                                        symbol_extractor_->extract(chunks[i]));
        }
        chunks_by_id_.emplace(id, std::move(chunks[i]));
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
    if (symbol_graph_) symbol_graph_->remove_file(file_path);
    for (auto id : it->second) {
        bool erased_representative = false;
        auto chunk_it = chunks_by_id_.find(id);
        if (chunk_it != chunks_by_id_.end() && chunk_it->second.file_path == file_path) {
            erased_representative = true;
        }

        auto refs_it = chunk_refs_by_id_.find(id);
        if (refs_it != chunk_refs_by_id_.end()) {
            refs_it->second.erase(file_path);
            if (!refs_it->second.empty()) {
                if (erased_representative) {
                    chunks_by_id_[id] = refs_it->second.begin()->second;
                    if (symbol_graph_ && symbol_extractor_) {
                        symbol_graph_->update_chunk(chunks_by_id_[id],
                                                    symbol_extractor_->extract(chunks_by_id_[id]));
                    }
                }
                continue;
            }
            chunk_refs_by_id_.erase(refs_it);
        }

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

bool RepoIndex::try_get_chunk(std::uint64_t id, CodeChunk& out) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = chunks_by_id_.find(id);
    if (it == chunks_by_id_.end()) return false;
    out = it->second;
    return true;
}

std::vector<SyncVectorEntry> RepoIndex::snapshot_vectors(std::size_t limit) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<SyncVectorEntry> out;
    const auto vectors = vectors_->snapshot(limit);
    out.reserve(vectors.size());
    for (const auto& v : vectors) {
        auto it = chunks_by_id_.find(v.id);
        if (it == chunks_by_id_.end()) continue;
        const auto& chunk = it->second;
        SyncVectorEntry entry;
        entry.chunk_id = v.id;
        entry.vec = v.embedding;
        entry.source_path = chunk.file_path;
        entry.text = chunk.text;
        entry.start_line = chunk.start_line;
        entry.end_line = chunk.end_line;
        entry.symbol = chunk.symbol;
        out.push_back(std::move(entry));
        if (limit > 0 && out.size() >= limit) break;
    }
    return out;
}

std::size_t RepoIndex::apply_synced_vectors(
    const std::vector<SyncVectorEntry>& vectors) {
    std::lock_guard<std::mutex> lock(mu_);
    std::size_t applied = 0;
    for (const auto& v : vectors) {
        if (v.chunk_id == 0 || v.vec.size() != config_.embedding_dim ||
            v.text.empty()) {
            continue;
        }

        CodeChunk chunk{
            v.chunk_id,
            v.source_path.empty()
                ? std::string{"synced:"} + std::to_string(v.chunk_id)
                : v.source_path,
            v.text,
            v.start_line == 0 ? 1 : v.start_line,
            v.end_line == 0 ? (v.start_line == 0 ? 1 : v.start_line) : v.end_line,
            v.symbol
        };

        auto& refs = chunk_refs_by_id_[chunk.id];
        refs[chunk.file_path] = chunk;
        ids_by_file_[chunk.file_path].insert(chunk.id);
        chunks_by_id_[chunk.id] = chunk;

        vectors_->add(chunk.id, v.vec);
        keywords_->add(chunk.id, chunk.text);
        if (symbol_graph_ && symbol_extractor_) {
            symbol_graph_->update_chunk(chunk,
                                        symbol_extractor_->extract(chunk));
        }
        ++applied;
    }
    return applied;
}

void RepoIndex::attach_symbol_graph(SymbolGraph* graph,
                                    ISymbolExtractor* extractor) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    symbol_graph_ = graph;
    symbol_extractor_ = extractor;
}

} // namespace preprocessor
