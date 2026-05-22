#pragma once

#include "code_chunker.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace preprocessor {

class IEmbeddingEngine;
class VectorStore;
class BM25Index;
class FileWatcher;
class HybridRetriever;

/// One retrieved chunk hydrated with its original text + metadata.
struct RetrievedChunk {
    CodeChunk chunk;
    float score;
};

/// Configuration knobs for `RepoIndex`.
struct RepoIndexConfig {
    std::size_t embedding_dim = 384;
    std::size_t max_chunks = 100000;
    std::string metric = "cosine";
    /// File extensions to index. Empty = accept all.
    std::vector<std::string> include_extensions = {
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
        ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
        ".md", ".txt"
    };
    /// Directory names to skip during the initial walk (in addition to any
    /// hidden directory starting with '.').
    std::vector<std::string> skip_dirs = {
        "build", "node_modules", "vcpkg_installed", "target",
        ".git", ".vscode", "__pycache__"
    };
    bool watch_for_changes = true;
    std::size_t rrf_k = 60;
};

/// Owns the full retrieval stack for one repository:
///   chunker -> embedder -> VectorStore + BM25Index -> HybridRetriever,
/// plus a `FileWatcher` that re-indexes individual files when they change.
///
/// All chunk metadata (file_path, line range, raw text) is held in memory in
/// `chunks_by_id_` so retrieval results can be hydrated without touching
/// disk. The vector + bm25 backends store only ids + features.
///
/// Thread-safety: `search()` may be called concurrently with itself. All
/// mutators (`index_path`, `reindex_file`, `forget_file`) and the file
/// watcher callback are serialised by an internal mutex.
class RepoIndex {
public:
    RepoIndex(std::shared_ptr<IEmbeddingEngine> embedder,
              std::shared_ptr<IChunker> chunker,
              RepoIndexConfig config = {});
    ~RepoIndex();

    RepoIndex(const RepoIndex&) = delete;
    RepoIndex& operator=(const RepoIndex&) = delete;

    /// Walk `root` recursively, chunk + embed + index every accepted file.
    /// Idempotent: files already indexed at their current content hash are
    /// skipped.
    void index_path(const std::string& root);

    /// Drop everything we know about `file_path` and re-chunk + re-embed
    /// from the current contents. No-op if the file no longer exists (the
    /// old entries are still removed).
    void reindex_file(const std::string& file_path);

    /// Forget every chunk derived from `file_path` (used on delete events).
    void forget_file(const std::string& file_path);

    /// Hybrid retrieval. Embeds the query once, fuses BM25 + vector hits,
    /// returns hydrated chunks ordered by RRF score descending.
    std::vector<RetrievedChunk> search(const std::string& query,
                                       std::size_t k = 8) const;

    std::size_t chunk_count() const;
    std::size_t file_count() const;

private:
    void index_file_locked(const std::string& file_path);
    void forget_file_locked(const std::string& file_path);
    bool accepts_path(const std::string& path) const;

    std::shared_ptr<IEmbeddingEngine> embedder_;
    std::shared_ptr<IChunker> chunker_;
    RepoIndexConfig config_;

    std::unique_ptr<VectorStore> vectors_;
    std::unique_ptr<BM25Index> keywords_;
    std::unique_ptr<HybridRetriever> retriever_;
    std::unique_ptr<FileWatcher> watcher_;

    mutable std::mutex mu_;
    // id -> chunk metadata (text, file_path, lines, symbol).
    std::unordered_map<std::uint64_t, CodeChunk> chunks_by_id_;
    // file_path -> ids contributed by that file.
    std::unordered_map<std::string, std::unordered_set<std::uint64_t>> ids_by_file_;
    // file_path -> watcher id of the directory we're watching for it.
    std::unordered_map<std::string, long> watch_ids_by_root_;
};

} // namespace preprocessor
