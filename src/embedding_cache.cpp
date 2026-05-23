#include "embedding_cache.hpp"

#include <cstring>
#include <mutex>
#include <stdexcept>

#include <sqlite3.h>
#include <xxhash.h>

namespace preprocessor {

struct EmbeddingCache::Impl {
    sqlite3* db = nullptr;
    mutable std::mutex mu;

    ~Impl() {
        if (db) sqlite3_close(db);
    }
};

namespace {

void exec(sqlite3* db, const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "unknown sqlite error";
        sqlite3_free(err);
        throw std::runtime_error("EmbeddingCache: " + msg);
    }
}

}  // namespace

EmbeddingCache::EmbeddingCache(const std::string& db_path, std::string model_id)
    : impl_(std::make_unique<Impl>()), model_id_(std::move(model_id)) {
    if (sqlite3_open(db_path.c_str(), &impl_->db) != SQLITE_OK) {
        throw std::runtime_error("EmbeddingCache: cannot open " + db_path);
    }
    exec(impl_->db,
         "CREATE TABLE IF NOT EXISTS embeddings ("
         "  key INTEGER PRIMARY KEY,"
         "  model TEXT NOT NULL,"
         "  dim INTEGER NOT NULL,"
         "  vec BLOB NOT NULL"
         ");");
    exec(impl_->db, "PRAGMA journal_mode=WAL;");
    exec(impl_->db, "PRAGMA synchronous=NORMAL;");
}

EmbeddingCache::~EmbeddingCache() = default;

std::uint64_t EmbeddingCache::key_for(std::string_view content) const noexcept {
    XXH64_state_t* st = XXH64_createState();
    XXH64_reset(st, 0);
    XXH64_update(st, model_id_.data(), model_id_.size());
    static const char sep = '|';
    XXH64_update(st, &sep, 1);
    XXH64_update(st, content.data(), content.size());
    auto h = XXH64_digest(st);
    XXH64_freeState(st);
    return h;
}

std::optional<std::vector<float>> EmbeddingCache::get(std::string_view content) const {
    return get_by_key(key_for(content));
}

std::optional<std::vector<float>> EmbeddingCache::get_by_key(std::uint64_t key) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    sqlite3_stmt* st = nullptr;
    const char* sql = "SELECT dim, vec FROM embeddings WHERE key = ?";
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &st, nullptr) != SQLITE_OK) {
        return std::nullopt;
    }
    sqlite3_bind_int64(st, 1, static_cast<sqlite3_int64>(key));
    std::optional<std::vector<float>> out;
    if (sqlite3_step(st) == SQLITE_ROW) {
        int dim = sqlite3_column_int(st, 0);
        const void* blob = sqlite3_column_blob(st, 1);
        int bytes = sqlite3_column_bytes(st, 1);
        if (dim > 0 && bytes == static_cast<int>(dim * sizeof(float))) {
            std::vector<float> v(dim);
            std::memcpy(v.data(), blob, bytes);
            out = std::move(v);
        }
    }
    sqlite3_finalize(st);
    return out;
}

void EmbeddingCache::put(std::string_view content, const std::vector<float>& vec) {
    put_by_key(key_for(content), vec);
}

void EmbeddingCache::put_by_key(std::uint64_t key, const std::vector<float>& vec) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    sqlite3_stmt* st = nullptr;
    const char* sql =
        "INSERT OR REPLACE INTO embeddings(key, model, dim, vec) VALUES(?, ?, ?, ?)";
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &st, nullptr) != SQLITE_OK) {
        throw std::runtime_error("EmbeddingCache::put: prepare failed");
    }
    sqlite3_bind_int64(st, 1, static_cast<sqlite3_int64>(key));
    sqlite3_bind_text(st, 2, model_id_.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(st, 3, static_cast<int>(vec.size()));
    sqlite3_bind_blob(st, 4, vec.data(),
                      static_cast<int>(vec.size() * sizeof(float)), SQLITE_TRANSIENT);
    if (sqlite3_step(st) != SQLITE_DONE) {
        sqlite3_finalize(st);
        throw std::runtime_error("EmbeddingCache::put: step failed");
    }
    sqlite3_finalize(st);
}

std::size_t EmbeddingCache::size() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(impl_->db, "SELECT COUNT(*) FROM embeddings", -1,
                           &st, nullptr) != SQLITE_OK) return 0;
    std::size_t n = 0;
    if (sqlite3_step(st) == SQLITE_ROW) {
        n = static_cast<std::size_t>(sqlite3_column_int64(st, 0));
    }
    sqlite3_finalize(st);
    return n;
}

void EmbeddingCache::clear() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    exec(impl_->db, "DELETE FROM embeddings");
}

}  // namespace preprocessor
