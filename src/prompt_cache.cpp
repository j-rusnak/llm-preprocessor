#include "prompt_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <stdexcept>

#include <sqlite3.h>
#include <xxhash.h>

namespace preprocessor {

namespace {

std::int64_t now_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void exec_or_throw(sqlite3* db, const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "(null)";
        sqlite3_free(err);
        throw std::runtime_error(std::string("sqlite exec failed: ") + msg);
    }
}

} // namespace

std::string PromptCache::make_key(const std::string& model,
                                  const std::string& prompt,
                                  const std::vector<std::uint64_t>& chunk_ids) {
    std::vector<std::uint64_t> sorted_ids = chunk_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());

    XXH64_state_t* st = XXH64_createState();
    XXH64_reset(st, 0);
    XXH64_update(st, model.data(), model.size());
    const char nul = '\0';
    XXH64_update(st, &nul, 1);
    XXH64_update(st, prompt.data(), prompt.size());
    XXH64_update(st, &nul, 1);
    for (auto id : sorted_ids) {
        XXH64_update(st, &id, sizeof(id));
    }
    const std::uint64_t hv = XXH64_digest(st);
    XXH64_freeState(st);

    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(hv));
    return std::string(buf);
}

PromptCache::PromptCache(const std::string& db_path, std::uint64_t ttl_seconds)
    : ttl_seconds_(ttl_seconds) {
    if (sqlite3_open(db_path.c_str(), &db_) != SQLITE_OK) {
        std::string msg = db_ ? sqlite3_errmsg(db_) : "open failed";
        if (db_) sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("PromptCache open: " + msg);
    }
    exec_or_throw(db_, "PRAGMA journal_mode=WAL;");
    exec_or_throw(db_,
        "CREATE TABLE IF NOT EXISTS cache ("
        " key TEXT PRIMARY KEY,"
        " payload TEXT NOT NULL,"
        " created_at INTEGER NOT NULL"
        ");");
}

PromptCache::~PromptCache() {
    if (db_) sqlite3_close(db_);
}

PromptCache::PromptCache(PromptCache&& other) noexcept
    : db_(other.db_), ttl_seconds_(other.ttl_seconds_) {
    other.db_ = nullptr;
}

PromptCache& PromptCache::operator=(PromptCache&& other) noexcept {
    if (this != &other) {
        if (db_) sqlite3_close(db_);
        db_ = other.db_;
        ttl_seconds_ = other.ttl_seconds_;
        other.db_ = nullptr;
    }
    return *this;
}

std::optional<std::string> PromptCache::get(const std::string& key) const {
    const char* sql = "SELECT payload, created_at FROM cache WHERE key = ?;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error(std::string("PromptCache get prepare: ") + sqlite3_errmsg(db_));
    }
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<std::string> out;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char* p = sqlite3_column_text(stmt, 0);
        std::int64_t created = sqlite3_column_int64(stmt, 1);
        bool expired = ttl_seconds_ > 0 &&
            (now_seconds() - created) > static_cast<std::int64_t>(ttl_seconds_);
        if (!expired && p) out.emplace(reinterpret_cast<const char*>(p));
    }
    sqlite3_finalize(stmt);
    return out;
}

void PromptCache::put(const std::string& key, const std::string& payload) {
    const char* sql =
        "INSERT INTO cache(key, payload, created_at) VALUES(?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET payload = excluded.payload, "
        "created_at = excluded.created_at;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error(std::string("PromptCache put prepare: ") + sqlite3_errmsg(db_));
    }
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, payload.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 3, now_seconds());
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        std::string msg = sqlite3_errmsg(db_);
        sqlite3_finalize(stmt);
        throw std::runtime_error("PromptCache put step: " + msg);
    }
    sqlite3_finalize(stmt);
}

void PromptCache::erase(const std::string& key) {
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "DELETE FROM cache WHERE key = ?;", -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error(std::string("PromptCache erase: ") + sqlite3_errmsg(db_));
    }
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

void PromptCache::clear() {
    exec_or_throw(db_, "DELETE FROM cache;");
}

std::size_t PromptCache::size() const {
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM cache;", -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error(std::string("PromptCache size: ") + sqlite3_errmsg(db_));
    }
    std::size_t n = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        n = static_cast<std::size_t>(sqlite3_column_int64(stmt, 0));
    }
    sqlite3_finalize(stmt);
    return n;
}

} // namespace preprocessor
