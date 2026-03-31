#include "memory_engine.hpp"

#include <stdexcept>
#include <algorithm>

#include <sqlite3.h>

namespace preprocessor {

MemoryEngine::MemoryEngine(const std::string& db_path) {
    int rc = sqlite3_open(db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::string err = sqlite3_errmsg(db_);
        sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("Failed to open SQLite database: " + err);
    }

    const char* create_sql =
        "CREATE TABLE IF NOT EXISTS messages ("
        "  id        INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  role      TEXT    NOT NULL,"
        "  content   TEXT    NOT NULL,"
        "  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP"
        ");";

    char* err_msg = nullptr;
    rc = sqlite3_exec(db_, create_sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::string err;
        if (err_msg) {
            err = err_msg;
            sqlite3_free(err_msg);
        } else if (db_) {
            err = sqlite3_errmsg(db_);
        } else {
            err = "unknown SQLite error";
        }

        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
        throw std::runtime_error("Failed to create messages table: " + err);
    }
}

MemoryEngine::~MemoryEngine() {
    if (db_) {
        sqlite3_close(db_);
    }
}

MemoryEngine::MemoryEngine(MemoryEngine&& other) noexcept
    : db_(other.db_) {
    other.db_ = nullptr;
}

MemoryEngine& MemoryEngine::operator=(MemoryEngine&& other) noexcept {
    if (this != &other) {
        if (db_) {
            sqlite3_close(db_);
        }
        db_ = other.db_;
        other.db_ = nullptr;
    }
    return *this;
}

void MemoryEngine::add_message(const std::string& role, const std::string& content) {
    const char* sql = "INSERT INTO messages (role, content) VALUES (?, ?);";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(
            std::string("Failed to prepare INSERT statement: ") + sqlite3_errmsg(db_));
    }

    sqlite3_bind_text(stmt, 1, role.c_str(), static_cast<int>(role.size()), SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, content.c_str(), static_cast<int>(content.size()), SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        throw std::runtime_error(
            std::string("Failed to insert message: ") + sqlite3_errmsg(db_));
    }
}

std::vector<std::pair<std::string, std::string>> MemoryEngine::get_recent_history(int limit) {
    const char* sql = "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?;";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(
            std::string("Failed to prepare SELECT statement: ") + sqlite3_errmsg(db_));
    }

    sqlite3_bind_int(stmt, 1, limit);

    std::vector<std::pair<std::string, std::string>> messages;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const char* role = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        const char* content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        messages.emplace_back(role ? role : "", content ? content : "");
    }

    if (rc != SQLITE_DONE) {
        std::string err = sqlite3_errmsg(db_);
        sqlite3_finalize(stmt);
        throw std::runtime_error("Failed to read messages: " + err);
    }

    sqlite3_finalize(stmt);

    // Reverse so messages are in chronological order (oldest first).
    std::reverse(messages.begin(), messages.end());
    return messages;
}

void MemoryEngine::update_last_message(const std::string& content) {
    const char* sql =
        "UPDATE messages SET content = ? WHERE id = (SELECT MAX(id) FROM messages);";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(
            std::string("Failed to prepare UPDATE statement: ") + sqlite3_errmsg(db_));
    }

    sqlite3_bind_text(stmt, 1, content.c_str(), static_cast<int>(content.size()), SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        throw std::runtime_error(
            std::string("Failed to update last message: ") + sqlite3_errmsg(db_));
    }
}

void MemoryEngine::clear_history() {
    const char* sql = "DELETE FROM messages;";
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::string err;
        if (err_msg) { err = err_msg; sqlite3_free(err_msg); }
        else { err = sqlite3_errmsg(db_); }
        throw std::runtime_error("Failed to clear history: " + err);
    }
}

void MemoryEngine::prune(int max_rows) {
    const char* sql =
        "DELETE FROM messages WHERE id NOT IN "
        "(SELECT id FROM messages ORDER BY id DESC LIMIT ?);";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(
            std::string("Failed to prepare prune statement: ") + sqlite3_errmsg(db_));
    }

    sqlite3_bind_int(stmt, 1, max_rows);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        throw std::runtime_error(
            std::string("Failed to prune messages: ") + sqlite3_errmsg(db_));
    }
}

} // namespace preprocessor
