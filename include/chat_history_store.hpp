#pragma once

#include <string>
#include <utility>
#include <vector>

struct sqlite3;

namespace preprocessor {

/// SQLite-backed conversation history store.
///
/// (Renamed from `MemoryEngine` in Phase 0. The legacy class was overloaded
/// with two unrelated concerns — chat turns and vector memory — which are now
/// split. Vector / embedding memory lives in `VectorStore`.)
class ChatHistoryStore {
public:
    explicit ChatHistoryStore(const std::string& db_path);
    ~ChatHistoryStore();

    ChatHistoryStore(const ChatHistoryStore&) = delete;
    ChatHistoryStore& operator=(const ChatHistoryStore&) = delete;
    ChatHistoryStore(ChatHistoryStore&& other) noexcept;
    ChatHistoryStore& operator=(ChatHistoryStore&& other) noexcept;

    void add_message(const std::string& role, const std::string& content);
    void update_last_message(const std::string& content);
    void clear_history();
    void prune(int max_rows);
    std::vector<std::pair<std::string, std::string>> get_recent_history(int limit = 5);

private:
    sqlite3* db_ = nullptr;
};

} // namespace preprocessor
