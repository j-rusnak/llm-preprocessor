#pragma once

#include <string>
#include <vector>
#include <utility>

struct sqlite3;

namespace preprocessor {

class MemoryEngine {
public:
    explicit MemoryEngine(const std::string& db_path);
    ~MemoryEngine();

    MemoryEngine(const MemoryEngine&) = delete;
    MemoryEngine& operator=(const MemoryEngine&) = delete;
    MemoryEngine(MemoryEngine&& other) noexcept;
    MemoryEngine& operator=(MemoryEngine&& other) noexcept;

    void add_message(const std::string& role, const std::string& content);
    std::vector<std::pair<std::string, std::string>> get_recent_history(int limit = 5);

private:
    sqlite3* db_ = nullptr;
};

} // namespace preprocessor
