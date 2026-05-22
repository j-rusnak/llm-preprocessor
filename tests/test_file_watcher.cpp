#include <gtest/gtest.h>
#include "file_watcher.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace fs = std::filesystem;

namespace {
fs::path make_temp_dir(const std::string& tag) {
    auto p = fs::temp_directory_path() /
             ("llm_pp_fw_" + tag + "_" +
              std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(p);
    return p;
}
} // namespace

// File watching is timing/platform sensitive; we keep this test deliberately
// permissive — it asserts the wrapper can attach a watch and at least receive
// SOME callback when a file is touched. Hard determinism is enforced by the
// smoke runner under controlled conditions.
TEST(FileWatcherTest, NotifiesOnFileChange) {
    auto dir = make_temp_dir("notify");

    std::atomic<int> events{0};
    preprocessor::FileWatcher watcher;
    long id = watcher.add_watch(dir.string(), [&](const preprocessor::FileEvent&) {
        events.fetch_add(1, std::memory_order_relaxed);
    });
    ASSERT_GE(id, 0);

    // Trigger a few file system changes.
    for (int i = 0; i < 3; ++i) {
        std::ofstream out(dir / ("f" + std::to_string(i) + ".txt"));
        out << "hello " << i;
    }

    // efsw polls on its own thread; give it a moment.
    for (int i = 0; i < 20 && events.load() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    watcher.remove_watch(id);

    EXPECT_GT(events.load(), 0);

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(FileWatcherTest, RemoveUnknownWatchIsSafe) {
    preprocessor::FileWatcher watcher;
    EXPECT_NO_THROW(watcher.remove_watch(999999));
}
