#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace efsw {
class FileWatcher;
}

namespace preprocessor {

enum class FileEventType {
    Added,
    Modified,
    Removed,
    Renamed
};

struct FileEvent {
    FileEventType type;
    std::string path;     // absolute path of the affected file
    std::string old_path; // populated only for Renamed events
};

/// Thin RAII wrapper around `efsw::FileWatcher`. Watches one or more
/// directories recursively and dispatches debounced events to a user
/// callback. The callback runs on efsw's internal watcher thread; consumers
/// should marshal heavy work (re-embedding, re-indexing) onto their own
/// queue.
///
/// Phase 0 ships the wrapper + integration tests. Phase 1 wires it into the
/// indexer so changed files are re-chunked and re-embedded incrementally.
class FileWatcher {
public:
    using Callback = std::function<void(const FileEvent&)>;

    FileWatcher();
    ~FileWatcher();

    FileWatcher(const FileWatcher&) = delete;
    FileWatcher& operator=(const FileWatcher&) = delete;

    /// Begin watching `directory` recursively. Returns an opaque watch id
    /// that can be passed to `remove_watch()`.
    long add_watch(const std::string& directory, Callback callback);

    /// Stop watching the directory associated with `watch_id`.
    void remove_watch(long watch_id);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace preprocessor
