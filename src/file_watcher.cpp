#include "file_watcher.hpp"

#include <mutex>
#include <unordered_map>

#include <efsw/efsw.hpp>

namespace preprocessor {

namespace {

FileEventType to_file_event_type(efsw::Action action) {
    switch (action) {
        case efsw::Actions::Add:      return FileEventType::Added;
        case efsw::Actions::Delete:   return FileEventType::Removed;
        case efsw::Actions::Modified: return FileEventType::Modified;
        case efsw::Actions::Moved:    return FileEventType::Renamed;
    }
    return FileEventType::Modified;
}

class DispatchListener : public efsw::FileWatchListener {
public:
    explicit DispatchListener(FileWatcher::Callback cb) : cb_(std::move(cb)) {}

    void handleFileAction(efsw::WatchID /*watchid*/,
                          const std::string& dir,
                          const std::string& filename,
                          efsw::Action action,
#ifdef LLM_PREPROCESSOR_EFSW_OLD_FILENAME_CONST_REF
                          const std::string& oldFilename) override {
#else
                          std::string oldFilename) override {
#endif
        if (!cb_) return;
        FileEvent ev;
        ev.type = to_file_event_type(action);
        ev.path = dir + filename;
        if (!oldFilename.empty()) {
            ev.old_path = dir + oldFilename;
        }
        cb_(ev);
    }

private:
    FileWatcher::Callback cb_;
};

} // namespace

struct FileWatcher::Impl {
    efsw::FileWatcher watcher;
    std::mutex mu;
    std::unordered_map<long, std::unique_ptr<DispatchListener>> listeners;
    bool started = false;
};

FileWatcher::FileWatcher() : impl_(std::make_unique<Impl>()) {}
FileWatcher::~FileWatcher() = default;

long FileWatcher::add_watch(const std::string& directory, Callback callback) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto listener = std::make_unique<DispatchListener>(std::move(callback));
    efsw::WatchID id = impl_->watcher.addWatch(directory, listener.get(), /*recursive*/ true);
    if (id < 0) {
        return id; // efsw returns negative codes on failure; surface them as-is.
    }
    impl_->listeners.emplace(static_cast<long>(id), std::move(listener));
    if (!impl_->started) {
        impl_->watcher.watch();
        impl_->started = true;
    }
    return static_cast<long>(id);
}

void FileWatcher::remove_watch(long watch_id) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->watcher.removeWatch(static_cast<efsw::WatchID>(watch_id));
    impl_->listeners.erase(watch_id);
}

} // namespace preprocessor
