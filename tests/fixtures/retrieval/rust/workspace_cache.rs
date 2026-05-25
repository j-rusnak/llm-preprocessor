use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

pub struct WorkspaceCache {
    capacity: usize,
    entries: HashMap<String, PathBuf>,
    lru: VecDeque<String>,
}

impl WorkspaceCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    pub fn remember_snapshot(&mut self, cache_key: String, path: PathBuf) {
        self.entries.insert(cache_key.clone(), path);
        self.lru.push_back(cache_key);
        while self.entries.len() > self.capacity {
            if let Some(evicted) = self.lru.pop_front() {
                self.entries.remove(&evicted);
            }
        }
    }

    pub fn get_snapshot(&mut self, cache_key: &str) -> Option<PathBuf> {
        let path = self.entries.get(cache_key)?.clone();
        self.lru.push_back(cache_key.to_string());
        Some(path)
    }
}
