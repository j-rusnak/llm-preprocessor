use std::sync::atomic::{AtomicBool, Ordering};

pub struct StreamSink {
    disconnected: AtomicBool,
}

impl StreamSink {
    pub fn abort_on_client_disconnect(&self) -> bool {
        self.disconnected.swap(true, Ordering::SeqCst)
    }

    pub fn should_stop_upstream(&self) -> bool {
        self.disconnected.load(Ordering::SeqCst)
    }
}
