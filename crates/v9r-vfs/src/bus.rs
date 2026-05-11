use tokio::sync::broadcast;

use v9r_core::{CapabilityId, NodeId, VfsPath};

/// Emitted whenever a non-synthetic file is written, or whenever a synthetic
/// node returns `WriteOutcome::Emit`. The orchestrator subscribes to these.
#[derive(Clone, Debug)]
pub struct WriteEvent {
    pub path: VfsPath,
    pub node: NodeId,
    pub bytes_len: usize,
    pub writer: CapabilityId,
}

/// Lossy broadcast channel for write events. Slow consumers get
/// `RecvError::Lagged` rather than back-pressuring writers.
pub struct WriteBus {
    tx: broadcast::Sender<WriteEvent>,
}

impl WriteBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<WriteEvent> {
        self.tx.subscribe()
    }

    /// Send an event. If there are no live subscribers, the event is dropped
    /// silently — that's the right behavior for a fan-out audit channel.
    pub fn publish(&self, ev: WriteEvent) {
        let _ = self.tx.send(ev);
    }
}

impl Default for WriteBus {
    fn default() -> Self {
        Self::new(1024)
    }
}
