use smol::channel::{unbounded, Receiver, Sender};
use smol::lock::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub mod network;

/// Unique identifier for a protocol.
pub type ProtocolID = Vec<u8>;

/// Computes unique protocol IDs.
///
/// When messages from different contexts (read protocols) are being communicated concurrently, we
/// need a mechanism to route them to make sure they get delivered to the intended context.
/// This struct computes IDs for subcontexts such that the resulting IDs are unique.
///
/// The child IDs are computed using the prefix of the parent ID.
/// Thus, the order in which subcontexts are created is important and should be preserved for the
/// mapping to hold.
///
/// The builder is valid only as long as it's parent ID exists.
pub struct ProtocolIDBuilder<'a> {
    id: &'a ProtocolID,
    suffix: Vec<u8>,
}

impl<'a> ProtocolIDBuilder<'a> {
    /// num is an upper bound on the number of children created.
    pub fn new(parent_id: &'a ProtocolID, num: u64) -> Self {
        let num_bytes = (num.next_power_of_two().ilog2() + 2) / 3;
        ProtocolIDBuilder {
            id: parent_id,
            suffix: vec![0; num_bytes.try_into().unwrap()],
        }
    }

    fn increment_bytes(&mut self, idx: usize) -> Result<(), ()> {
        if idx >= self.suffix.len() {
            return Err(());
        }

        if self.suffix[idx] == u8::MAX {
            self.suffix[idx] = 0;
            return self.increment_bytes(idx + 1);
        }

        self.suffix[idx] += 1;
        Ok(())
    }
}

impl<'a> Iterator for ProtocolIDBuilder<'a> {
    type Item = ProtocolID;

    /// Returns the next child ID.
    /// Number of child IDs that can be generated might not be equal to the bound used in new.
    fn next(&mut self) -> Option<Self::Item> {
        let mut child_id = self.id.to_vec();
        child_id.extend_from_slice(&self.suffix);

        match self.increment_bytes(0) {
            Ok(_) => Some(child_id),
            Err(_) => None,
        }
    }
}

pub struct ProtoChannel<S, R> {
    sender: Sender<S>,
    receiver: Receiver<R>,
}

impl<S, R> ProtoChannel<S, R> {
    pub async fn send(&self, message: S) {
        self.sender
            .send(message)
            .await
            .expect("Channel sender to be open.");
    }

    pub async fn recv(&self) -> R {
        self.receiver
            .recv()
            .await
            .expect("Channel receiver to be open.")
    }

    pub fn close(&self) {
        self.sender.close();
        self.receiver.close();
    }
}

pub struct ProtoChannelBuilder<S, R> {
    worker_s: Sender<S>,
    receiver_s: Arc<RwLock<HashMap<ProtocolID, Sender<R>>>>,
    receiver_r: Arc<RwLock<HashMap<ProtocolID, Receiver<R>>>>,
}

impl<S, R> ProtoChannelBuilder<S, R> {
    fn new(worker_task_s: Sender<S>) -> Self {
        Self {
            worker_s: worker_task_s,
            receiver_s: Arc::new(RwLock::new(HashMap::new())),
            receiver_r: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn create_channel(&self, id: ProtocolID) -> (Sender<R>, Receiver<R>) {
        let (sender, receiver) = unbounded();

        {
            let mut sender_map = self.receiver_s.write().await;
            sender_map.insert(id.clone(), sender.clone());
        }

        {
            let mut receiver_map = self.receiver_r.write().await;
            receiver_map.insert(id, receiver.clone());
        }

        (sender, receiver)
    }

    pub async fn channel(&self, id: &ProtocolID) -> ProtoChannel<S, R> {
        let receiver_map = self.receiver_r.read().await;
        if let Some(receiver) = receiver_map.get(id) {
            return ProtoChannel {
                sender: self.worker_s.clone(),
                receiver: receiver.clone(),
            };
        }
        std::mem::drop(receiver_map);

        let (_, receiver) = self.create_channel(id.clone()).await;

        ProtoChannel {
            sender: self.worker_s.clone(),
            receiver,
        }
    }

    async fn receiver_handle(&self, id: &ProtocolID) -> Sender<R> {
        let sender_map = self.receiver_s.read().await;
        if let Some(sender) = sender_map.get(id) {
            return sender.clone();
        }
        std::mem::drop(sender_map);

        let (sender, _) = self.create_channel(id.clone()).await;
        sender
    }
}

impl<S, R> Clone for ProtoChannelBuilder<S, R> {
    fn clone(&self) -> Self {
        Self {
            worker_s: self.worker_s.clone(),
            receiver_s: self.receiver_s.clone(),
            receiver_r: self.receiver_r.clone(),
        }
    }
}
