use crate::math::galois::GF;
use crate::sharing::PackedSharing;
use crate::PartyID;
use smol::channel::{unbounded, Receiver, Sender};
use smol::lock::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub mod core;
pub mod garble;
pub mod network;
pub mod preproc;

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

    pub fn receiver(&self) -> Receiver<R> {
        self.receiver.clone()
    }
}

pub struct ProtoChannelBuilder<S, R> {
    worker_s: Sender<S>,
    clients: Arc<RwLock<HashMap<ProtocolID, (Sender<R>, Receiver<R>)>>>,
}

impl<S, R> ProtoChannelBuilder<S, R> {
    fn new(worker_task_s: Sender<S>) -> Self {
        Self {
            worker_s: worker_task_s,
            clients: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn create_channel(&self, id: ProtocolID) -> (Sender<R>, Receiver<R>) {
        let mut client_map = self.clients.write().await;
        match client_map.get(&id) {
            Some((sender, receiver)) => (sender.clone(), receiver.clone()),
            None => {
                let (sender, receiver) = unbounded();
                client_map.insert(id, (sender.clone(), receiver.clone()));
                (sender, receiver)
            }
        }
    }

    pub async fn channel(&self, id: &ProtocolID) -> ProtoChannel<S, R> {
        {
            let client_map = self.clients.read().await;
            if let Some((_, receiver)) = client_map.get(id) {
                return ProtoChannel {
                    sender: self.worker_s.clone(),
                    receiver: receiver.clone(),
                };
            }
        }

        let (_, receiver) = self.create_channel(id.clone()).await;

        ProtoChannel {
            sender: self.worker_s.clone(),
            receiver,
        }
    }

    async fn receiver_handle(&self, id: &ProtocolID) -> Sender<R> {
        {
            let client_map = self.clients.read().await;
            if let Some((sender, _)) = client_map.get(id) {
                return sender.clone();
            }
        }

        let (sender, _) = self.create_channel(id.clone()).await;
        sender
    }
}

impl<S, R> Clone for ProtoChannelBuilder<S, R> {
    fn clone(&self) -> Self {
        Self {
            worker_s: self.worker_s.clone(),
            clients: self.clients.clone(),
        }
    }
}

#[derive(Clone)]
pub struct MPCContext {
    pub id: PartyID,         // ID of party
    pub n: usize,            // Number of parties
    pub t: usize,            // Threshold of corrupt parties
    pub l: usize,            // Packing parameter i.e., number of secrets per share
    pub lpn_tau: usize,      // LPN error parameter; Bernoulli errors with bias 2^{-lnp_tau}
    pub lpn_key_len: usize,  // LPN key length
    pub lpn_mssg_len: usize, // LPN message/expanded length
    pub gf: Arc<GF>,
    pub pss: Arc<PackedSharing>,
    pub net_builder: network::NetworkChannelBuilder,
}
