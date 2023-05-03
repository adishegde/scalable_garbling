use crate::math::galois::GF;
use crate::sharing::PackedSharing;
use crate::PartyID;
use std::sync::Arc;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

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
        let num_bytes = (num.next_power_of_two().ilog2() + 7) / 8;
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
    sender: UnboundedSender<S>,
    receiver: UnboundedReceiver<R>,
}

impl<S: std::fmt::Debug, R> ProtoChannel<S, R> {
    pub fn send(&self, message: S) {
        self.sender
            .send(message)
            .expect("Channel sender to be open.");
    }

    pub async fn recv(&mut self) -> R {
        self.receiver
            .recv()
            .await
            .expect("Channel receiver to be open.")
    }
}

#[derive(Clone)]
pub struct ProtoChannelBuilder {
    worker_s: UnboundedSender<network::SendMessage>,
    register_s: UnboundedSender<network::RegisterProtocol>,
}

impl ProtoChannelBuilder {
    fn new(
        worker_s: UnboundedSender<network::SendMessage>,
        register_s: UnboundedSender<network::RegisterProtocol>,
    ) -> Self {
        Self {
            worker_s,
            register_s,
        }
    }

    pub fn channel(
        &self,
        id: ProtocolID,
    ) -> ProtoChannel<network::SendMessage, network::ReceivedMessage> {
        let (sender, receiver) = unbounded_channel();

        self.register_s
            .send(network::RegisterProtocol(id, sender))
            .unwrap();

        ProtoChannel {
            sender: self.worker_s.clone(),
            receiver,
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
