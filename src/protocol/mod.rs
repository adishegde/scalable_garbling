use crate::sharing::PackedSharing;
use crate::PartyID;
use std::fmt::Debug;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;

pub mod core;
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

#[derive(Clone)]
pub struct ProtoHandle<S: Debug, R: Debug> {
    tx_send: UnboundedSender<S>,
    tx_recv: UnboundedSender<(ProtocolID, oneshot::Sender<R>)>,
}

impl<S: Debug, R: Debug> ProtoHandle<S, R> {
    pub fn new(
        tx_send: UnboundedSender<S>,
        tx_recv: UnboundedSender<(ProtocolID, oneshot::Sender<R>)>,
    ) -> Self {
        Self { tx_send, tx_recv }
    }

    pub async fn send(&self, mssg: S) {
        self.tx_send.send(mssg).unwrap();
    }

    pub async fn recv(&self, id: ProtocolID) -> R {
        let (tx, rx) = oneshot::channel();
        self.tx_recv.send((id, tx)).unwrap();
        rx.await.unwrap()
    }
}

#[derive(Clone)]
pub struct MPCContext<const W: u8> {
    pub id: PartyID,         // ID of party
    pub n: usize,            // Number of parties
    pub t: usize,            // Threshold of corrupt parties
    pub l: usize,            // Packing parameter i.e., number of secrets per share
    pub lpn_tau: usize,      // LPN error parameter; Bernoulli errors with bias 2^{-lnp_tau}
    pub lpn_key_len: usize,  // LPN key length
    pub lpn_mssg_len: usize, // LPN message/expanded length
    pub pss: Arc<PackedSharing<W>>,
    pub pss_n: Arc<PackedSharing<W>>,
}
