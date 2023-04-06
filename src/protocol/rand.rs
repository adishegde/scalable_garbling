use super::network;
use super::network::{NetworkChannelBuilder, Recipient, SendMessage};
use super::{MPCContext, ProtocolID};
use crate::math;
use crate::math::galois::{GFElement, GF};
use crate::sharing::{PackedShare, PackedSharing};
use crate::PartyID;
use rand::thread_rng;
use std::sync::Arc;

#[derive(Clone)]
pub struct RandContext {
    super_inv: Arc<Vec<Vec<GFElement>>>,
    pss: Arc<PackedSharing>,
    gf: Arc<GF>,
    net_builder: NetworkChannelBuilder,
    n: usize,
}

impl RandContext {
    pub fn new(mpc_context: &MPCContext) -> Self {
        Self {
            super_inv: Arc::new(math::super_inv_matrix(
                mpc_context.n,
                mpc_context.n - mpc_context.t,
                mpc_context.gf.as_ref(),
            )),
            pss: mpc_context.pss.clone(),
            gf: mpc_context.gf.clone(),
            net_builder: mpc_context.net_builder.clone(),
            n: mpc_context.n,
        }
    }
}

pub async fn rand(id: ProtocolID, context: RandContext) -> Vec<PackedShare> {
    let chan = context.net_builder.channel(&id).await;
    let shares = {
        let mut rng = thread_rng();
        context.pss.rand(context.gf.as_ref(), &mut rng)
    };

    for (i, share) in shares.into_iter().enumerate() {
        chan.send(SendMessage {
            to: Recipient::One(i as PartyID),
            proto_id: id.clone(),
            data: context.gf.serialize_element(&share),
        })
        .await;
    }

    let sent_shares: Vec<_> = network::message_from_each_party(chan.receiver(), context.n)
        .await
        .into_iter()
        .map(|d| context.gf.deserialize_element(&d))
        .collect();

    context
        .super_inv
        .iter()
        .map(|row| crate::utils::iprod(row.iter(), sent_shares.iter(), context.gf.as_ref()))
        .collect()
}
