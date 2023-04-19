use super::core;
use super::network;
use super::network::{NetworkChannelBuilder, Recipient, SendMessage};
use super::{MPCContext, ProtocolID};
use crate::math::galois::{GFMatrix, GF};
use crate::math::utils;
use crate::sharing::{PackedShare, PackedSharing};
use crate::PartyID;
use rand::{thread_rng, Rng};
use std::sync::Arc;

#[derive(Clone)]
pub struct RandContext {
    super_inv: Arc<GFMatrix>,
    pss: Arc<PackedSharing>,
    gf: Arc<GF>,
    net_builder: NetworkChannelBuilder,
    n: usize,
}

impl RandContext {
    pub fn new(super_inv_matrix: Arc<GFMatrix>, mpc_context: &MPCContext) -> Self {
        Self {
            super_inv: super_inv_matrix,
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
        .map(|row| utils::iprod(row.iter(), sent_shares.iter(), context.gf.as_ref()))
        .collect()
}

#[derive(Clone)]
pub struct ZeroContext {
    super_inv: Arc<GFMatrix>,
    pss: Arc<PackedSharing>,
    gf: Arc<GF>,
    net_builder: NetworkChannelBuilder,
    n: usize,
    l: usize,
}

impl ZeroContext {
    pub fn new(super_inv_matrix: Arc<GFMatrix>, mpc_context: &MPCContext) -> Self {
        Self {
            super_inv: super_inv_matrix,
            pss: mpc_context.pss.clone(),
            gf: mpc_context.gf.clone(),
            net_builder: mpc_context.net_builder.clone(),
            n: mpc_context.n,
            l: mpc_context.l,
        }
    }
}

pub async fn zero(id: ProtocolID, context: ZeroContext) -> Vec<PackedShare> {
    let chan = context.net_builder.channel(&id).await;
    let shares = {
        let secrets = vec![context.gf.zero(); context.l];
        let mut rng = thread_rng();
        context.pss.share_n(&secrets, context.gf.as_ref(), &mut rng)
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
        .map(|row| utils::iprod(row.iter(), sent_shares.iter(), context.gf.as_ref()))
        .collect()
}

#[derive(Clone)]
pub struct RandBitContext {
    bin_super_inv: Arc<GFMatrix>,
    pss: Arc<PackedSharing>,
    gf: Arc<GF>,
    net_builder: NetworkChannelBuilder,
    n: usize,
    l: usize,
}

impl RandBitContext {
    pub fn new(binary_supinv_matrix: Arc<GFMatrix>, mpc_context: &MPCContext) -> Self {
        Self {
            bin_super_inv: binary_supinv_matrix,
            pss: mpc_context.pss.clone(),
            gf: mpc_context.gf.clone(),
            net_builder: mpc_context.net_builder.clone(),
            n: mpc_context.n,
            l: mpc_context.l,
        }
    }
}

pub async fn randbit(id: ProtocolID, context: RandBitContext) -> Vec<PackedShare> {
    let chan = context.net_builder.channel(&id).await;
    let shares = {
        let mut rng = thread_rng();
        let secrets: Vec<_> = (0..context.l)
            .map(|_| {
                if rng.gen::<bool>() {
                    context.gf.one()
                } else {
                    context.gf.zero()
                }
            })
            .collect();
        context.pss.share(&secrets, context.gf.as_ref(), &mut rng)
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
        .bin_super_inv
        .iter()
        .map(|row| utils::iprod(row.iter(), sent_shares.iter(), context.gf.as_ref()))
        .collect()
}

pub async fn lpn_error(
    id: ProtocolID,
    randbits: Vec<PackedShare>,
    randoms: Vec<PackedShare>,
    zeros: Vec<PackedShare>,
    context: MPCContext,
) -> PackedShare {
    let mut prod = randbits[0];

    let mult_preproc_iter = randoms.into_iter().zip(zeros.into_iter());
    for (inp, (r, z)) in randbits.into_iter().skip(1).zip(mult_preproc_iter) {
        prod = core::mult(id.clone(), prod, inp, r, z, context.clone()).await;
    }

    prod
}
