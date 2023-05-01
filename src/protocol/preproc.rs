use super::core;
use super::network;
use super::network::{NetworkChannelBuilder, Recipient, SendMessage};
use super::{MPCContext, ProtocolID};
use crate::circuit::{PackedCircuit, PackedGate};
use crate::math::galois::{GFMatrix, GF};
use crate::math::utils;
use crate::sharing::{PackedShare, PackedSharing};
use crate::PartyID;
use rand::{thread_rng, Rng, SeedableRng};
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

    utils::matrix_vector_prod(
        context.super_inv.as_ref(),
        &sent_shares,
        context.gf.as_ref(),
    )
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

    utils::matrix_vector_prod(
        context.super_inv.as_ref(),
        &sent_shares,
        context.gf.as_ref(),
    )
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

pub async fn randbit(
    id: ProtocolID,
    share_coeffs: Arc<GFMatrix>,
    context: RandBitContext,
) -> Vec<PackedShare> {
    let chan = context.net_builder.channel(&id).await;
    let shares: Vec<_> = {
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
        context.pss.share_using_coeffs(
            secrets,
            share_coeffs.as_ref(),
            context.gf.as_ref(),
            &mut rng,
        )
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

    utils::matrix_vector_prod(
        context.bin_super_inv.as_ref(),
        &sent_shares,
        context.gf.as_ref(),
    )
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

pub struct PreProc {
    // Mask for each packed gate
    pub masks: Vec<PackedShare>,
    // keys[b][l][g] gives the l-th index of the b-label LPN key for the g-th gate.
    pub keys: [Vec<Vec<PackedShare>>; 2],
    pub randoms: Vec<PackedShare>,
    pub zeros: Vec<PackedShare>,
    pub errors: Vec<PackedShare>,
}

impl PreProc {
    pub fn dummy(id: PartyID, seed: u64, circ: &PackedCircuit, context: &MPCContext) -> Self {
        let num_circ_inp_blocks = circ.inputs().len();
        let num_gate_blocks = circ.gates().len();
        let num_blocks = num_circ_inp_blocks + num_gate_blocks;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let masks = (0..num_blocks)
            .map(|_| {
                let secrets: Vec<_> = (0..context.l)
                    .map(|_| {
                        if rng.gen::<bool>() {
                            context.gf.one()
                        } else {
                            context.gf.zero()
                        }
                    })
                    .collect();
                context.pss.share(&secrets, context.gf.as_ref(), &mut rng)[id as usize]
            })
            .collect();

        let keys = [(); 2].map(|_| {
            (0..context.lpn_key_len)
                .map(|_| {
                    (0..num_blocks)
                        .map(|_| context.pss.rand(context.gf.as_ref(), &mut rng)[id as usize])
                        .collect()
                })
                .collect()
        });

        let mut num_and = 0;
        let mut num_xor = 0;
        let mut num_inv = 0;

        for gate in circ.gates() {
            match gate {
                PackedGate::And(_) => num_and = num_and + 1,
                PackedGate::Xor(_) => num_xor = num_xor + 1,
                PackedGate::Inv(_) => num_inv = num_inv + 1,
            }
        }

        let num_inp_blocks = num_and * 2 + num_xor * 2 + num_inv;

        let num_trans_per_wire = 2 * context.lpn_key_len + 1;
        let num_rtrans = ((num_blocks + context.l - 1) / context.l
            + (num_inp_blocks + context.l - 1) / context.l)
            * num_trans_per_wire;
        let num_mult = num_and * (4 * context.lpn_mssg_len + 1)
            + num_xor * 4 * context.lpn_mssg_len
            + num_inv * 2 * context.lpn_mssg_len;

        let num_rand = num_rtrans * (context.n + context.t) + num_mult;
        let num_zeros = num_rtrans * 2 * context.n + num_mult;
        let num_errors = num_and * 4 * context.lpn_mssg_len
            + num_xor * 4 * context.lpn_mssg_len
            + num_inv * 2 * context.lpn_mssg_len;

        let randoms = (0..num_rand)
            .map(|_| context.pss.rand(context.gf.as_ref(), &mut rng)[id as usize])
            .collect();
        let zeros = {
            let secrets = vec![context.gf.zero(); context.l];
            (0..num_zeros)
                .map(|_| context.pss.share(&secrets, context.gf.as_ref(), &mut rng)[id as usize])
                .collect()
        };
        let errors = {
            // Errors are set to 0 to make writing tests easier (can avoid error correction when
            // decrypting).
            let secrets = vec![context.gf.zero(); context.l];
            (0..num_errors)
                .map(|_| context.pss.share(&secrets, context.gf.as_ref(), &mut rng)[id as usize])
                .collect()
        };

        Self {
            masks,
            keys,
            randoms,
            zeros,
            errors,
        }
    }
}
