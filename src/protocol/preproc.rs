use super::core;
use super::network;
use super::network::{NetworkChannelBuilder, Recipient, SendMessage};
use super::{MPCContext, ProtocolID, ProtocolIDBuilder};
use crate::circuit::{PackedCircuit, PackedGate};
use crate::math::galois::{GFMatrix, GF};
use crate::math::utils;
use crate::sharing::{PackedShare, PackedSharing};
use crate::PartyID;
use futures_lite::stream::StreamExt;
use rand::{thread_rng, Rng, SeedableRng};
use std::sync::Arc;
use tokio::spawn;

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
    let mut chan = context.net_builder.channel(id.clone());
    let shares = {
        let mut rng = thread_rng();
        context.pss.rand(context.gf.as_ref(), &mut rng)
    };

    for (i, share) in shares.into_iter().enumerate() {
        chan.send(SendMessage {
            to: Recipient::One(i as PartyID),
            proto_id: id.clone(),
            data: context.gf.serialize_element(&share),
        });
    }

    let sent_shares: Vec<_> = network::message_from_each_party(&mut chan, context.n)
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
    let mut chan = context.net_builder.channel(id.clone());
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
        });
    }

    let sent_shares: Vec<_> = network::message_from_each_party(&mut chan, context.n)
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

pub async fn randbit(id: ProtocolID, context: RandBitContext) -> Vec<PackedShare> {
    let mut chan = context.net_builder.channel(id.clone());
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
        });
    }

    let sent_shares: Vec<_> = network::message_from_each_party(&mut chan, context.n)
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

#[derive(Clone)]
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
        let mut rand_gen = {
            let small_pool: Vec<_> = (0..100)
                .map(|_| context.pss.rand(context.gf.as_ref(), &mut rng)[id as usize])
                .collect();
            let mut iter = small_pool.into_iter().cycle();
            move || iter.next().unwrap()
        };
        let mut zero_n_gen = {
            let secrets = vec![context.gf.zero(); context.l];
            let small_pool: Vec<_> = (0..100)
                .map(|_| context.pss.share_n(&secrets, context.gf.as_ref(), &mut rng)[id as usize])
                .collect();
            let mut iter = small_pool.into_iter().cycle();
            move || iter.next().unwrap()
        };
        let mut bit_gen = {
            let small_pool: Vec<_> = (0..100)
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
            let mut iter = small_pool.into_iter().cycle();
            move || iter.next().unwrap()
        };

        let masks = (0..num_blocks).map(|_| bit_gen()).collect();
        let keys = [(); 2].map(|_| {
            (0..context.lpn_key_len)
                .map(|_| (0..num_blocks).map(|_| rand_gen()).collect())
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

        let randoms = (0..num_rand).map(|_| rand_gen()).collect();
        let zeros = (0..num_zeros).map(|_| zero_n_gen()).collect();
        let errors = (0..num_errors).map(|_| bit_gen()).collect();

        Self {
            masks,
            keys,
            randoms,
            zeros,
            errors,
        }
    }
}

pub async fn preproc(
    id: ProtocolID,
    num_circ_inp_blocks: usize,
    num_and_blocks: usize,
    num_xor_blocks: usize,
    num_inv_blocks: usize,
    context: MPCContext,
    rcontext: RandContext,
    zcontext: ZeroContext,
    bcontext: RandBitContext,
) -> PreProc {
    assert_eq!(context.lpn_tau, 2);

    let num_batches = |num: usize, batch_size: usize| (num + batch_size - 1) / batch_size;

    let num_gate_block = num_and_blocks + num_xor_blocks + num_inv_blocks;
    let num_blocks = num_circ_inp_blocks + num_gate_block;
    let num_inp_blocks = num_and_blocks * 2 + num_xor_blocks * 2 + num_inv_blocks;

    let num_trans_per_wire = 2 * context.lpn_key_len + 1;
    let num_rtrans = (num_batches(num_blocks, context.l) + num_batches(num_inp_blocks, context.l))
        * num_trans_per_wire;
    let num_mult = num_and_blocks * (4 * context.lpn_mssg_len + 1)
        + num_xor_blocks * 4 * context.lpn_mssg_len
        + num_inv_blocks * 2 * context.lpn_mssg_len;

    let num_masks = num_blocks;
    let num_keys = num_blocks * 2 * context.lpn_key_len;
    let num_rand_preproc = num_rtrans * (context.n + context.t) + num_mult;
    let num_zeros_preproc = num_rtrans * 2 * context.n + num_mult;

    let num_errors = num_and_blocks * 4 * context.lpn_mssg_len
        + num_xor_blocks * 4 * context.lpn_mssg_len
        + num_inv_blocks * 2 * context.lpn_mssg_len;

    let num_rand_batches = num_batches(
        num_keys + num_rand_preproc + num_errors,
        context.n - context.t,
    );
    let num_zero_batches = num_batches(num_zeros_preproc + num_errors, context.n - context.t);
    let num_rbit_batches = num_batches(num_masks + 2 * num_errors, bcontext.bin_super_inv.len());

    let num_subproto = num_rand_batches + num_zero_batches + num_rbit_batches + 2;
    let mut id_gen = ProtocolIDBuilder::new(&id, num_subproto as u64);

    // Compute base preproc material.
    let mut rbit_shares = {
        let mut handles = Vec::with_capacity(num_rbit_batches);
        for _ in 0..num_rbit_batches {
            let sub_id = id_gen.next().unwrap();
            handles.push(spawn(randbit(sub_id, bcontext.clone())));
        }

        futures_lite::stream::iter(handles)
            .then(|fut| async { fut.await.unwrap() })
            .map(|shares| futures_lite::stream::iter(shares))
            .flatten()
            .boxed()
    };

    let mut rand_shares = {
        let mut handles = Vec::with_capacity(num_rand_batches);
        for _ in 0..num_rand_batches {
            let sub_id = id_gen.next().unwrap();
            handles.push(spawn(rand(sub_id, rcontext.clone())));
        }

        futures_lite::stream::iter(handles)
            .then(|fut| async { fut.await.unwrap() })
            .map(|shares| futures_lite::stream::iter(shares))
            .flatten()
            .boxed()
    };

    let mut zero_shares = {
        let mut handles = Vec::with_capacity(num_zero_batches);
        for _ in 0..num_zero_batches {
            let sub_id = id_gen.next().unwrap();
            handles.push(spawn(zero(sub_id, zcontext.clone())));
        }

        futures_lite::stream::iter(handles)
            .then(|fut| async { fut.await.unwrap() })
            .map(|shares| futures_lite::stream::iter(shares))
            .flatten()
            .boxed()
    };

    let errors = {
        let mut handles = Vec::new();
        for _ in 0..num_errors {
            let x = rbit_shares.next().await.unwrap();
            let y = rbit_shares.next().await.unwrap();
            let r = rand_shares.next().await.unwrap();
            let z = zero_shares.next().await.unwrap();
            let sub_id = id_gen.next().unwrap();

            handles.push(spawn(core::mult(sub_id, x, y, r, z, context.clone())));
        }
        futures_lite::stream::iter(handles).then(|fut| async { fut.await.unwrap() })
    };

    let masks = {
        let mut res = Vec::with_capacity(num_masks);
        for _ in 0..num_masks {
            res.push(rbit_shares.next().await.unwrap());
        }

        res
    };

    let keys = {
        let mut keys = [
            Vec::with_capacity(context.lpn_key_len),
            Vec::with_capacity(context.lpn_key_len),
        ];

        for i in 0..2 {
            for _ in 0..context.lpn_key_len {
                let mut key = Vec::with_capacity(num_blocks);

                for _ in 0..num_blocks {
                    key.push(rand_shares.next().await.unwrap());
                }

                keys[i].push(key);
            }
        }

        keys
    };

    PreProc {
        masks,
        keys,
        randoms: rand_shares.collect().await,
        zeros: zero_shares.collect().await,
        errors: errors.collect().await,
    }
}
