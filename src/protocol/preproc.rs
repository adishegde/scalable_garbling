use super::core;
use super::network;
use super::network::{Network, Recipient, SendMessage};
use super::{MPCContext, ProtocolID, ProtocolIDBuilder};
use crate::circuit::PackedCircuit;
use crate::math::galois::GF;
use crate::sharing::{PackedShare, PackedSharing};
use crate::PartyID;
use bincode::{deserialize, serialize};
use ndarray::{parallel::prelude::*, s, Array1, Array2, Array3, ArrayView1};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::task::spawn;

fn num_batches(num: usize, batch_size: usize) -> usize {
    (num + batch_size - 1) / batch_size
}

#[derive(Clone)]
pub struct RandContext<const W: u8> {
    super_inv: Arc<Array2<GF<W>>>,
    pss: Arc<PackedSharing<W>>,
    n: usize,
}

impl<const W: u8> RandContext<W> {
    pub fn new(super_inv_matrix: Arc<Array2<GF<W>>>, mpc_context: &MPCContext<W>) -> Self {
        Self {
            super_inv: super_inv_matrix,
            pss: mpc_context.pss.clone(),
            n: mpc_context.n,
        }
    }
}

pub async fn rand<const W: u8>(
    id: ProtocolID,
    num: usize,
    context: RandContext<W>,
    net: Network,
) -> Array1<PackedShare<W>> {
    let batch_size = context.super_inv.shape()[0];
    let batches = num_batches(num, batch_size);

    // Secret sharing random values.
    let shares: Vec<_> = (0..batches)
        .into_par_iter()
        .map_init(rand::thread_rng, |rng, _| context.pss.rand(rng))
        .flatten_iter()
        .collect();
    let shares = Array2::from_shape_vec((batches, context.n), shares).unwrap();

    for i in 0..context.n {
        net.send(SendMessage {
            to: Recipient::One(i as PartyID),
            proto_id: id.clone(),
            data: serialize(&shares.slice(s![.., i])).unwrap(),
        })
        .await;
    }

    // Receives shares from all parties.
    let sent_shares: Vec<_> = network::message_from_each_party(id.clone(), &net, context.n)
        .await
        .into_par_iter()
        .flat_map(|d| deserialize::<Array1<GF<W>>>(&d).unwrap().to_vec())
        .collect();
    let sent_shares = Array2::from_shape_vec((context.n, batches), sent_shares).unwrap();

    context
        .super_inv
        .dot(&sent_shares)
        .into_shape(batches * batch_size)
        .unwrap()
}

#[derive(Clone)]
pub struct ZeroContext<const W: u8> {
    super_inv: Arc<Array2<GF<W>>>,
    pss_n: Arc<PackedSharing<W>>,
    n: usize,
}

impl<const W: u8> ZeroContext<W> {
    pub fn new(super_inv_matrix: Arc<Array2<GF<W>>>, mpc_context: &MPCContext<W>) -> Self {
        Self {
            super_inv: super_inv_matrix,
            pss_n: mpc_context.pss_n.clone(),
            n: mpc_context.n,
        }
    }
}

pub async fn zero<const W: u8>(
    id: ProtocolID,
    num: usize,
    context: ZeroContext<W>,
    net: Network,
) -> Array1<PackedShare<W>> {
    let batch_size = context.super_inv.shape()[0];
    let batches = num_batches(num, batch_size);

    let secrets = Array1::zeros(batches);

    let shares: Vec<_> = (0..batches)
        .into_par_iter()
        .map_init(rand::thread_rng, |rng, _| {
            context.pss_n.share(secrets.view(), rng)
        })
        .flatten_iter()
        .collect();
    let shares = Array2::from_shape_vec((batches, context.n), shares).unwrap();

    for i in 0..context.n {
        net.send(SendMessage {
            to: Recipient::One(i as PartyID),
            proto_id: id.clone(),
            data: serialize(&shares.slice(s![.., i])).unwrap(),
        })
        .await;
    }

    let sent_shares: Vec<_> = network::message_from_each_party(id.clone(), &net, context.n)
        .await
        .into_par_iter()
        .flat_map(|d| deserialize::<Array1<GF<W>>>(&d).unwrap().to_vec())
        .collect();
    let sent_shares = Array2::from_shape_vec((context.n, batches), sent_shares).unwrap();

    context
        .super_inv
        .dot(&sent_shares)
        .into_shape(batches * batch_size)
        .unwrap()
}

#[derive(Clone)]
pub struct RandBitContext<const W: u8> {
    bin_super_inv: Arc<Array2<GF<W>>>,
    pss: Arc<PackedSharing<W>>,
    n: usize,
}

impl<const W: u8> RandBitContext<W> {
    pub fn new(binary_supinv_matrix: Arc<Array2<GF<W>>>, mpc_context: &MPCContext<W>) -> Self {
        Self {
            bin_super_inv: binary_supinv_matrix,
            pss: mpc_context.pss.clone(),
            n: mpc_context.n,
        }
    }
}

pub async fn randbit<const W: u8>(
    id: ProtocolID,
    num: usize,
    context: RandBitContext<W>,
    net: Network,
) -> Array1<PackedShare<W>> {
    let batch_size = context.bin_super_inv.shape()[0];
    let batches = num_batches(num, batch_size);

    let shares: Vec<_> = (0..batches)
        .into_par_iter()
        .map_init(rand::thread_rng, |rng, _| {
            let secrets: Vec<_> = (0..context.pss.num_secrets())
                .map(|_| if rng.gen::<bool>() { GF::ONE } else { GF::ZERO })
                .collect();
            context.pss.share(ArrayView1::from(&secrets), rng)
        })
        .flatten_iter()
        .collect();
    let shares = Array2::from_shape_vec((batches, context.n), shares).unwrap();

    for i in 0..context.n {
        net.send(SendMessage {
            to: Recipient::One(i as PartyID),
            proto_id: id.clone(),
            data: serialize(&shares.slice(s![.., i])).unwrap(),
        })
        .await;
    }

    let sent_shares: Vec<_> = network::message_from_each_party(id.clone(), &net, context.n)
        .await
        .into_par_iter()
        .flat_map(|d| deserialize::<Array1<GF<W>>>(&d).unwrap().to_vec())
        .collect();
    let sent_shares = Array2::from_shape_vec((context.n, batches), sent_shares).unwrap();

    context
        .bin_super_inv
        .dot(&sent_shares)
        .into_shape(batches * batch_size)
        .unwrap()
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PreProc<const W: u8> {
    // Mask for each packed gate
    pub masks: Array1<PackedShare<W>>,
    // keys[g][b][l] gives the l-th index of the b-label LPN key for the g-th gate.
    pub keys: Array3<GF<W>>,
    pub randoms: VecDeque<PackedShare<W>>,
    pub zeros: VecDeque<PackedShare<W>>,
    pub errors: VecDeque<PackedShare<W>>,
}

#[derive(Clone, Copy)]
pub struct PreProcCount {
    pub masks: usize,
    pub keys: (usize, usize, usize),
    pub randoms: usize,
    pub zeros: usize,
    pub errors: usize,
}

impl PreProcCount {
    fn new<const W: u8>(circ: &PackedCircuit, context: &MPCContext<W>) -> Self {
        let num_circ_inp_blocks = circ.inputs().len();
        let num_gate_blocks = circ.gates().len();
        let num_blocks = num_circ_inp_blocks + num_gate_blocks;

        let (num_and, num_xor, num_inv) = circ.get_gate_counts();
        let num_inp_blocks = num_and * 2 + num_xor * 2 + num_inv;
        let num_trans_per_wire = 2 * context.lpn_key_len + 1;
        let num_rtrans = (num_batches(num_blocks, context.l)
            + num_batches(num_inp_blocks, context.l))
            * num_trans_per_wire;
        let num_mult = num_and * (4 * context.lpn_mssg_len + 1)
            + num_xor * 4 * context.lpn_mssg_len
            + num_inv * 2 * context.lpn_mssg_len;

        let num_rand = num_rtrans * (context.n + context.t) + num_mult;
        let num_zeros = num_rtrans * 2 * context.n + num_mult;
        let num_errors = num_and * 4 * context.lpn_mssg_len
            + num_xor * 4 * context.lpn_mssg_len
            + num_inv * 2 * context.lpn_mssg_len;

        Self {
            masks: num_blocks,
            keys: (num_blocks, 2, context.lpn_key_len),
            randoms: num_rand,
            zeros: num_zeros,
            errors: num_errors,
        }
    }
}

impl<const W: u8> PreProc<W> {
    pub fn describe(circ: &PackedCircuit, context: &MPCContext<W>) -> PreProcCount {
        PreProcCount::new(circ, context)
    }

    pub fn dummy(id: PartyID, seed: u64, circ: &PackedCircuit, context: &MPCContext<W>) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut rand_gen = {
            let small_pool: Vec<_> = (0..100)
                .map(|_| context.pss.rand(&mut rng)[id as usize])
                .collect();
            let mut iter = small_pool.into_iter().cycle();
            move || iter.next().unwrap()
        };

        let mut zero_n_gen = {
            let secrets = Array1::zeros(context.l);
            let small_pool: Vec<_> = (0..100)
                .map(|_| context.pss_n.share(secrets.view(), &mut rng)[id as usize])
                .collect();
            let mut iter = small_pool.into_iter().cycle();
            move || iter.next().unwrap()
        };

        let bit_gen = {
            let small_pool: Vec<_> = (0..100)
                .map(|_| {
                    let secrets: Vec<_> = (0..context.l)
                        .map(|_| if rng.gen::<bool>() { GF::ONE } else { GF::ZERO })
                        .collect();
                    context.pss.share(ArrayView1::from(&secrets), &mut rng)[id as usize]
                })
                .collect();
            let mut iter = small_pool.into_iter().cycle();
            move || iter.next().unwrap()
        };

        let counts = PreProcCount::new(circ, context);

        Self {
            masks: Array1::from_shape_simple_fn(counts.masks, bit_gen),
            keys: Array3::from_shape_simple_fn(counts.keys, &mut rand_gen),
            randoms: VecDeque::from_iter(
                std::iter::repeat_with(&mut rand_gen).take(counts.randoms),
            ),
            zeros: VecDeque::from_iter(std::iter::repeat_with(&mut zero_n_gen).take(counts.zeros)),
            errors: VecDeque::from_iter(std::iter::repeat_with(&mut rand_gen).take(counts.errors)),
        }
    }
}

pub async fn preproc<const W: u8>(
    id: ProtocolID,
    desc: PreProcCount,
    context: MPCContext<W>,
    rcontext: RandContext<W>,
    zcontext: ZeroContext<W>,
    bcontext: RandBitContext<W>,
    net: Network,
) -> PreProc<W> {
    // TODO: Implement for general bais parameter.
    debug_assert_eq!(context.lpn_tau, 2);

    let mut id_gen = ProtocolIDBuilder::new(&id, 4 + context.n as u64);

    let rand_fut = {
        let num_rand_shares =
            desc.keys.0 * desc.keys.1 * desc.keys.2 + desc.randoms + 3 * desc.errors;
        spawn(rand(
            id_gen.next().unwrap(),
            num_rand_shares,
            rcontext,
            net.clone(),
        ))
    };

    let zero_fut = {
        let num_zero_shares = desc.zeros + 2 * desc.errors;
        spawn(zero(
            id_gen.next().unwrap(),
            num_zero_shares,
            zcontext,
            net.clone(),
        ))
    };

    let rbit_fut = {
        let num_rbit_shares = desc.masks + 2 * desc.errors;
        spawn(randbit(
            id_gen.next().unwrap(),
            num_rbit_shares,
            bcontext,
            net.clone(),
        ))
    };

    let mut rand_shares: VecDeque<_> = rand_fut.await.unwrap().to_vec().into();
    let mut zero_shares: VecDeque<_> = zero_fut.await.unwrap().to_vec().into();
    let mut rbit_shares: VecDeque<_> = rbit_fut.await.unwrap().to_vec().into();

    let mut error_rand: VecDeque<_> = rand_shares.drain(..(3 * desc.errors)).collect();
    let mut error_zeros: VecDeque<_> = zero_shares.drain(..(2 * desc.errors)).collect();
    let mut error_rbits: VecDeque<_> = rbit_shares.drain(..(2 * desc.errors)).collect();

    let error_shares = {
        let batch_size = desc.errors / context.n;

        let mut handles = Vec::with_capacity(context.n);
        for (leader, bsize) in std::iter::repeat(batch_size)
            .take(context.n - 1)
            .chain(std::iter::once(desc.errors - (context.n - 1) * batch_size))
            .enumerate()
        {
            let rbit0 = Array1::from_vec(error_rbits.drain(..bsize).collect());
            let rbit1 = Array1::from_vec(error_rbits.drain(..bsize).collect());
            let mut brand: VecDeque<_> = error_rand.drain(..(3 * bsize)).collect();
            let mut bzeros: VecDeque<_> = error_zeros.drain(..(2 * bsize)).collect();
            let context = context.clone();
            let net = net.clone();
            let leader: PartyID = leader.try_into().unwrap();
            let err_id = id_gen.next().unwrap();

            handles.push(spawn(async move {
                let rands = Array1::from_vec(brand.drain(..bsize).collect());
                let zeros = Array1::from_vec(bzeros.drain(..bsize).collect());

                let biased_bits = core::mult(
                    err_id.clone(),
                    rbit0,
                    rbit1,
                    rands,
                    zeros,
                    Some(leader),
                    context.clone(),
                    net.clone(),
                )
                .await;

                let scalars = Array1::from_vec(brand.drain(..bsize).collect());
                let rand = Array1::from_vec(brand.into());
                let zeros = Array1::from_vec(bzeros.into());

                core::mult(
                    err_id,
                    biased_bits,
                    scalars,
                    rand,
                    zeros,
                    Some(leader),
                    context,
                    net,
                )
                .await
            }));
        }

        let mut errors = Vec::with_capacity(desc.errors);
        for handle in handles {
            errors.extend(handle.await.unwrap().into_iter());
        }
        errors.into()
    };

    let key_shares = rand_shares
        .drain(..(desc.keys.0 * desc.keys.1 * desc.keys.2))
        .collect();
    rbit_shares.truncate(desc.masks);

    PreProc {
        masks: Array1::from_vec(rbit_shares.into()),
        keys: Array3::from_shape_vec(desc.keys, key_shares).unwrap(),
        randoms: rand_shares,
        zeros: zero_shares,
        errors: error_shares,
    }
}
