use super::core;
use super::core::{RandSharingTransform, SharingTransform};
use super::network::Network;
use super::preproc::PreProc;
use super::{MPCContext, ProtocolID, ProtocolIDBuilder};
use crate::circuit::{PackedCircuit, PackedGate, PackedGateInfo, WireID};
use crate::math;
use crate::math::galois::GF;
use crate::math::Combination;
use crate::sharing::{PackedShare, PackedSharing};
use ndarray::{
    array, parallel::prelude::*, s, ArcArray, Array1, Array2, Array3, ArrayView2, Axis, Ix3,
};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::task::spawn;

pub struct GarbleContext<const W: u8> {
    tf_out: Arc<Transforms<W>>,
    tf_inp: Arc<Transforms<W>>,
    lookup: HashMap<WireID, usize>,
    lpn_enc: Array2<GF<W>>,
    lpn_mat: LPNMatrix<W>,
    circ: PackedCircuit,
    mpc: MPCContext<W>,
}

impl<const W: u8> GarbleContext<W> {
    pub fn new(circ: PackedCircuit, context: MPCContext<W>) -> Self {
        let l: u32 = context.l.try_into().unwrap();
        let n: u32 = context.n.try_into().unwrap();
        let defpos = PackedSharing::default_pos(n, l);

        let tf_out = {
            let tf = circ
                .inputs()
                .iter()
                .chain(circ.gates().iter().map(|g| match g {
                    PackedGate::Xor(ginf) => &ginf.out,
                    PackedGate::And(ginf) => &ginf.out,
                    PackedGate::Inv(ginf) => &ginf.out,
                }))
                .map(|wires| {
                    let pos: Vec<_> =
                        PackedSharing::wire_to_pos(n, l, wires.iter().cloned()).collect();
                    let f = Combination::new((0..pos.len()).collect());
                    let num_pos = pos.len();
                    ((defpos[..num_pos].to_owned(), pos), f)
                })
                .collect();

            Transforms::new(tf, &context)
        };

        let tf_inp = {
            let tf = circ
                .gates()
                .iter()
                .flat_map(|g| match g {
                    PackedGate::Xor(ginf) => ginf.inp.to_vec(),
                    PackedGate::And(ginf) => ginf.inp.to_vec(),
                    PackedGate::Inv(ginf) => ginf.inp.to_vec(),
                })
                .map(|wires| {
                    let mut dedup = wires.clone();
                    dedup.sort_unstable();
                    dedup.dedup();

                    let f = Combination::from_instance(&dedup, &wires);

                    let opos: Vec<_> =
                        PackedSharing::wire_to_pos(n, l, dedup.into_iter()).collect();

                    ((opos, defpos[..wires.len()].to_owned()), f)
                })
                .collect();

            Transforms::new(tf, &context)
        };

        Self {
            tf_out: Arc::new(tf_out),
            tf_inp: Arc::new(tf_inp),
            lookup: Self::compute_wire_to_block_map(&circ),
            lpn_enc: math::rs_gen_mat(
                context.lpn_key_len + 1, // Need to encode key and mask
                context.lpn_mssg_len,
            ),
            lpn_mat: LPNMatrix::new(&circ, &context),
            circ,
            mpc: context,
        }
    }

    fn compute_wire_to_block_map(circ: &PackedCircuit) -> HashMap<WireID, usize> {
        let mut lookup = HashMap::new();

        for (i, inp_wires) in circ.inputs().iter().enumerate() {
            for inp_wire in inp_wires {
                lookup.insert(*inp_wire, i);
            }
        }

        // Input wires and gate output wires are packed separately.
        // The offset accounts for the number of input wire blocks.
        let offset = circ.inputs().len();

        for (i, gate) in circ.gates().iter().enumerate() {
            let out_wires = match gate {
                PackedGate::Xor(ginf) => &ginf.out,
                PackedGate::And(ginf) => &ginf.out,
                PackedGate::Inv(ginf) => &ginf.out,
            };

            for wire in out_wires {
                lookup.insert(*wire, i + offset);
            }
        }

        lookup
    }
}

pub struct GarbledCircuit<const W: u8> {
    pub gates: Vec<Array2<GF<W>>>,
}

pub async fn garble<const W: u8>(
    id: ProtocolID,
    mut preproc: PreProc<W>,
    context: Arc<GarbleContext<W>>,
    net: Network,
) -> GarbledCircuit<W> {
    let mpcctx = &context.mpc;
    let circ = &context.circ;
    let num_circ_inp_blocks = circ.inputs().len();
    let num_gate_blocks = circ.gates().len();
    let num_blocks = num_circ_inp_blocks + num_gate_blocks;

    // Sanity checks.
    #[cfg(debug_assertions)]
    {
        assert_eq!(preproc.masks.len(), num_blocks);
        assert_eq!(preproc.keys.shape(), [2, mpcctx.lpn_key_len, num_blocks]);
    }

    let num_trans_per_wire = 2 * mpcctx.lpn_key_len + 1;
    let mut id_gen = ProtocolIDBuilder::new(&id, 4 + num_gate_blocks as u64);

    let mut run_rtrans = |tf: Arc<Transforms<W>>| {
        let num = tf.rtrans.len() * num_trans_per_wire;
        let randoms = preproc
            .randoms
            .drain(..(num * (mpcctx.n + mpcctx.t)))
            .collect();
        let zeros = preproc.zeros.drain(..(num * 2 * mpcctx.n)).collect();
        let mpcctx = mpcctx.clone();
        let net = net.clone();
        let id = id_gen.next().unwrap();

        spawn(async move {
            tf.gen_rand_shares(id, num_trans_per_wire, randoms, zeros, mpcctx, net.clone())
                .await
        })
    };

    let tf_out_fut = run_rtrans(context.tf_out.clone());
    let tf_inp_fut = run_rtrans(context.tf_inp.clone());

    // Transform masks and keys for output wire of each gate.
    // Shape: (num_blocks, 2 * lpn_key_len + 1).
    let out_shares = {
        let (rand, rand_n) = tf_out_fut.await.unwrap();

        let mut inp = preproc
            .keys
            .clone()
            .into_shape((num_blocks, 2 * mpcctx.lpn_key_len))
            .unwrap();
        inp.push(Axis(1), preproc.masks.view()).unwrap();

        context
            .tf_out
            .transform(
                id_gen.next().unwrap(),
                inp,
                rand,
                rand_n,
                mpcctx.clone(),
                net.clone(),
            )
            .await
    };

    // Select secrets and transform shares for gate input wires.
    // Shape: (num_inp_blocks, 2 * lpn_key_len + 1).
    let inp_shares = {
        let selected: Vec<_> = circ
            .gates()
            .into_par_iter()
            .flat_map(|g| match g {
                PackedGate::Xor(ginf) => select(out_shares.view(), ginf, context.as_ref()),
                PackedGate::And(ginf) => select(out_shares.view(), ginf, context.as_ref()),
                PackedGate::Inv(ginf) => select(out_shares.view(), ginf, context.as_ref()),
            })
            .collect();

        let num_rows = selected.len() / out_shares.shape()[1];
        let selected = Array2::from_shape_vec((num_rows, out_shares.shape()[1]), selected).unwrap();

        let (rand, rand_n) = tf_inp_fut.await.unwrap();
        context
            .tf_inp
            .transform(
                id_gen.next().unwrap(),
                selected,
                rand,
                rand_n,
                mpcctx.clone(),
                net.clone(),
            )
            .await
    };

    let mut gc_handles = Vec::with_capacity(circ.gates().len());
    let mut inp_shares_iter = inp_shares.axis_iter(Axis(0));
    for (gid, gate) in circ.gates().iter().enumerate() {
        let id = id_gen.next().unwrap();
        let out_mask = preproc.masks[num_circ_inp_blocks + gid];
        let out_keys = preproc
            .keys
            .slice(s![.., .., num_circ_inp_blocks + gid])
            .to_owned();

        let fut = match gate {
            PackedGate::And(_) => {
                let (inp_masks, inp_keys) = {
                    let mut left_inp = inp_shares_iter.next().unwrap().to_vec();
                    let mut right_inp = inp_shares_iter.next().unwrap().to_vec();

                    let masks = [left_inp.pop().unwrap(), right_inp.pop().unwrap()];
                    let keys = {
                        let mut iter = left_inp.into_iter().chain(right_inp.into_iter());
                        Array3::from_shape_simple_fn((2, 2, mpcctx.lpn_key_len), move || {
                            iter.next().unwrap()
                        })
                    };

                    (masks, keys)
                };

                let randoms = preproc
                    .randoms
                    .drain(..((3 * (mpcctx.lpn_key_len + 1)) + 1))
                    .collect();
                let zeros = preproc
                    .zeros
                    .drain(..((3 * (mpcctx.lpn_key_len + 1)) + 1))
                    .collect();
                let errors = preproc.errors.drain(..(4 * mpcctx.lpn_mssg_len)).collect();

                spawn(garble_and(
                    id,
                    gid,
                    inp_masks,
                    inp_keys,
                    out_mask,
                    out_keys,
                    randoms,
                    zeros,
                    errors,
                    context.clone(),
                    net.clone(),
                ))
            }
            PackedGate::Xor(_) => {
                let (inp_masks, inp_keys) = {
                    let mut left_inp = inp_shares_iter.next().unwrap().to_vec();
                    let mut right_inp = inp_shares_iter.next().unwrap().to_vec();

                    let masks = [left_inp.pop().unwrap(), right_inp.pop().unwrap()];
                    let keys = {
                        let mut iter = left_inp.into_iter().chain(right_inp.into_iter());
                        Array3::from_shape_simple_fn((2, 2, mpcctx.lpn_key_len), move || {
                            iter.next().unwrap()
                        })
                    };

                    (masks, keys)
                };

                let randoms = preproc.randoms.drain(..(mpcctx.lpn_key_len + 1)).collect();
                let zeros = preproc.zeros.drain(..(mpcctx.lpn_key_len + 1)).collect();
                let errors = preproc.errors.drain(..(4 * mpcctx.lpn_mssg_len)).collect();

                spawn(garble_xor(
                    id,
                    gid,
                    inp_masks,
                    inp_keys,
                    out_mask,
                    out_keys,
                    randoms,
                    zeros,
                    errors,
                    context.clone(),
                    net.clone(),
                ))
            }
            PackedGate::Inv(_) => {
                let (inp_masks, inp_keys) = {
                    let mut inp = inp_shares_iter.next().unwrap().to_vec();

                    let masks = inp.pop().unwrap();
                    let keys = Array2::from_shape_vec((2, mpcctx.lpn_key_len), inp).unwrap();

                    (masks, keys)
                };

                let randoms = preproc.randoms.drain(..(mpcctx.lpn_key_len + 1)).collect();
                let zeros = preproc.zeros.drain(..(mpcctx.lpn_key_len + 1)).collect();
                let errors = preproc.errors.drain(..(2 * mpcctx.lpn_mssg_len)).collect();

                spawn(garble_inv(
                    id,
                    gid,
                    inp_masks,
                    inp_keys,
                    out_mask,
                    out_keys,
                    randoms,
                    zeros,
                    errors,
                    context.clone(),
                    net.clone(),
                ))
            }
        };

        gc_handles.push(fut);
    }

    let mut gc = Vec::with_capacity(circ.gates().len());
    for handle in gc_handles.into_iter() {
        gc.push(handle.await.unwrap());
    }

    GarbledCircuit { gates: gc }
}

async fn garble_and<const W: u8>(
    id: ProtocolID,
    gid: usize,
    inp_masks: [PackedShare<W>; 2],
    inp_keys: Array3<PackedShare<W>>, // shape: (2 (left and right), 2 (0 and 1), lpn_key_len)
    out_mask: PackedShare<W>,
    mut out_keys: Array2<PackedShare<W>>, // shape: (2, lpn_key_len)
    mut randoms: Vec<PackedShare<W>>,
    mut zeros: Vec<PackedShare<W>>,
    errors: Vec<PackedShare<W>>,
    context: Arc<GarbleContext<W>>,
    net: Network,
) -> Array2<PackedShare<W>> {
    let mpcctx = &context.mpc;
    let num_mult = 3 * (mpcctx.lpn_key_len + 1) + 1;
    let mut id_gen = ProtocolIDBuilder::new(&id, 2);

    debug_assert_eq!(inp_keys.shape(), [2, 2, mpcctx.lpn_key_len]);
    debug_assert_eq!(out_keys.shape(), [2, mpcctx.lpn_key_len]);
    debug_assert_eq!(randoms.len(), num_mult);
    debug_assert_eq!(zeros.len(), num_mult);
    debug_assert_eq!(errors.len(), 4 * mpcctx.lpn_mssg_len);

    // Append corresponding masks to keys.
    out_keys
        .push_column(Array1::from_vec(vec![out_mask, out_mask + GF::ONE]).view())
        .unwrap();

    let inp_mask_prod = core::mult(
        id_gen.next().unwrap(),
        array![inp_masks[0]],
        array![inp_masks[1]],
        array![randoms.pop().unwrap()],
        array![zeros.pop().unwrap()],
        None,
        mpcctx.clone(),
        net.clone(),
    )
    .await[0];

    // Select the message to be encrypted in each row of the garbled table.
    // Shape: (4, lpn_mssg_len)
    let mssgs = {
        let selector = {
            let s = inp_mask_prod + out_mask;
            Array1::from_iter(
                std::iter::repeat(s)
                    .take(mpcctx.lpn_key_len + 1)
                    .chain(std::iter::repeat(s + inp_masks[0]).take(mpcctx.lpn_key_len + 1))
                    .chain(std::iter::repeat(s + inp_masks[1]).take(mpcctx.lpn_key_len + 1)),
            )
        };

        let multiplicand = (&out_keys.index_axis(Axis(0), 1) - &out_keys.index_axis(Axis(0), 0))
            .broadcast((3, mpcctx.lpn_key_len + 1))
            .unwrap()
            .to_owned()
            .into_shape(3 * (mpcctx.lpn_key_len + 1))
            .unwrap();

        let selected = core::mult(
            id,
            selector,
            multiplicand,
            Array1::from_vec(randoms),
            Array1::from_vec(zeros),
            None,
            mpcctx.clone(),
            net.clone(),
        )
        .await;
        let mut selected = selected.into_shape((3, mpcctx.lpn_key_len + 1)).unwrap();

        // Compute the selection for the 4th row as a function of the selection for the first 3
        // rows.
        let mat = selected.sum_axis(Axis(0)) + out_keys.index_axis(Axis(0), 1)
            - out_keys.index_axis(Axis(0), 0);
        selected.push(Axis(0), mat.view()).unwrap();

        // Finish computing the selected plaintext for each row.
        // Shape: (4, lpn_key_len + 1)
        selected += &out_keys
            .index_axis(Axis(0), 0)
            .broadcast((4, mpcctx.lpn_key_len + 1))
            .unwrap();

        // Encode plaintext for encryption.
        selected.dot(&context.lpn_enc.view().reversed_axes())
    };

    // Compute garbled tables
    let lpn_mat = context.lpn_mat.get(gid);
    let ctxs: Vec<_> = (0..4)
        .into_par_iter()
        .flat_map(|row_idx| {
            let i = row_idx / 2;
            let j = row_idx % 2;

            // Shape: lpn_mssg_len
            let masks = {
                let keys: Array1<_> = &inp_keys.slice(s![0, i, ..]) + &inp_keys.slice(s![1, j, ..]);
                lpn_mat.index_axis(Axis(0), row_idx).dot(&keys)
            };

            // Shape: (lpn_mssg_len, num_blocks)
            (masks + mssgs.index_axis(Axis(0), row_idx)).to_vec()
        })
        .collect();
    let errors = Array2::from_shape_vec((4, mpcctx.lpn_mssg_len), errors).unwrap();
    let ctxs = Array2::from_shape_vec((4, mpcctx.lpn_mssg_len), ctxs).unwrap();
    ctxs + errors
}

async fn garble_xor<const W: u8>(
    id: ProtocolID,
    gid: usize,
    inp_masks: [PackedShare<W>; 2],
    inp_keys: Array3<PackedShare<W>>, // shape: (2 (left and right), 2 (0 and 1), lpn_key_len)
    out_mask: PackedShare<W>,
    mut out_keys: Array2<PackedShare<W>>, // shape: (2, lpn_key_len)
    randoms: Array1<PackedShare<W>>,
    zeros: Array1<PackedShare<W>>,
    errors: Vec<PackedShare<W>>,
    context: Arc<GarbleContext<W>>,
    net: Network,
) -> Array2<PackedShare<W>> {
    let mpcctx = &context.mpc;
    let num_mult = mpcctx.lpn_key_len + 1;

    debug_assert_eq!(inp_keys.shape(), [2, 2, mpcctx.lpn_key_len]);
    debug_assert_eq!(out_keys.shape(), [2, mpcctx.lpn_key_len]);
    debug_assert_eq!(randoms.len(), num_mult);
    debug_assert_eq!(zeros.len(), num_mult);
    debug_assert_eq!(errors.len(), 4 * mpcctx.lpn_mssg_len);

    // Append corresponding masks to keys.
    out_keys
        .push_column(Array1::from_vec(vec![out_mask, out_mask + GF::ONE]).view())
        .unwrap();

    // Select the message to be encrypted in each row of the garbled table.
    // Shape: (4, lpn_mssg_len)
    let mssgs = {
        let selector = {
            let s = inp_masks[0] + inp_masks[1] + out_mask;
            Array1::from_iter(std::iter::repeat(s).take(mpcctx.lpn_key_len + 1))
        };

        let multiplicand = &out_keys.index_axis(Axis(0), 1) - &out_keys.index_axis(Axis(0), 0);

        let mut mssg1 = core::mult(
            id,
            selector,
            multiplicand,
            randoms,
            zeros,
            None,
            mpcctx.clone(),
            net.clone(),
        )
        .await;

        // Compute 2nd row from the product.
        let mssg2 = &mssg1 + &out_keys.index_axis(Axis(0), 1);
        // Compute 1st row from the product.
        mssg1 += &out_keys.index_axis(Axis(0), 0);
        // Compute 3rd and 4th row from the product.
        let mssgs = ndarray::stack![Axis(0), mssg1, mssg2, mssg2, mssg1];

        // Encode plaintext for encryption.
        mssgs.dot(&context.lpn_enc.view().reversed_axes())
    };

    // Compute garbled tables
    let lpn_mat = context.lpn_mat.get(gid);
    let ctxs: Vec<_> = (0..4)
        .into_par_iter()
        .flat_map(|row_idx| {
            let i = row_idx / 2;
            let j = row_idx % 2;

            // Shape: lpn_mssg_len
            let masks = {
                let keys: Array1<_> = &inp_keys.slice(s![0, i, ..]) + &inp_keys.slice(s![1, j, ..]);
                lpn_mat.index_axis(Axis(0), row_idx).dot(&keys)
            };

            // Shape: (lpn_mssg_len, num_blocks)
            (masks + mssgs.index_axis(Axis(0), row_idx)).to_vec()
        })
        .collect();
    let errors = Array2::from_shape_vec((4, mpcctx.lpn_mssg_len), errors).unwrap();
    let ctxs = Array2::from_shape_vec((4, mpcctx.lpn_mssg_len), ctxs).unwrap();
    ctxs + errors
}

async fn garble_inv<const W: u8>(
    id: ProtocolID,
    gid: usize,
    inp_mask: PackedShare<W>,
    inp_keys: Array2<PackedShare<W>>, // shape: (2 (0 and 1), lpn_key_len)
    out_mask: PackedShare<W>,
    mut out_keys: Array2<PackedShare<W>>, // shape: (2, lpn_key_len)
    randoms: Array1<PackedShare<W>>,
    zeros: Array1<PackedShare<W>>,
    errors: Vec<PackedShare<W>>,
    context: Arc<GarbleContext<W>>,
    net: Network,
) -> Array2<PackedShare<W>> {
    let mpcctx = &context.mpc;
    let num_mult = mpcctx.lpn_key_len + 1;

    debug_assert_eq!(inp_keys.shape(), [2, mpcctx.lpn_key_len]);
    debug_assert_eq!(out_keys.shape(), [2, mpcctx.lpn_key_len]);
    debug_assert_eq!(randoms.len(), num_mult);
    debug_assert_eq!(zeros.len(), num_mult);
    debug_assert_eq!(errors.len(), 2 * mpcctx.lpn_mssg_len);

    // Append corresponding masks to keys.
    out_keys
        .push_column(Array1::from_vec(vec![out_mask, out_mask + GF::ONE]).view())
        .unwrap();

    // Select the message to be encrypted in each row of the garbled table.
    // Shape: (2, lpn_mssg_len)
    let mssgs = {
        let selector = Array1::from_elem(mpcctx.lpn_key_len + 1, inp_mask + out_mask);
        let multiplicand =
            (&out_keys.index_axis(Axis(0), 1) - &out_keys.index_axis(Axis(0), 0)).to_owned();

        let mut selected = core::mult(
            id,
            selector,
            multiplicand,
            randoms,
            zeros,
            None,
            mpcctx.clone(),
            net.clone(),
        )
        .await;
        // This completes computing the plaintext for the first row.
        selected += &out_keys.index_axis(Axis(0), 0);
        let mut selected = selected.into_shape((1, mpcctx.lpn_key_len + 1)).unwrap();

        // The plaintext for the second row corresponds to the key not selected in the first row.
        selected
            .append(Axis(0), (out_keys.sum_axis(Axis(0)) + &selected).view())
            .unwrap();

        // Encode plaintext for encryption.
        selected.dot(&context.lpn_enc.view().reversed_axes())
    };

    // Compute garbled tables
    let lpn_mat = context.lpn_mat.get(gid);
    let ctxs: Vec<_> = (0..2)
        .into_par_iter()
        .flat_map(|i| {
            // Shape: lpn_mssg_len
            let masks = {
                lpn_mat
                    .index_axis(Axis(0), i)
                    .dot(&inp_keys.slice(s![i, ..]))
            };

            // Shape: (lpn_mssg_len, num_blocks)
            (masks + mssgs.index_axis(Axis(0), i)).to_vec()
        })
        .collect();
    let errors = Array2::from_shape_vec((2, mpcctx.lpn_mssg_len), errors).unwrap();
    let ctxs = Array2::from_shape_vec((2, mpcctx.lpn_mssg_len), ctxs).unwrap();
    ctxs + errors
}

fn select<const N: usize, const W: u8>(
    shares: ArrayView2<GF<W>>, // shape: (num_blocks, num_trans)
    ginf: &PackedGateInfo<N>,
    context: &GarbleContext<W>,
) -> Vec<GF<W>> {
    let n: u32 = context.mpc.n.try_into().unwrap();
    let l: u32 = context.mpc.l.try_into().unwrap();

    let mypos = PackedSharing::share_pos(n)[context.mpc.id as usize];

    (0..N)
        .into_par_iter()
        .flat_map(|i| {
            let mut wires = ginf.inp[i].clone();
            wires.sort_unstable();
            wires.dedup();

            let pos: Vec<_> = PackedSharing::wire_to_pos(n, l, wires.iter().cloned()).collect();

            let indices: Vec<_> = wires
                .iter()
                .map(|j| *context.lookup.get(j).unwrap())
                .collect();
            let selected = shares.select(Axis(0), &indices);

            // Let p_i(x) be the polynomial such that p_i(pos[i]) = 1 and p_i(pos[j]) = 0
            // everywhere else i.e., it is the i-th lagrange polynomial for pos.
            // Then by definition const_coeffs[i] = p_i(my_pos).
            let const_coeffs = math::lagrange_coeffs(&pos, &[mypos]).remove_axis(Axis(0));
            // Thus, the following inner product corresponds to the selection since the i-th
            // multiplication corresponds to point wise (at my_pos) multiplication of p_i(x) and
            // the i-th selected sharing polynomial v_i(x).
            const_coeffs.dot(&selected).to_vec()
        })
        .collect()
}

// Used to transform the secret shares corresponding to the input and output wires of gates to
// ensure that the underlying secrets are at the correct positions.
struct Transforms<const W: u8> {
    rtrans: Vec<RandSharingTransform<W>>,
    trans: Vec<SharingTransform<W>>,
}

impl<const W: u8> Transforms<W> {
    // TODO: Clean up interface.
    fn new(tf: Vec<((Vec<GF<W>>, Vec<GF<W>>), Combination)>, context: &MPCContext<W>) -> Self {
        let trans = tf
            .par_iter()
            .cloned()
            .map(|((opos, npos), f)| SharingTransform::new(opos, npos, f))
            .collect();

        let rtrans = tf
            .into_par_iter()
            .chunks(context.l)
            .map(|chunk| {
                let (pos, f): (Vec<_>, Vec<_>) = chunk.iter().cloned().unzip();
                let (opos, npos): (Vec<_>, Vec<_>) = pos.into_iter().unzip();
                RandSharingTransform::new(&opos, &npos, &f, context)
            })
            .collect();

        Self { rtrans, trans }
    }

    // Outputs degree-d and degree-(n-1) matrix of random sharings for computing share
    // transformations.
    // Shape of output matrices is (number of transforms, number of repetitions).
    async fn gen_rand_shares(
        &self,
        id: ProtocolID,
        reps: usize,
        mut randoms: VecDeque<GF<W>>,
        mut zeros: VecDeque<GF<W>>,
        context: MPCContext<W>,
        net: Network,
    ) -> (Array2<GF<W>>, Array2<GF<W>>) {
        let num_rands_per_rtrans = (context.n + context.t) * reps;
        let num_zeros_per_rtrans = (2 * context.n) * reps;

        debug_assert_eq!(randoms.len(), self.rtrans.len() * num_rands_per_rtrans);
        debug_assert_eq!(zeros.len(), self.rtrans.len() * num_zeros_per_rtrans);

        let mut id_gen = ProtocolIDBuilder::new(&id, self.rtrans.len() as u64);

        let mut handles = Vec::with_capacity(self.rtrans.len());
        for rt in self.rtrans.iter().cloned() {
            handles.push(spawn(core::randtrans(
                id_gen.next().unwrap(),
                randoms.drain(..(num_rands_per_rtrans)).collect(),
                zeros.drain(..(num_zeros_per_rtrans)).collect(),
                rt,
                context.clone(),
                net.clone(),
            )));
        }

        let mut rand = Array2::zeros((0, reps));
        let mut rand_n = Array2::zeros((0, reps));
        for (i, handle) in handles.into_iter().enumerate() {
            let (r, rn) = handle.await.unwrap();

            if i == self.rtrans.len() - 1 {
                let end = std::cmp::min(context.l, self.trans.len() - i * context.l);
                rand.append(Axis(0), r.slice(s![..end, ..]).view()).unwrap();
                rand_n
                    .append(Axis(0), rn.slice(s![..end, ..]).view())
                    .unwrap();
            } else {
                rand.append(Axis(0), r.view()).unwrap();
                rand_n.append(Axis(0), rn.view()).unwrap();
            }
        }

        (rand, rand_n)
    }

    // Shape of input and output matrices is (number of transforms, number of repetitions).
    async fn transform(
        &self,
        id: ProtocolID,
        inp: Array2<GF<W>>,
        rand: Array2<GF<W>>,
        rand_n: Array2<GF<W>>,
        context: MPCContext<W>,
        net: Network,
    ) -> Array2<GF<W>> {
        let reps = rand.shape()[1];
        let num = self.trans.len();
        let mut id_gen = ProtocolIDBuilder::new(&id, num as u64);

        debug_assert_eq!(inp.shape(), [num, reps]);
        debug_assert_eq!(rand.shape(), [num, reps]);
        debug_assert_eq!(rand_n.shape(), [num, reps]);

        let mut handles = Vec::with_capacity(num);
        for i in 0..num {
            handles.push(spawn(core::trans(
                id_gen.next().unwrap(),
                inp.row(i).into_owned(),
                rand.row(i).into_owned(),
                rand_n.row(i).into_owned(),
                self.trans[i].clone(),
                Some((i % context.n).try_into().unwrap()),
                context.clone(),
                net.clone(),
            )));
        }

        let mut res = Vec::with_capacity(num * reps);
        for handle in handles {
            res.extend(handle.await.unwrap().into_iter());
        }

        Array2::from_shape_vec((num, reps), res).unwrap()
    }
}

// Computes LPN matrices for encryption.
// Attempts to decrease number of required matrices by re-using matrices in such a way that no
// matrix is re-used with the same key.
struct LPNMatrix<const W: u8> {
    mats: Vec<ArcArray<PackedShare<W>, Ix3>>,
    lookup: Vec<usize>,
}

impl<const W: u8> LPNMatrix<W> {
    fn new(circ: &PackedCircuit, context: &MPCContext<W>) -> Self {
        let mats = {
            let mut rng = ChaCha12Rng::seed_from_u64(200);

            let mat = Array3::from_shape_simple_fn(
                (4, context.lpn_mssg_len, context.lpn_key_len),
                || GF::rand(&mut rng),
            )
            .to_shared();

            vec![mat]
        };

        Self {
            mats,
            lookup: vec![0; circ.gates().len()],
        }
    }

    fn get(&self, gid: usize) -> ArcArray<PackedShare<W>, Ix3> {
        self.mats[self.lookup[gid]].clone()
    }
}
