use super::core;
use super::core::{RandSharingTransform, SharingTransform};
use super::preproc::PreProc;
use super::{MPCContext, PartyID, ProtocolID, ProtocolIDBuilder};
use crate::circuit::{PackedCircuit, PackedGate, PackedGateInfo, WireID};
use crate::math;
use crate::math::galois::{GFElement, GFMatrix, GF};
use crate::math::Combination;
use crate::sharing::{PackedShare, PackedSharing};
use crate::spawn;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use sha2::{Digest, Sha256};
use smol::stream::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;

struct WireSelector {
    lookup: HashMap<WireID, usize>,
    // This is a bit unweildy but essentially is a list of coefficients to select secrets from the
    // list of wires and masks.
    // It is a vector of coefficients (Vec<GFElement>) for a block of input wires to a packed gate
    // (Vec<Vec<GFElement>>, since PackedGates can have fan-in 2 or 1) for every packed gate
    // (Vec<Vec<Vec<GFElement>>>).
    coeffs: Vec<Vec<Vec<GFElement>>>,
}

impl WireSelector {
    fn new(id: PartyID, circ: &PackedCircuit, pss: &PackedSharing, gf: &GF) -> Self {
        Self {
            lookup: Self::compute_wire_to_block_map(circ),
            coeffs: Self::compute_coeffs(id, circ, pss, gf),
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

    fn compute_coeffs(
        id: PartyID,
        circ: &PackedCircuit,
        pss: &PackedSharing,
        gf: &GF,
    ) -> Vec<Vec<Vec<GFElement>>> {
        let l: usize = pss.packing_param() as usize;

        let identity_matrix = {
            let mut acc = Vec::with_capacity(l);
            let mut row = vec![gf.zero(); l];

            row[0] = gf.one();
            acc.push(row.clone());

            for i in 1..l {
                row[i] = gf.one();
                row[i - 1] = gf.zero();
                acc.push(row.clone());
            }

            acc
        };

        let offset = pss.pos_offset();

        let gen_coeffs_for_block = |wires: &[WireID]| -> Vec<_> {
            let mut pos: Vec<_> = wires.iter().map(|w| gf.get(offset + *w)).collect();
            pos.sort_unstable();
            pos.dedup();

            // Embed each row of the pos.len() order identity matrix into a share.
            let coeffs = pss.const_coeffs(&pos, id, gf);
            identity_matrix[..pos.len()]
                .iter()
                .map(|r| math::utils::iprod(&coeffs, &r[..pos.len()], gf))
                .collect()
        };

        let mut coeffs = Vec::new();

        for gate in circ.gates() {
            match gate {
                PackedGate::Xor(ginf) => {
                    coeffs.push(vec![
                        gen_coeffs_for_block(&ginf.inp[0]),
                        gen_coeffs_for_block(&ginf.inp[1]),
                    ]);
                }
                PackedGate::And(ginf) => {
                    coeffs.push(vec![
                        gen_coeffs_for_block(&ginf.inp[0]),
                        gen_coeffs_for_block(&ginf.inp[1]),
                    ]);
                }
                PackedGate::Inv(ginf) => {
                    coeffs.push(vec![gen_coeffs_for_block(&ginf.inp[0])]);
                }
            }
        }

        coeffs
    }

    fn select<const N: usize>(
        &self,
        inp: &[PackedShare],
        gid: usize,
        ginf: &PackedGateInfo<N>,
        gf: &GF,
    ) -> [PackedShare; N] {
        let mut output = [gf.zero(); N];
        let coeffs_list = &self.coeffs[gid];

        for i in 0..N {
            let mut wires = ginf.inp[i].clone();
            wires.sort_unstable();
            wires.dedup();

            let filtered_inps = wires.iter().map(|j| &inp[*self.lookup.get(j).unwrap()]);
            output[i] = math::utils::iprod(&coeffs_list[i], filtered_inps, gf);
        }

        output
    }
}

pub struct GarbleContextData {
    out_rtrans: Vec<RandSharingTransform>,
    out_trans: Vec<SharingTransform>,
    inp_rtrans: Vec<RandSharingTransform>,
    inp_trans: Vec<Vec<SharingTransform>>,
    selector: WireSelector,
    lpn_enc: GFMatrix,
    circ: PackedCircuit,
    mpc: MPCContext,
}

type GarbleContext = Arc<GarbleContextData>;

impl GarbleContextData {
    pub fn new(circ: PackedCircuit, context: MPCContext) -> Self {
        let selector =
            WireSelector::new(context.id, &circ, context.pss.as_ref(), context.gf.as_ref());

        let (inp_rtrans, inp_trans) = Self::compute_inp_transforms(&circ, &context);
        let (out_rtrans, out_trans) = Self::compute_out_transforms(&circ, &context);

        Self {
            out_rtrans,
            out_trans,
            inp_rtrans,
            inp_trans,
            selector,
            lpn_enc: math::rs_gen_mat(
                context.lpn_key_len + 1, // Need to encode key and mask
                context.lpn_mssg_len,
                context.gf.as_ref(),
            ),
            circ,
            mpc: context,
        }
    }

    fn prepare_inp_transform(
        inp_wires: Vec<WireID>,
        out_wires: &[WireID],
        pss: &PackedSharing,
        gf: &GF,
    ) -> (Vec<GFElement>, Combination, Vec<GFElement>) {
        let mut inp_dedup = inp_wires.clone();
        inp_dedup.sort_unstable();
        inp_dedup.dedup();

        // Build the transform that orders and repeats secrets correctly.
        let f = Combination::from_instance(&inp_dedup, &inp_wires);

        let offset = pss.pos_offset();
        let opos: Vec<_> = inp_dedup.into_iter().map(|x| gf.get(offset + x)).collect();
        let npos = out_wires.into_iter().map(|x| gf.get(offset + *x)).collect();

        (opos, f, npos)
    }

    fn compute_inp_transforms(
        circ: &PackedCircuit,
        context: &MPCContext,
    ) -> (Vec<RandSharingTransform>, Vec<Vec<SharingTransform>>) {
        let mut trans = Vec::new();
        let mut opos_list = Vec::new();
        let mut npos_list = Vec::new();
        let mut f_list = Vec::new();

        let mut update_lists = |inp_wires: &[WireID], out_wires: &[WireID]| {
            let (opos, f, npos) = Self::prepare_inp_transform(
                inp_wires.to_vec(),
                out_wires,
                context.pss.as_ref(),
                context.gf.as_ref(),
            );

            let transform = SharingTransform::new(
                &opos,
                &npos,
                f.clone(),
                context.pss.as_ref(),
                context.gf.as_ref(),
            );
            opos_list.push(opos);
            npos_list.push(npos);
            f_list.push(f);

            transform
        };

        for gate in circ.gates() {
            match gate {
                PackedGate::Xor(ginf) => {
                    trans.push(vec![
                        update_lists(&ginf.inp[0], &ginf.out),
                        update_lists(&ginf.inp[1], &ginf.out),
                    ]);
                }
                PackedGate::And(ginf) => {
                    trans.push(vec![
                        update_lists(&ginf.inp[0], &ginf.out),
                        update_lists(&ginf.inp[1], &ginf.out),
                    ]);
                }
                PackedGate::Inv(ginf) => {
                    trans.push(vec![update_lists(&ginf.inp[0], &ginf.out)]);
                }
            }
        }

        let mut rand_trans = Vec::new();

        for ((opos, f), npos) in opos_list
            .chunks(context.l)
            .zip(f_list.chunks(context.l))
            .zip(npos_list.chunks(context.l))
        {
            rand_trans.push(RandSharingTransform::new(
                context.id,
                &opos,
                &npos,
                &f,
                context.pss.as_ref(),
                context.gf.as_ref(),
            ));
        }

        (rand_trans, trans)
    }

    fn compute_out_transforms(
        circ: &PackedCircuit,
        context: &MPCContext,
    ) -> (Vec<RandSharingTransform>, Vec<SharingTransform>) {
        let mut trans = Vec::new();
        let mut npos_list = Vec::new();
        let mut f_list = Vec::new();

        let def_pos = context.pss.default_pos(context.gf.as_ref());
        let offset = context.pss.pos_offset();

        let mut update_lists = |out_wires: &[WireID]| {
            let npos: Vec<_> = out_wires
                .iter()
                .map(|&x| context.gf.get(offset + x))
                .collect();
            let f = Combination::new((0..npos.len()).collect());

            let transform = SharingTransform::new(
                &def_pos,
                &npos,
                f.clone(),
                context.pss.as_ref(),
                context.gf.as_ref(),
            );
            npos_list.push(npos);
            f_list.push(f);

            transform
        };

        for inp_wires in circ.inputs() {
            trans.push(update_lists(inp_wires));
        }

        for gate in circ.gates() {
            match gate {
                PackedGate::Xor(ginf) => {
                    trans.push(update_lists(&ginf.out));
                }
                PackedGate::And(ginf) => {
                    trans.push(update_lists(&ginf.out));
                }
                PackedGate::Inv(ginf) => {
                    trans.push(update_lists(&ginf.out));
                }
            }
        }

        let opos = vec![def_pos; context.l];
        let mut rand_trans = Vec::new();

        for (f, npos) in f_list.chunks(context.l).zip(npos_list.chunks(context.l)) {
            rand_trans.push(RandSharingTransform::new(
                context.id,
                &opos[..npos.len()],
                &npos,
                &f,
                context.pss.as_ref(),
                context.gf.as_ref(),
            ));
        }

        (rand_trans, trans)
    }
}

// TODO: Hide members and provide methods.
pub struct GarbledTable<const N: usize> {
    pub ctxs: [Vec<PackedShare>; N],
}

pub enum GarbledGate {
    Xor(GarbledTable<4>),
    And(GarbledTable<4>),
    Inv(GarbledTable<2>),
}

pub struct GarbledCircuit {
    pub gates: Vec<GarbledGate>,
}

pub async fn garble(
    id: ProtocolID,
    mut preproc: PreProc,
    context: GarbleContext,
) -> GarbledCircuit {
    let mpcctx = &context.mpc;
    let circ = &context.circ;
    let num_circ_inp_blocks = circ.inputs().len();
    let num_gate_blocks = circ.gates().len();
    let num_blocks = num_circ_inp_blocks + num_gate_blocks;

    // Sanity checks.
    #[cfg(debug_assertions)]
    {
        assert_eq!(context.out_trans.len(), num_blocks);
        assert_eq!(preproc.masks.len(), num_blocks);
        assert_eq!(preproc.keys[0].len(), mpcctx.lpn_key_len);
        assert_eq!(preproc.keys[1].len(), mpcctx.lpn_key_len);

        for b in 0..2 {
            for i in 0..mpcctx.lpn_key_len {
                assert_eq!(preproc.keys[b][i].len(), num_blocks);
            }
        }
    }

    let num_trans_per_wire = 2 * mpcctx.lpn_key_len + 1;
    let num_subproto =
        context.out_rtrans.len() + context.inp_rtrans.len() + num_blocks + num_gate_blocks;
    let mut id_gen = ProtocolIDBuilder::new(&id, num_subproto as u64);

    // Run randtrans for transforming keys and masks.
    let mut out_rtrans_handles = Vec::with_capacity(context.out_rtrans.len());
    for rtrans in context.out_rtrans.iter() {
        let randoms = preproc
            .randoms
            .split_off(preproc.randoms.len() - (mpcctx.n + mpcctx.t) * num_trans_per_wire);
        let zeros = preproc
            .zeros
            .split_off(preproc.zeros.len() - 2 * mpcctx.n * num_trans_per_wire);
        out_rtrans_handles.push(spawn(core::batch_randtrans(
            id_gen.next().unwrap(),
            randoms,
            zeros,
            rtrans.clone(),
            mpcctx.clone(),
        )));
    }

    // Run randtrans for transforming input keys and masks for each packed gate.
    let mut inp_rtrans_handles = Vec::with_capacity(context.inp_rtrans.len());
    for rtrans in context.inp_rtrans.iter() {
        let randoms = preproc
            .randoms
            .split_off(preproc.randoms.len() - (mpcctx.n + mpcctx.t) * num_trans_per_wire);
        let zeros = preproc
            .zeros
            .split_off(preproc.zeros.len() - 2 * mpcctx.n * num_trans_per_wire);
        inp_rtrans_handles.push(spawn(core::batch_randtrans(
            id_gen.next().unwrap(),
            randoms,
            zeros,
            rtrans.clone(),
            mpcctx.clone(),
        )));
    }

    // Transform masks and keys for output wire of each gate.
    let mut out_trans_handles = Vec::with_capacity(num_blocks);
    let mut out_rtrans_stream = smol::stream::iter(out_rtrans_handles)
        .then(|fut| async { fut.await })
        .map(|shares| smol::stream::iter(shares))
        .flatten()
        .boxed();

    for i in 0..num_blocks {
        let trans_shares = out_rtrans_stream.next().await.unwrap();
        let (share, share_n) = trans_shares.into_iter().unzip();

        let trans_inp = (0..num_trans_per_wire)
            .map(|j| {
                if j < mpcctx.lpn_key_len {
                    preproc.keys[0][j][i]
                } else if j < 2 * mpcctx.lpn_key_len {
                    preproc.keys[1][j - mpcctx.lpn_key_len][i]
                } else {
                    preproc.masks[i]
                }
            })
            .collect();

        out_trans_handles.push(spawn(core::batch_trans(
            id_gen.next().unwrap(),
            trans_inp,
            share,
            share_n,
            context.out_trans[i].clone(),
            mpcctx.clone(),
        )));
    }

    // Collect transformed keys and masks for output wires.
    let mut masks = Vec::with_capacity(num_blocks);
    let mut keys: [Vec<_>; 2] = [
        (0..mpcctx.lpn_key_len)
            .map(|_| Vec::with_capacity(num_blocks))
            .collect(),
        (0..mpcctx.lpn_key_len)
            .map(|_| Vec::with_capacity(num_blocks))
            .collect(),
    ];

    for handle in out_trans_handles {
        let mut shares = handle.await;
        masks.push(shares.pop().unwrap());

        for i in 0..2 {
            for j in 0..mpcctx.lpn_key_len {
                keys[i][j].push(shares[i * mpcctx.lpn_key_len + j]);
            }
        }
    }
    #[cfg(debug_assertions)]
    {
        assert_eq!(keys[0].len(), mpcctx.lpn_key_len);
        assert_eq!(keys[1].len(), mpcctx.lpn_key_len);

        for i in 0..mpcctx.lpn_key_len {
            assert_eq!(keys[0][i].len(), num_blocks);
            assert_eq!(keys[1][i].len(), num_blocks);
        }
    }

    // These are used for selecting inputs for each packed gate.
    let masks = Arc::new(masks);
    let keys = Arc::new(keys);

    // Transform inputs to gates and compute garbled circuit.
    let mut inp_trans_stream = smol::stream::iter(inp_rtrans_handles)
        .then(|fut| async { fut.await })
        .map(|shares| smol::stream::iter(shares))
        .flatten()
        .boxed();

    let mut gc_handles = Vec::with_capacity(circ.gates().len());
    for (gid, gate) in circ.gates().iter().enumerate() {
        let masks = masks.clone();
        let keys = keys.clone();
        let out_mask = preproc.masks[num_circ_inp_blocks + gid];
        let out_key: [Vec<_>; 2] = [0, 1].map(|i| {
            preproc.keys[i]
                .iter()
                .map(|v| v[num_circ_inp_blocks + gid])
                .collect()
        });
        let context = context.clone();

        match gate {
            PackedGate::Xor(ginf) => {
                let rtrans = [
                    inp_trans_stream.next().await.unwrap(),
                    inp_trans_stream.next().await.unwrap(),
                ];
                let sub_id = id_gen.next().unwrap();
                let num = 4 * mpcctx.lpn_mssg_len;
                let randoms = preproc.randoms.split_off(preproc.randoms.len() - num);
                let zeros = preproc.zeros.split_off(preproc.zeros.len() - num);
                let errors = preproc.errors.split_off(preproc.errors.len() - num);
                let ginf = ginf.to_owned();

                gc_handles.push(spawn(async move {
                    let (inp_masks, inp_keys) = transform_inputs(
                        sub_id.clone(),
                        gid,
                        &ginf,
                        &masks,
                        &keys,
                        rtrans,
                        &context,
                    )
                    .await;

                    garble_xor(
                        sub_id, gid, inp_masks, inp_keys, out_mask, out_key, randoms, zeros,
                        errors, context,
                    )
                    .await
                }));
            }
            PackedGate::And(ginf) => {
                let rtrans = [
                    inp_trans_stream.next().await.unwrap(),
                    inp_trans_stream.next().await.unwrap(),
                ];
                let sub_id = id_gen.next().unwrap();
                let num = 4 * mpcctx.lpn_mssg_len;
                let randoms = preproc.randoms.split_off(preproc.randoms.len() - num - 1);
                let zeros = preproc.zeros.split_off(preproc.zeros.len() - num - 1);
                let errors = preproc.errors.split_off(preproc.errors.len() - num);
                let ginf = ginf.to_owned();

                gc_handles.push(spawn(async move {
                    let (inp_masks, inp_keys) = transform_inputs(
                        sub_id.clone(),
                        gid,
                        &ginf,
                        &masks,
                        &keys,
                        rtrans,
                        &context,
                    )
                    .await;

                    garble_and(
                        sub_id, gid, inp_masks, inp_keys, out_mask, out_key, randoms, zeros,
                        errors, context,
                    )
                    .await
                }));
            }
            PackedGate::Inv(ginf) => {
                let rtrans = [inp_trans_stream.next().await.unwrap()];
                let sub_id = id_gen.next().unwrap();
                let num = 2 * mpcctx.lpn_mssg_len;
                let randoms = preproc.randoms.split_off(preproc.randoms.len() - num);
                let zeros = preproc.zeros.split_off(preproc.zeros.len() - num);
                let errors = preproc.errors.split_off(preproc.errors.len() - num);
                let ginf = ginf.to_owned();

                gc_handles.push(spawn(async move {
                    let (inp_masks, inp_keys) = transform_inputs(
                        sub_id.clone(),
                        gid,
                        &ginf,
                        &masks,
                        &keys,
                        rtrans,
                        &context,
                    )
                    .await;

                    let inp_mask = inp_masks[0];
                    let inp_key = inp_keys.into_iter().next().unwrap();

                    garble_inv(
                        sub_id, gid, inp_mask, inp_key, out_mask, out_key, randoms, zeros, errors,
                        context,
                    )
                    .await
                }));
            }
        }
    }

    let mut gc_gates = Vec::with_capacity(circ.gates().len());
    for handle in gc_handles {
        gc_gates.push(handle.await);
    }

    GarbledCircuit { gates: gc_gates }
}

async fn transform_inputs<const N: usize>(
    id: ProtocolID,
    gid: usize,
    ginf: &PackedGateInfo<N>,
    masks: &[PackedShare],
    keys: &[Vec<Vec<PackedShare>>; 2],
    rtrans: [Vec<(PackedShare, PackedShare)>; N],
    context: &GarbleContext,
) -> ([PackedShare; N], [[Vec<PackedShare>; 2]; N]) {
    let mpcctx = &context.mpc;
    let mut id_gen = ProtocolIDBuilder::new(&id, (N + 1) as u64);

    let mut trans_inp: Vec<_> = (0..N)
        .map(|_| Vec::with_capacity(2 * mpcctx.lpn_key_len + 1))
        .collect();
    for k in 0..2 {
        for l in 0..mpcctx.lpn_key_len {
            let selected_keys = context
                .selector
                .select(&keys[k][l], gid, ginf, mpcctx.gf.as_ref());
            for (i, val) in selected_keys.into_iter().enumerate() {
                trans_inp[i].push(val);
            }
        }
    }

    for i in 0..N {
        trans_inp[i].push(masks[i]);
    }

    let mut handles = Vec::with_capacity(N);
    for (i, (trans_shares, trans_inp)) in rtrans.into_iter().zip(trans_inp).enumerate() {
        let (share, share_n) = trans_shares.into_iter().unzip();

        handles.push(spawn(core::batch_trans(
            id_gen.next().unwrap(),
            trans_inp,
            share,
            share_n,
            context.inp_trans[gid][i].clone(),
            mpcctx.clone(),
        )));
    }

    let mut trans_masks = [mpcctx.gf.zero(); N];
    let mut trans_keys = [(); N].map(|_| [Vec::new(), Vec::new()]);

    for (i, handle) in handles.into_iter().enumerate() {
        let mut shares = handle.await;
        trans_masks[i] = shares.pop().unwrap();
        trans_keys[i][1] = shares.split_off(mpcctx.lpn_key_len);
        trans_keys[i][0] = shares;

        debug_assert_eq!(trans_keys[i][0].len(), mpcctx.lpn_key_len);
    }

    (trans_masks, trans_keys)
}

// Using pseudorandomness for LPN matrix.
// TODO: Refactor so that it can be used in evaluation phase too.
fn lpn_mat_row(
    gid: usize,
    row: usize,
    mssg_idx: usize,
    lpn_mssg_len: usize,
    gf: &GF,
) -> Vec<GFElement> {
    let rng_seed: Vec<_> = [gid, row, mssg_idx]
        .iter()
        .map(|v| v.to_be_bytes())
        .flatten()
        .collect();
    let mut hasher = Sha256::new();
    hasher.update(&rng_seed);
    let rng_seed = hasher.finalize();
    let mut rng = ChaCha12Rng::from_seed(rng_seed.into());

    (0..lpn_mssg_len).map(|_| gf.rand(&mut rng)).collect()
}

async fn garble_and(
    id: ProtocolID,
    gid: usize,
    inp_masks: [PackedShare; 2],
    inp_keys: [[Vec<PackedShare>; 2]; 2],
    out_mask: PackedShare,
    out_key: [Vec<PackedShare>; 2],
    mut randoms: Vec<PackedShare>,
    mut zeros: Vec<PackedShare>,
    errors: Vec<PackedShare>,
    context: GarbleContext,
) -> GarbledGate {
    let mpcctx = &context.mpc;
    let num_ctxs = 4 * mpcctx.lpn_mssg_len;

    debug_assert_eq!(randoms.len(), num_ctxs + 1);
    debug_assert_eq!(zeros.len(), num_ctxs + 1);
    debug_assert_eq!(errors.len(), num_ctxs);

    // Compute product of input masks.
    // Will be used to compute the garbled table later.
    let mask_prod = core::mult(
        id.clone(),
        inp_masks[0],
        inp_masks[1],
        randoms.pop().unwrap(),
        zeros.pop().unwrap(),
        mpcctx.clone(),
    )
    .await;

    // RS encode the output keys concatenated with their labels.
    let mut out_key = out_key;
    out_key[0].push(mpcctx.gf.zero());
    out_key[1].push(mpcctx.gf.one());
    let mssgs: [Vec<_>; 2] = out_key.map(|k| {
        math::utils::matrix_vector_prod(&context.lpn_enc, &k, mpcctx.gf.as_ref()).collect()
    });

    // Compute ciphertexts for the garbled table.
    let mut select_bits = Vec::with_capacity(num_ctxs);
    let mut mssg_diff = Vec::with_capacity(num_ctxs);
    for i in 0..4 {
        let select_bit = if i == 0 {
            mask_prod + out_mask
        } else if i == 1 {
            mask_prod + inp_masks[0] + out_mask
        } else if i == 2 {
            mask_prod + inp_masks[1] + out_mask
        } else {
            mask_prod + inp_masks[0] + inp_masks[1] + mpcctx.gf.one() + out_mask
        };

        for k in 0..mpcctx.lpn_mssg_len {
            select_bits.push(select_bit);
            mssg_diff.push(mssgs[1][k] - mssgs[0][k]);
        }
    }

    let selected_mssgs =
        core::batch_mult(id, select_bits, mssg_diff, randoms, zeros, mpcctx.clone()).await;

    // Collect all ciphertexts.
    let mut gtable = GarbledTable {
        ctxs: [(); 4].map(|_| Vec::with_capacity(mpcctx.lpn_mssg_len)),
    };

    for i in 0..2 {
        for j in 0..2 {
            let row_idx = 2 * i + j;

            let lpn_key: Vec<_> = inp_keys[0][i]
                .iter()
                .zip(inp_keys[1][j].iter())
                .map(|(&k1, &k2)| k1 + k2)
                .collect();

            for k in 0..mpcctx.lpn_mssg_len {
                let mssg = selected_mssgs[row_idx * mpcctx.lpn_mssg_len + k] + mssgs[0][k];
                let row = lpn_mat_row(gid, row_idx, k, mpcctx.lpn_mssg_len, mpcctx.gf.as_ref());
                gtable.ctxs[row_idx].push(
                    math::utils::iprod(&lpn_key, &row, mpcctx.gf.as_ref())
                        + mssg
                        + errors[row_idx * mpcctx.lpn_mssg_len + k],
                );
            }
            debug_assert_eq!(gtable.ctxs[row_idx].len(), mpcctx.lpn_mssg_len);
        }
    }

    GarbledGate::And(gtable)
}

async fn garble_xor(
    id: ProtocolID,
    gid: usize,
    inp_masks: [PackedShare; 2],
    inp_keys: [[Vec<PackedShare>; 2]; 2],
    out_mask: PackedShare,
    out_key: [Vec<PackedShare>; 2],
    randoms: Vec<PackedShare>,
    zeros: Vec<PackedShare>,
    errors: Vec<PackedShare>,
    context: GarbleContext,
) -> GarbledGate {
    let mpcctx = &context.mpc;
    let num_ctxs = 4 * mpcctx.lpn_mssg_len;

    debug_assert_eq!(randoms.len(), num_ctxs);
    debug_assert_eq!(zeros.len(), num_ctxs);
    debug_assert_eq!(errors.len(), num_ctxs);

    // RS encode the output keys concatenated with their labels.
    let mut out_key = out_key;
    out_key[0].push(mpcctx.gf.zero());
    out_key[1].push(mpcctx.gf.one());
    let mssgs: [Vec<_>; 2] = out_key.map(|k| {
        math::utils::matrix_vector_prod(&context.lpn_enc, &k, mpcctx.gf.as_ref()).collect()
    });

    // Compute ciphertexts for the garbled table.
    let mut select_bits = Vec::with_capacity(num_ctxs);
    let mut mssg_diff = Vec::with_capacity(num_ctxs);
    for i in 0..4 {
        let select_bit = if i == 0 {
            inp_masks[0] + inp_masks[1] + out_mask
        } else if i == 1 {
            inp_masks[0] + inp_masks[1] + mpcctx.gf.one() + out_mask
        } else if i == 2 {
            inp_masks[0] + inp_masks[1] + mpcctx.gf.one() + out_mask
        } else {
            inp_masks[0] + inp_masks[1] + out_mask
        };

        for k in 0..mpcctx.lpn_mssg_len {
            select_bits.push(select_bit);
            mssg_diff.push(mssgs[1][k] - mssgs[0][k]);
        }
    }

    let selected_mssgs =
        core::batch_mult(id, select_bits, mssg_diff, randoms, zeros, mpcctx.clone()).await;

    // Collect all ciphertexts.
    let mut gtable = GarbledTable {
        ctxs: [(); 4].map(|_| Vec::with_capacity(mpcctx.lpn_mssg_len)),
    };

    for i in 0..2 {
        for j in 0..2 {
            let row_idx = 2 * i + j;

            let lpn_key: Vec<_> = inp_keys[0][i]
                .iter()
                .zip(inp_keys[1][j].iter())
                .map(|(&k1, &k2)| k1 + k2)
                .collect();

            for k in 0..mpcctx.lpn_mssg_len {
                let mssg = selected_mssgs[row_idx * mpcctx.lpn_mssg_len + k] + mssgs[0][k];
                let row = lpn_mat_row(gid, row_idx, k, mpcctx.lpn_mssg_len, mpcctx.gf.as_ref());
                gtable.ctxs[row_idx].push(
                    math::utils::iprod(&lpn_key, &row, mpcctx.gf.as_ref())
                        + mssg
                        + errors[row_idx * mpcctx.lpn_mssg_len + k],
                );
            }
            debug_assert_eq!(gtable.ctxs[row_idx].len(), mpcctx.lpn_mssg_len);
        }
    }

    GarbledGate::Xor(gtable)
}

async fn garble_inv(
    id: ProtocolID,
    gid: usize,
    inp_mask: PackedShare,
    inp_key: [Vec<PackedShare>; 2],
    out_mask: PackedShare,
    out_key: [Vec<PackedShare>; 2],
    randoms: Vec<PackedShare>,
    zeros: Vec<PackedShare>,
    errors: Vec<PackedShare>,
    context: GarbleContext,
) -> GarbledGate {
    let mpcctx = &context.mpc;
    let num_ctxs = 2 * mpcctx.lpn_mssg_len;

    debug_assert_eq!(randoms.len(), num_ctxs);
    debug_assert_eq!(zeros.len(), num_ctxs);
    debug_assert_eq!(errors.len(), num_ctxs);

    // RS encode the output keys concatenated with their labels.
    let mut out_key = out_key;
    out_key[0].push(mpcctx.gf.zero());
    out_key[1].push(mpcctx.gf.one());
    let mssgs: [Vec<_>; 2] = out_key.map(|k| {
        math::utils::matrix_vector_prod(&context.lpn_enc, &k, mpcctx.gf.as_ref()).collect()
    });

    let mut select_bits = Vec::with_capacity(num_ctxs);
    let mut mssg_diff = Vec::with_capacity(num_ctxs);
    for i in 0..2 {
        let select_bit = if i == 0 {
            inp_mask + mpcctx.gf.one() + out_mask
        } else {
            inp_mask + out_mask
        };

        for k in 0..mpcctx.lpn_mssg_len {
            select_bits.push(select_bit);
            mssg_diff.push(mssgs[1][k] - mssgs[0][k]);
        }
    }

    let selected_mssgs =
        core::batch_mult(id, select_bits, mssg_diff, randoms, zeros, mpcctx.clone()).await;

    let mut gtable = GarbledTable {
        ctxs: [(); 2].map(|_| Vec::with_capacity(mpcctx.lpn_mssg_len)),
    };

    // Compute ciphertexts for the garbled table.
    for (i, lpn_key) in inp_key.into_iter().enumerate() {
        for k in 0..mpcctx.lpn_mssg_len {
            let mssg = selected_mssgs[i * mpcctx.lpn_mssg_len + k] + mssgs[0][k];
            let row = lpn_mat_row(gid, i, k, mpcctx.lpn_mssg_len, mpcctx.gf.as_ref());
            gtable.ctxs[i].push(
                math::utils::iprod(&lpn_key, &row, mpcctx.gf.as_ref())
                    + mssg
                    + errors[i * mpcctx.lpn_mssg_len + k],
            );
        }
        debug_assert_eq!(gtable.ctxs[i].len(), mpcctx.lpn_mssg_len);
    }

    GarbledGate::Inv(gtable)
}
