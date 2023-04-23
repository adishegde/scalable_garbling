use super::core::RandSharingTransform;
use super::{MPCContext, PartyID};
use crate::circuit::{PackedCircuit, PackedGate, PackedGateInfo, WireID};
use crate::math;
use crate::math::galois::{GFElement, GF};
use crate::sharing::{PackedShare, PackedSharing};
use std::collections::HashMap;

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

        let gen_coeffs_for_block = |mut wires: Vec<WireID>| -> Vec<_> {
            wires.sort_unstable();
            wires.dedup();
            let wires: Vec<_> = wires.iter().map(|i| gf.get(*i)).collect();

            let coeffs = pss.const_coeffs(&wires, id, gf);
            identity_matrix[..wires.len()]
                .iter()
                .map(|r| math::utils::iprod(&coeffs, r, gf))
                .collect()
        };

        let mut coeffs = Vec::new();

        for gate in circ.gates() {
            match gate {
                PackedGate::Xor(ginf) => {
                    coeffs.push(vec![
                        gen_coeffs_for_block(ginf.inp[0].clone()),
                        gen_coeffs_for_block(ginf.inp[1].clone()),
                    ]);
                }
                PackedGate::And(ginf) => {
                    coeffs.push(vec![
                        gen_coeffs_for_block(ginf.inp[0].clone()),
                        gen_coeffs_for_block(ginf.inp[1].clone()),
                    ]);
                }
                PackedGate::Inv(ginf) => {
                    coeffs.push(vec![gen_coeffs_for_block(ginf.inp[0].clone())]);
                }
            }
        }

        coeffs
    }

    fn select<const N: usize>(
        &self,
        inp: Vec<PackedShare>,
        ginf: PackedGateInfo<N>,
        gf: &GF,
    ) -> [PackedShare; N] {
        let mut output = [gf.zero(); N];
        let coeffs_list = &self.coeffs[*self.lookup.get(&ginf.out[0]).unwrap()];

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

pub struct GarbleContext {
    transforms: Vec<RandSharingTransform>,
    circ: PackedCircuit,
    mpc: MPCContext,
}

impl GarbleContext {
    pub fn new(circ: PackedCircuit, context: MPCContext) -> Self {
        Self {
            transforms: Self::compute_transforms(&circ, &context),
            circ,
            mpc: context,
        }
    }

    fn compute_transforms(circ: &PackedCircuit, context: &MPCContext) -> Vec<RandSharingTransform> {
        // Default positions.
        let def_pos = context.pss.default_pos(context.gf.as_ref());

        let offset = context.pss.pos_offset(context.gf.as_ref());
        let wires_to_pos = |wires: &[WireID]| -> Vec<GFElement> {
            wires.iter().map(|x| offset + context.gf.get(*x)).collect()
        };

        let mut pos_list = Vec::new();
        for gate in circ.gates() {
            match gate {
                PackedGate::Xor(ginf) => {
                    pos_list.push((wires_to_pos(&ginf.inp[0]), wires_to_pos(&ginf.out)));
                    pos_list.push((wires_to_pos(&ginf.inp[1]), wires_to_pos(&ginf.out)));
                }
                PackedGate::And(ginf) => {
                    pos_list.push((wires_to_pos(&ginf.inp[0]), wires_to_pos(&ginf.out)));
                    pos_list.push((wires_to_pos(&ginf.inp[1]), wires_to_pos(&ginf.out)));
                }
                PackedGate::Inv(ginf) => {
                    pos_list.push((wires_to_pos(&ginf.inp[0]), wires_to_pos(&ginf.out)));
                }
            }
        }

        let mut transforms = Vec::new();

        for pos_batch in pos_list.chunks(context.l) {
            let mut f_trans: Vec<Box<dyn Fn(&[PackedShare]) -> Vec<PackedShare>>> = Vec::new();
            let mut opos_batch = Vec::new();
            let mut npos_batch = Vec::new();

            for (inp_wires, npos) in pos_batch {
                // Remove repetitions in wire IDs.
                // The linear map will ensure repetitions are handled correctly.
                let mut opos = inp_wires.clone();
                opos.sort_unstable();
                opos.dedup();

                if opos.len() < context.l {
                    opos.extend_from_slice(&def_pos[opos.len()..context.l]);
                }

                // Build the transform that orders and repeats secrets correctly.
                let mut wire_lookup = HashMap::new();
                for (i, w) in opos.iter().enumerate() {
                    wire_lookup.insert(*w, i);
                }

                let output = inp_wires.clone();
                let vec_len = context.l;
                let f = move |v: &[GFElement]| -> Vec<GFElement> {
                    let mut res: Vec<_> = output
                        .iter()
                        .map(|w| v[*wire_lookup.get(w).unwrap()])
                        .collect();

                    // Ensure that the output vector is of length `l` by concatenating appropriate
                    // length suffix of `v`.
                    if res.len() < vec_len {
                        res.extend_from_slice(&v[res.len()..vec_len]);
                    }

                    res
                };

                // Ensure that the number of new positions is `l`.
                let mut npos = npos.clone();
                if npos.len() < context.l {
                    npos.extend_from_slice(&opos[npos.len()..context.l]);
                }

                opos_batch.push(opos);
                f_trans.push(Box::new(f));
                npos_batch.push(npos);
            }

            transforms.push(RandSharingTransform::new(
                context.id,
                &opos_batch,
                &npos_batch,
                &f_trans,
                context.pss.as_ref(),
                context.gf.as_ref(),
            ));
        }

        transforms
    }
}
