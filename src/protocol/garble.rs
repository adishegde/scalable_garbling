use super::core::RandSharingTransform;
use super::{MPCContext, PartyID};
use crate::circuit::{PackedCircuit, PackedGate, PackedGateInfo, WireID};
use crate::math;
use crate::math::galois::{GFElement, GF};
use crate::sharing::{PackedShare, PackedSharing};
use std::collections::HashMap;

fn wires_to_pos(wires: &[WireID], pss: &PackedSharing, gf: &GF) -> Vec<GFElement> {
    let offset = pss.pos_offset(gf);
    let mut pos: Vec<_> = wires.iter().map(|w| offset + gf.get(*w)).collect();
    pos.sort_unstable();
    pos.dedup();
    pos
}

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

        let gen_coeffs_for_block = |wires: &[WireID]| -> Vec<_> {
            let pos = wires_to_pos(wires, pss, gf);

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
        inp: Vec<PackedShare>,
        ginf: PackedGateInfo<N>,
        gf: &GF,
    ) -> [PackedShare; N] {
        let mut output = [gf.zero(); N];
        let coeffs_list = &self.coeffs[*self.lookup.get(&ginf.out[0]).unwrap()];

        for i in 0..N {
            // NOTE: This uses the internals of how wires_to_pos is implemented. Any changes to
            // the function should also be reflected here.
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
    selector: WireSelector,
}

impl GarbleContext {
    pub fn new(circ: PackedCircuit, context: MPCContext) -> Self {
        let selector =
            WireSelector::new(context.id, &circ, context.pss.as_ref(), context.gf.as_ref());

        Self {
            transforms: Self::compute_transforms(&circ, &context),
            circ,
            selector,
        }
    }

    fn compute_transforms(circ: &PackedCircuit, context: &MPCContext) -> Vec<RandSharingTransform> {
        let mut wires_list = Vec::new();
        for gate in circ.gates() {
            match gate {
                PackedGate::Xor(ginf) => {
                    wires_list.push((ginf.inp[0].clone(), ginf.out.clone()));
                    wires_list.push((ginf.inp[1].clone(), ginf.out.clone()));
                }
                PackedGate::And(ginf) => {
                    wires_list.push((ginf.inp[0].clone(), ginf.out.clone()));
                    wires_list.push((ginf.inp[1].clone(), ginf.out.clone()));
                }
                PackedGate::Inv(ginf) => {
                    wires_list.push((ginf.inp[0].clone(), ginf.out.clone()));
                }
            }
        }

        let mut transforms = Vec::new();
        let offset = context.pss.pos_offset(context.gf.as_ref());

        for wires_block in wires_list.chunks(context.l) {
            let mut f_trans: Vec<Box<dyn Fn(&[PackedShare]) -> Vec<PackedShare>>> = Vec::new();
            let mut opos_block = Vec::new();
            let mut npos_block = Vec::new();

            for (inp_wires, out_wires) in wires_block {
                let mut inp_dedup = inp_wires.clone();
                inp_dedup.sort_unstable();
                inp_dedup.dedup();

                // Build the transform that orders and repeats secrets correctly.
                let f = {
                    let mut wire_lookup = HashMap::new();
                    for (i, w) in inp_dedup.iter().enumerate() {
                        wire_lookup.insert(*w, i);
                    }

                    let inp_wires = inp_wires.clone();
                    move |v: &[GFElement]| -> Vec<GFElement> {
                        inp_wires
                            .iter()
                            .map(|w| v[*wire_lookup.get(w).unwrap()])
                            .collect()
                    }
                };

                let opos: Vec<_> = inp_dedup
                    .into_iter()
                    .map(|x| offset + context.gf.get(x))
                    .collect();
                let npos = out_wires
                    .into_iter()
                    .map(|x| offset + context.gf.get(*x))
                    .collect();

                opos_block.push(opos);
                f_trans.push(Box::new(f));
                npos_block.push(npos);
            }

            transforms.push(RandSharingTransform::new(
                context.id,
                &opos_block,
                &npos_block,
                &f_trans,
                context.pss.as_ref(),
                context.gf.as_ref(),
            ));
        }

        transforms
    }
}
