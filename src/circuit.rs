use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub type WireID = u32;

#[derive(Copy, Clone)]
pub struct GateInfo<const N: usize> {
    pub inp: [WireID; N],
    pub out: WireID,
}

#[derive(Copy, Clone)]
pub enum Gate {
    Xor(GateInfo<2>),
    And(GateInfo<2>),
    Inv(GateInfo<1>),
}

#[derive(Clone)]
pub struct Circuit {
    // Gates should be topologically ordered so that they can be evaluated in sequence.
    gates: Vec<Gate>,
    inputs: Vec<Vec<WireID>>,
    outputs: Vec<Vec<WireID>>,
    num_wires: u32,
}

#[derive(Clone)]
pub struct PackedGateInfo<const N: usize> {
    pub inp: [Vec<WireID>; N],
    pub out: Vec<WireID>,
}

impl<const N: usize> PackedGateInfo<N> {
    fn from(gates: &[GateInfo<N>]) -> Self {
        let inp = core::array::from_fn(|i| gates.iter().map(|g| g.inp[i]).collect());
        let out = gates.iter().map(|g| g.out).collect();

        PackedGateInfo { inp, out }
    }
}

#[derive(Clone)]
pub enum PackedGate {
    Xor(PackedGateInfo<2>),
    And(PackedGateInfo<2>),
    Inv(PackedGateInfo<1>),
}

// Inputs and outputs are also packed into blocks of `gates_per_block`.
#[derive(Clone)]
pub struct PackedCircuit {
    gates: Vec<PackedGate>,
    inputs: Vec<Vec<WireID>>,
    outputs: Vec<Vec<WireID>>,
    num_wires: u32,
    gates_per_block: u32,
}

impl Circuit {
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    pub fn inputs(&self) -> &[Vec<WireID>] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Vec<WireID>] {
        &self.outputs
    }

    pub fn num_wires(&self) -> u32 {
        self.num_wires
    }

    pub fn from_bristol_fashion(path: &Path) -> Self {
        let file = File::open(path).expect("Circuit file doesn't exist");
        let mut reader = BufReader::new(file);

        let mut meta_data = String::new();
        for _ in 0..4 {
            reader.read_line(&mut meta_data).unwrap();
        }
        let mut meta_iter = meta_data.split_ascii_whitespace();
        let mut next_meta_number = || -> u32 {
            let s = meta_iter.next().unwrap();
            s.trim().parse::<u32>().unwrap()
        };

        let num_gates = next_meta_number();
        let num_wires = next_meta_number();

        let num_inputs = next_meta_number();
        let inp_lens: Vec<_> = (0..num_inputs).map(|_| next_meta_number()).collect();
        let inputs: Vec<Vec<WireID>> = inp_lens
            .into_iter()
            .scan(0, |state, v| {
                *state = *state + v;
                Some(((*state - v)..(*state)).collect())
            })
            .collect();

        let num_outputs = next_meta_number();
        let out_lens: Vec<_> = (0..num_outputs).map(|_| next_meta_number()).collect();
        let outputs: Vec<Vec<WireID>> = out_lens
            .into_iter()
            .rev()
            .scan(num_wires, |state, v| {
                *state = *state - v;
                Some(((*state)..(*state + v)).collect())
            })
            .collect();

        let mut gates = Vec::new();

        for _ in 0..num_gates {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let line: Vec<_> = line.split_ascii_whitespace().collect();
            let gate_type = line.last().unwrap().to_owned();
            let mut iter = line[..(line.len() - 1)]
                .into_iter()
                .map(|s| s.parse::<u32>().unwrap());

            let fan_in = iter.next().unwrap();
            let _fan_out = iter.next().unwrap();

            let inputs: Vec<_> = (0..fan_in).map(|_| iter.next().unwrap()).collect();
            let output = iter.next().unwrap();

            match gate_type {
                "AND" => gates.push(Gate::And(GateInfo {
                    inp: [inputs[0], inputs[1]],
                    out: output,
                })),
                "XOR" => gates.push(Gate::Xor(GateInfo {
                    inp: [inputs[0], inputs[1]],
                    out: output,
                })),
                "INV" => gates.push(Gate::Inv(GateInfo {
                    inp: [inputs[0]],
                    out: output,
                })),
                _ => panic!("Invalid gate"),
            }
        }

        Self {
            gates,
            inputs,
            outputs,
            num_wires,
        }
    }

    pub fn eval(&self, inputs: &[Vec<bool>]) -> Vec<Vec<bool>> {
        let mut wires = vec![false; self.num_wires as usize];

        for (inp_wires, inps) in self.inputs.iter().zip(inputs.iter()) {
            for (&inp_wire, &inp) in inp_wires.iter().zip(inps.iter()) {
                wires[inp_wire as usize] = inp;
            }
        }

        for gate in &self.gates {
            match gate {
                Gate::Xor(ginf) => {
                    let left_val = wires[ginf.inp[0] as usize];
                    let right_val = wires[ginf.inp[1] as usize];
                    wires[ginf.out as usize] = !(left_val == right_val);
                }
                Gate::And(ginf) => {
                    let left_val = wires[ginf.inp[0] as usize];
                    let right_val = wires[ginf.inp[1] as usize];
                    wires[ginf.out as usize] = left_val & right_val;
                }
                Gate::Inv(ginf) => {
                    let inp_val = wires[ginf.inp[0] as usize];
                    wires[ginf.out as usize] = !inp_val;
                }
            }
        }

        let mut outputs = Vec::with_capacity(self.outputs.len());
        for out_wires in &self.outputs {
            let mut output = Vec::with_capacity(out_wires.len());
            for out_wire in out_wires {
                output.push(wires[*out_wire as usize]);
            }

            outputs.push(output);
        }

        outputs
    }

    pub fn pack(self, gates_per_block: u32) -> PackedCircuit {
        let mut g_and = Vec::new();
        let mut g_xor = Vec::new();
        let mut g_inv = Vec::new();

        for gate in self.gates {
            match gate {
                Gate::Xor(ginf) => {
                    g_xor.push(ginf);
                }
                Gate::And(ginf) => {
                    g_and.push(ginf);
                }
                Gate::Inv(ginf) => {
                    g_inv.push(ginf);
                }
            }
        }

        let gates: Vec<_> = g_and
            .chunks(gates_per_block as usize)
            .map(|x| PackedGate::And(PackedGateInfo::from(x)))
            .chain(
                g_xor
                    .chunks(gates_per_block as usize)
                    .map(|x| PackedGate::Xor(PackedGateInfo::from(x))),
            )
            .chain(
                g_inv
                    .chunks(gates_per_block as usize)
                    .map(|x| PackedGate::Inv(PackedGateInfo::from(x))),
            )
            .collect();

        let flattend_and_partition = |v: Vec<Vec<_>>| {
            let flattened: Vec<_> = v.into_iter().flatten().collect();
            flattened
                .chunks(gates_per_block as usize)
                .map(|x| x.to_vec())
                .collect()
        };

        let inputs = flattend_and_partition(self.inputs);
        let outputs = flattend_and_partition(self.outputs);

        PackedCircuit {
            gates,
            inputs,
            outputs,
            num_wires: self.num_wires,
            gates_per_block,
        }
    }
}

impl PackedCircuit {
    pub fn gates(&self) -> &[PackedGate] {
        &self.gates
    }

    pub fn inputs(&self) -> &[Vec<WireID>] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Vec<WireID>] {
        &self.outputs
    }

    pub fn num_wires(&self) -> u32 {
        self.num_wires
    }

    pub fn gates_per_block(&self) -> u32 {
        self.gates_per_block
    }
}
