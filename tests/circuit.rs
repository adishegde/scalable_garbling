use scalable_mpc::circuit::Circuit;
use std::path::PathBuf;

#[test]
fn load_circuit_file() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/sub64.txt");

    let int_to_bits = |v| (0..64).map(|i| ((v >> i) & 1u64) == 1u64).collect();
    let bits_to_int = |bits: &[bool]| {
        bits.iter()
            .enumerate()
            .fold(0u64, |acc, (i, v)| if *v { acc + (1 << i) } else { acc })
    };

    let inp_a: u64 = 23948;
    let inp_b: u64 = 48;

    let mut inputs = Vec::new();
    inputs.push(int_to_bits(inp_a));
    inputs.push(int_to_bits(inp_b));

    let circ = Circuit::from_bristol_fashion(&path);
    assert_eq!(circ.gates().len(), 439);
    assert_eq!(circ.inputs().len(), 2);
    assert_eq!(circ.outputs().len(), 1);
    assert_eq!(circ.num_wires(), 567);

    let out = bits_to_int(&circ.eval(&inputs)[0]);

    assert_eq!(inp_a - inp_b, out);
}

#[test]
fn pack_circuit() {
    let circ = {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data/sub64.txt");
        Circuit::from_bristol_fashion(&path)
    };

    let pcirc = circ.pack(4);

    assert_eq!(pcirc.gates().len(), 111);
    assert_eq!(pcirc.inputs().len(), 32);
    assert_eq!(pcirc.outputs().len(), 16);
    assert_eq!(pcirc.num_wires(), 567);
    assert_eq!(pcirc.gates_per_block(), 4);
}
