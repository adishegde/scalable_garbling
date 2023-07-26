use argh::FromArgs;
use scalable_mpc::circuit::{Circuit, PackedCircuit};
use scalable_mpc::math::galois::GF;
use scalable_mpc::protocol::{preproc, MPCContext};
use scalable_mpc::sharing;
use std::path::PathBuf;
use std::sync::Arc;

const W: u8 = 18;

/// Compute size of preprocessing material.
#[derive(FromArgs)]
struct PreProcSize {
    /// file containing circuit description
    #[argh(option)]
    circ: String,

    /// number of parties
    #[argh(option)]
    num_parties: u32,

    /// corruption threshold
    #[argh(option)]
    threshold: u32,

    /// packing parameter
    #[argh(option)]
    packing_param: u32,

    /// base 2 log of LPN bernoulli error probability
    #[argh(option, default = "2")]
    lpn_error_bias: usize,

    /// length of LPN key
    #[argh(option, default = "127")]
    lpn_key_len: usize,

    /// length of expanded LPN pseudorandomness
    #[argh(option, default = "555")]
    lpn_mssg_len: usize,
}

fn benchmark(circ: PackedCircuit, opts: PreProcSize) {
    GF::<W>::init().unwrap();

    let n = opts.num_parties;

    let defpos: Vec<GF<W>> = sharing::PackedSharing::default_pos(n, opts.packing_param);
    let pss = Arc::new(sharing::PackedSharing::new(
        opts.threshold + opts.packing_param - 1,
        n,
        &defpos,
    ));
    let pss_n = Arc::new(sharing::PackedSharing::new(n - 1, n, &defpos));

    let context = MPCContext {
        id: 0,
        n: n as usize,
        t: opts.threshold as usize,
        l: opts.packing_param as usize,
        lpn_tau: opts.lpn_error_bias,
        lpn_key_len: opts.lpn_key_len,
        lpn_mssg_len: opts.lpn_mssg_len,
        pss,
        pss_n,
    };

    let desc = preproc::PreProc::describe(&circ, &context);
    let preproc = preproc::PreProc::dummy(0, desc, &context);
    let preproc_size = bincode::serialized_size(&preproc).unwrap();

    println!("--- Preproc material info ---");
    println!("Mask shares: {}", desc.masks);
    println!("Key shares: {}", desc.keys.0 * desc.keys.1 * desc.keys.2);
    println!("Random shares: {}", desc.randoms);
    println!("Random zeros: {}", desc.zeros);
    println!("Random errors: {}", desc.errors);
    println!("\nTotal preproc size: {} bytes", preproc_size);

    let num_rand_shares = desc.keys.0 * desc.keys.1 * desc.keys.2 + desc.randoms + 3 * desc.errors;
    let num_zero_shares = desc.zeros + 2 * desc.errors;
    let num_bit_shares = desc.masks + 2 * desc.errors;

    println!("\n--- Preproc generation info ---");
    println!("Random shares: {}", num_rand_shares);
    println!("Zero shares: {}", num_zero_shares);
    println!("Bit shares: {}", num_bit_shares);
}

fn main() {
    let opts: PreProcSize = argh::from_env();

    let circ = {
        let path = PathBuf::from(opts.circ.clone());
        let circ = Circuit::from_bristol_fashion(&path);
        circ.pack(opts.packing_param)
    };

    benchmark(circ, opts);
}
