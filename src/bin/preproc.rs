use argh::FromArgs;
use scalable_mpc::circuit::{Circuit, PackedCircuit};
use scalable_mpc::math::galois::GF;
use scalable_mpc::protocol::{network, preproc, MPCContext};
use scalable_mpc::{math, sharing, PartyID, Stats};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;

const W: u8 = 18;

/// Benchmark preprocessing phase.
#[derive(FromArgs)]
struct PreProc {
    /// id of the party
    #[argh(option)]
    id: PartyID,

    /// file containing list of IP address of each party
    #[argh(option)]
    net: String,

    /// file containing circuit description
    #[argh(option)]
    circ: String,

    /// corruption threshold
    #[argh(option)]
    threshold: u32,

    /// packing parameter
    #[argh(option)]
    packing_param: u32,

    /// file containing binary super invertible matrix
    #[argh(option)]
    binsup_mat: String,

    /// base 2 log of LPN bernoulli error probability
    #[argh(option, default = "2")]
    lpn_error_bias: usize,

    /// length of LPN key
    #[argh(option, default = "127")]
    lpn_key_len: usize,

    /// length of expanded LPN pseudorandomness
    #[argh(option, default = "555")]
    lpn_mssg_len: usize,

    /// number of repetitions
    #[argh(option, default = "1")]
    reps: usize,

    /// file to save benchmark data
    #[argh(option)]
    save: Option<String>,

    /// number of threads
    #[argh(option, default = "1")]
    threads: usize,
}

async fn benchmark(circ: PackedCircuit, ipaddrs: Vec<String>, opts: PreProc) {
    let n: u32 = ipaddrs.len().try_into().unwrap();

    GF::<W>::init().unwrap();

    let defpos: Vec<GF<W>> = sharing::PackedSharing::default_pos(n, opts.packing_param);
    let pss = Arc::new(sharing::PackedSharing::new(
        opts.threshold + opts.packing_param - 1,
        n,
        &defpos,
    ));
    let pss_n = Arc::new(sharing::PackedSharing::new(n - 1, n, &defpos));

    let bin_supinv_matrix = {
        let path = PathBuf::from(opts.binsup_mat);
        Arc::new(math::binary_super_inv_matrix(&path))
    };

    let supinv_matrix = Arc::new(math::super_inv_matrix(
        n as usize,
        (n - opts.threshold) as usize,
    ));

    let context = MPCContext {
        id: opts.id,
        n: n as usize,
        t: opts.threshold as usize,
        l: opts.packing_param as usize,
        lpn_tau: opts.lpn_error_bias,
        lpn_key_len: opts.lpn_key_len,
        lpn_mssg_len: opts.lpn_mssg_len,
        pss,
        pss_n,
    };
    let rcontext = preproc::RandContext::new(supinv_matrix.clone(), &context);
    let zcontext = preproc::ZeroContext::new(supinv_matrix, &context);
    let bcontext = preproc::RandBitContext::new(bin_supinv_matrix, &context);

    let desc = preproc::PreProc::describe(&circ, &context);

    let bench_proto = b"benchmark communication protocol".to_vec();
    let mut bench_data = json::JsonValue::new_array();

    println!("--- Party {} ---", opts.id);

    let (stats, net) = network::setup_tcp_network(opts.id, &ipaddrs).await;
    println!("Connected to network.");
    std::io::stdout().flush().unwrap();

    for i in 0..opts.reps {
        network::sync(bench_proto.clone(), &net, n as usize).await;

        let context = context.clone();
        let rcontext = rcontext.clone();
        let zcontext = zcontext.clone();
        let bcontext = bcontext.clone();

        let gproto_id = b"".to_vec();
        let net = net.clone();

        let start = Stats::now(&stats, n).await;
        preproc::preproc(gproto_id, desc, context, rcontext, zcontext, bcontext, net).await;
        let (runtime, comm) = start.elapsed(&stats).await;

        println!();
        println!("--- Repetition {} ---", i);
        println!("Runtime: {} ms", runtime);
        println!(
            "Communication: {} bytes",
            comm.iter().map(|(x, _)| *x).sum::<u64>()
        );
        std::io::stdout().flush().unwrap();

        bench_data
            .push(json::object! {
                time: runtime,
                comm: comm.into_iter().map(|(x, y)| json::array![x, y]).collect::<Vec<_>>()
            })
            .unwrap();
    }

    if let Some(save_path) = opts.save {
        let data = bench_data.dump().as_bytes().to_vec();

        net.send(network::SendMessage {
            to: network::Recipient::One(0),
            proto_id: bench_proto.clone(),
            data,
        });

        if opts.id == 0 {
            let mut save_data = json::object! {
                details: {
                    n: n,
                    t: opts.threshold,
                    l: opts.packing_param,
                    circ: opts.circ,
                    lpn_tau: opts.lpn_error_bias,
                    lpn_key_len: opts.lpn_key_len,
                    lpn_mssg_len: opts.lpn_mssg_len,
                    reps: opts.reps,
                    threads: opts.reps
                },
                benchmarks: json::JsonValue::new_array()
            };

            network::message_from_each_party(bench_proto.clone(), &net, n as usize)
                .await
                .into_iter()
                .for_each(|data| {
                    let data = String::from_utf8(data).unwrap();
                    let data = json::parse(&data).unwrap();
                    save_data["benchmarks"].push(data).unwrap();
                });

            let save_path = PathBuf::from(save_path);
            std::fs::write(save_path, save_data.dump()).expect("Can write to save file");

            net.send(network::SendMessage {
                to: network::Recipient::All,
                proto_id: bench_proto,
                data: b"done".to_vec(),
            });
        } else {
            net.recv(bench_proto).await;
        }
    }
}

fn main() {
    let opts: PreProc = argh::from_env();

    let ipaddrs: Vec<_> = {
        let path = PathBuf::from(opts.net.clone());
        let file =
            File::open(path).expect("Network file should be created with list of IP addresses.");
        BufReader::new(file).lines().map(|l| l.unwrap()).collect()
    };

    let circ = {
        let path = PathBuf::from(opts.circ.clone());
        let circ = Circuit::from_bristol_fashion(&path);
        circ.pack(opts.packing_param)
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads)
        .build_global()
        .unwrap();

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(benchmark(circ, ipaddrs, opts));
}
