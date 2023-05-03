use argh::FromArgs;
use json;
use scalable_mpc::circuit::{Circuit, PackedCircuit};
use scalable_mpc::protocol::{garble, network, preproc, MPCContext};
use scalable_mpc::{math, sharing, PartyID};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

struct Stats {
    time: Instant,
    comm: Vec<(u64, u64)>,
}

impl Stats {
    async fn now(stats: &network::Stats, n: u32) -> Self {
        let mut comm = Vec::with_capacity(n as usize);
        for i in 0..n {
            comm.push(stats.party(i.try_into().unwrap()).await);
        }

        Self {
            time: Instant::now(),
            comm,
        }
    }

    async fn elapsed(&self, stats: &network::Stats) -> (u64, Vec<(u64, u64)>) {
        let time: u64 = self.time.elapsed().as_millis().try_into().unwrap();
        let mut comm = Vec::with_capacity(self.comm.len());
        for i in 0..self.comm.len() {
            let pcom = stats.party(i.try_into().unwrap()).await;
            comm.push((pcom.0 - self.comm[i].0, pcom.1 - self.comm[i].1));
        }

        (time, comm)
    }
}

/// Benchmark garbling phase.
#[derive(FromArgs)]
struct Garble {
    /// id of the party
    #[argh(option)]
    id: PartyID,

    /// file containing list of IP address of each party
    #[argh(option)]
    net: String,

    /// file containing circuit description
    #[argh(option)]
    circ: String,

    /// field size in number of bits
    #[argh(option, default = "18")]
    gf_width: u8,

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
    #[argh(option, default = "246")]
    lpn_key_len: usize,

    /// length of expanded LPN pseudorandomness
    #[argh(option, default = "721")]
    lpn_mssg_len: usize,

    /// number of repetitions
    #[argh(option, default = "1")]
    reps: usize,

    /// file to save benchmark data
    #[argh(option)]
    save: Option<String>,
}

async fn benchmark(circ: PackedCircuit, ipaddrs: Vec<String>, opts: Garble) {
    let n: u32 = ipaddrs.len().try_into().unwrap();

    println!("--- Party {} ---", opts.id);

    let gf = Arc::new(math::galois::GF::new(opts.gf_width).unwrap());
    let pss = Arc::new(sharing::PackedSharing::new(
        n,
        opts.threshold,
        opts.packing_param,
        gf.as_ref(),
    ));

    let (stats, net) = network::setup_tcp_network(opts.id, &ipaddrs).await;
    println!("Connected to network.");

    let mpcctx = MPCContext {
        id: opts.id,
        n: n as usize,
        t: opts.threshold as usize,
        l: opts.packing_param as usize,
        lpn_tau: opts.lpn_error_bias,
        lpn_key_len: opts.lpn_key_len,
        lpn_mssg_len: opts.lpn_mssg_len,
        gf,
        pss,
        net_builder: net.clone(),
    };

    let preproc = preproc::PreProc::dummy(opts.id, 200, &circ, &mpcctx);
    println!("Computed dummy preprocessing.");
    std::io::stdout().flush().unwrap();

    let mut bench_data = json::JsonValue::new_object();

    let context = {
        let start = Instant::now();
        let context = garble::GarbleContextData::new(circ, mpcctx);
        let comp_time: u64 = start.elapsed().as_millis().try_into().unwrap();
        bench_data["onetime_comp"] = comp_time.into();
        Arc::new(context)
    };
    println!();
    println!("Completed onetime circuit dependent computation.");
    println!("Time: {} ms", bench_data["onetime_comp"]);
    std::io::stdout().flush().unwrap();

    let bench_proto = b"benchmark communication protocol".to_vec();
    let chan = net.channel(&bench_proto).await;
    bench_data["garbling"] = json::JsonValue::new_array();

    for i in 0..opts.reps {
        network::sync(bench_proto.clone(), &chan, n as usize).await;
        let preproc = preproc.clone();
        let context = context.clone();
        let gproto_id = b"".to_vec();

        let start = Stats::now(&stats, n).await;
        garble::garble(gproto_id, preproc, context).await;
        let (runtime, comm) = start.elapsed(&stats).await;

        println!();
        println!("--- Repetition {} ---", i);
        println!("Runtime: {} ms", runtime);
        println!(
            "Communication: {} bytes",
            comm.iter().map(|(x, _)| *x).sum::<u64>()
        );
        std::io::stdout().flush().unwrap();

        bench_data["garbling"]
            .push(json::object! {
                time: runtime,
                comm: comm.into_iter().map(|(x, y)| json::array![x, y]).collect::<Vec<_>>()
            })
            .unwrap();
    }

    if let Some(save_path) = opts.save {
        let data = bench_data.dump().as_bytes().to_vec();

        chan.send(network::SendMessage {
            to: network::Recipient::One(0),
            proto_id: bench_proto.clone(),
            data,
        })
        .await;

        if opts.id == 0 {
            let mut save_data = json::object! {
                details: {
                    id: opts.id,
                    n: n,
                    t: opts.threshold,
                    l: opts.packing_param,
                    lpn_tau: opts.lpn_error_bias,
                    lpn_key_len: opts.lpn_key_len,
                    lpn_mssg_len: opts.lpn_mssg_len,
                },
                benchmarks: json::JsonValue::new_array()
            };

            network::message_from_each_party(&chan, n as usize)
                .await
                .into_iter()
                .for_each(|data| {
                    let data = String::from_utf8(data).unwrap();
                    let data = json::parse(&data).unwrap();
                    save_data["benchmarks"].push(data).unwrap();
                });

            let save_path = PathBuf::from(save_path);
            std::fs::write(save_path, save_data.dump()).expect("Can write to save file");
        }
    }
}

#[tokio::main]
async fn main() {
    let opts: Garble = argh::from_env();

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

    benchmark(circ, ipaddrs, opts).await;
}
