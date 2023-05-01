use argh::FromArgs;
use json;
use scalable_mpc::circuit::{Circuit, PackedCircuit};
use scalable_mpc::protocol::{garble, network, preproc, MPCContext};
use scalable_mpc::{math, sharing, PartyID};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

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

    /// file to save benchmark data
    #[argh(option)]
    save: Option<String>,
}

async fn benchmark(circ: PackedCircuit, ipaddrs: Vec<String>, opts: Garble) {
    let n: u32 = ipaddrs.len().try_into().unwrap();

    let gf = Arc::new(math::galois::GF::new(opts.gf_width).unwrap());
    let pss = Arc::new(sharing::PackedSharing::new(
        n,
        opts.threshold,
        opts.packing_param,
        gf.as_ref(),
    ));
    let (stats, net) = network::setup_tcp_network(opts.id, &ipaddrs).await;

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

    let mut bench_data = json::JsonValue::new_object();

    let context = {
        let start = Instant::now();
        let context = garble::GarbleContextData::new(circ, mpcctx);
        let comp_time: u64 = start.elapsed().as_millis().try_into().unwrap();
        bench_data["comp_time"] = comp_time.into();
        Arc::new(context)
    };

    let start = Instant::now();
    garble::garble(b"".to_vec(), preproc, context).await;
    let comp_time: u64 = start.elapsed().as_millis().try_into().unwrap();
    bench_data["runtime"] = comp_time.into();

    bench_data["comms"] = json::JsonValue::new_array();
    let mut total_comm = 0;
    for i in 0..(n.try_into().unwrap()) {
        let (net_data, proto_data) = stats.party(i).await;
        total_comm += proto_data;
        bench_data["comms"]
            .push(json::array![net_data, proto_data])
            .unwrap();
    }

    println!("--- Party {} ---", opts.id);
    println!("One time computation: {} ms", bench_data["comp_time"]);
    println!("Garbling runtime: {} ms", bench_data["runtime"]);
    println!("Garbling communication: {} bytes", total_comm);

    if let Some(save_path) = opts.save {
        let bench_proto = b"benchmark communication protocol".to_vec();

        let chan = net.channel(&bench_proto).await;
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

            let chan = net.channel(&bench_proto).await;
            network::message_from_each_party(chan.receiver(), n as usize)
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

fn main() {
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

    scalable_mpc::block_on(benchmark(circ, ipaddrs, opts));
}
