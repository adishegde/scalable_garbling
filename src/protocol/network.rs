use super::{ProtoHandle, ProtocolID};
use crate::PartyID;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::{oneshot, RwLock};
use tokio::task::spawn;

#[derive(Clone, Copy, Debug)]
pub enum Recipient {
    One(PartyID),
    All,
}

#[derive(Clone, Debug)]
pub struct SendMessage {
    pub proto_id: ProtocolID,
    pub to: Recipient,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct ReceivedMessage {
    pub proto_id: ProtocolID,
    pub from: PartyID,
    pub data: Vec<u8>,
}

#[derive(Clone)]
pub struct Stats(Arc<RwLock<Vec<(u64, u64)>>>);

impl Stats {
    fn new(num_parties: usize) -> Self {
        Stats(Arc::new(RwLock::new(vec![(0, 0); num_parties])))
    }

    async fn increment(&self, from: PartyID, proto_bytes: u64, net_bytes: u64) {
        let mut vals = self.0.write().await;
        vals[from as usize].0 += proto_bytes;
        vals[from as usize].1 += net_bytes;
    }

    pub async fn party(&self, from: PartyID) -> (u64, u64) {
        self.0.read().await[from as usize]
    }

    pub async fn total(&self) -> (u64, u64) {
        let mut proto_bytes = 0;
        let mut net_bytes = 0;
        let vals = self.0.read().await;

        for val in vals.iter() {
            proto_bytes += val.0;
            net_bytes += val.1;
        }

        (proto_bytes, net_bytes)
    }
}

pub type Network = ProtoHandle<SendMessage, ReceivedMessage>;

pub async fn setup_tcp_network(party_id: PartyID, addresses: &[String]) -> (Stats, Network) {
    let num_parties: u16 = addresses
        .len()
        .try_into()
        .expect("Number of parties to be at most 16 bytes long.");

    let tcp_streams = connect_to_peers(party_id, addresses).await;

    // For a message being sent to the i-th party, the flow is as follows:
    //   msg -> net_inp
    //   net_inp -> party_inp[i]
    //   party_inp[i] -> tcp_stream[i] (send router task)
    //   tcp_stream[sender] -> party_out (per party receiver task)
    //   party_out -> net_out (through requests sent by received task)
    let (net_inp_s, net_inp_r) = unbounded_channel();
    let (party_out_s, party_out_r) = unbounded_channel();
    let (net_out_s, net_out_r) = unbounded_channel();
    let channel_builder = ProtoHandle::new(net_inp_s, net_out_s);
    let stats = Stats::new(num_parties as usize);

    let mut party_inp_s = Vec::with_capacity(num_parties as usize);
    for (pid, tcp_stream) in tcp_streams.into_iter().enumerate() {
        let (tx, rx) = unbounded_channel();
        party_inp_s.push(tx);

        match tcp_stream {
            Some(stream) => {
                let (read_stream, write_stream) = stream.into_split();

                spawn(party_sender(rx, write_stream));
                spawn(party_receiver(
                    pid.try_into().unwrap(),
                    read_stream,
                    party_out_s.clone(),
                ));
            }
            None => {
                assert_eq!(pid, party_id as usize);
            }
        }
    }

    spawn(send_router(
        party_id,
        num_parties,
        net_inp_r,
        party_inp_s,
        party_out_s,
    ));
    spawn(receive_router(party_out_r, net_out_r, stats.clone()));

    (stats, channel_builder)
}

pub async fn setup_local_network(num_parties: u16) -> Vec<(Stats, Network)> {
    let (router_s, mut router_r) = unbounded_channel();

    let mut res = Vec::with_capacity(num_parties as usize);
    let mut party_out = Vec::with_capacity(num_parties as usize);

    for pid in 0..(num_parties) {
        let router_s = router_s.clone();
        let (net_inp_s, mut net_inp_r) = unbounded_channel();
        let (net_out_s, net_out_r) = unbounded_channel();
        let (party_out_s, party_out_r) = unbounded_channel();
        let stats = Stats::new(num_parties as usize);

        spawn(async move {
            while let Some(mssg) = net_inp_r.recv().await {
                router_s.send((pid, mssg)).unwrap();
            }
        });
        spawn(receive_router(party_out_r, net_out_r, stats.clone()));

        res.push((stats, Network::new(net_inp_s, net_out_s)));
        party_out.push(party_out_s);
    }

    spawn(async move {
        while let Some((sender_id, mssg)) = router_r.recv().await {
            match mssg.to {
                Recipient::One(pid) => {
                    party_out[pid as usize]
                        .send(ReceivedMessage {
                            proto_id: mssg.proto_id,
                            from: sender_id,
                            data: mssg.data,
                        })
                        .unwrap();
                }
                Recipient::All => {
                    for tx in party_out.iter() {
                        tx.send(ReceivedMessage {
                            proto_id: mssg.proto_id.clone(),
                            from: sender_id,
                            data: mssg.data.clone(),
                        })
                        .unwrap();
                    }
                }
            }
        }
    });

    res
}

pub async fn sync(proto_id: ProtocolID, net: &Network, num_parties: usize) {
    let data = b"sync".to_vec();

    net.send(SendMessage {
        proto_id: proto_id.clone(),
        to: Recipient::All,
        data,
    })
    .await;

    message_from_each_party(proto_id, net, num_parties).await;
}

/// Waits to receive a message from each party and then returns the list of messages in order of
/// the party ID.
/// If multiple messages are received from the same party, the first one is returned and the latter
/// ones are dropped.
pub async fn message_from_each_party(
    proto_id: ProtocolID,
    net: &Network,
    num_parties: usize,
) -> Vec<Vec<u8>> {
    let mut mssgs = vec![Vec::new(); num_parties];
    let mut has_sent = vec![false; num_parties];
    let mut counter = 0;

    while counter != num_parties {
        let mssg = net.recv(proto_id.clone()).await;
        if !has_sent[mssg.from as usize] {
            mssgs[mssg.from as usize] = mssg.data;
            counter += 1;
            has_sent[mssg.from as usize] = true;
        }
    }

    mssgs
}

async fn connect_to_peers(party_id: PartyID, addresses: &[String]) -> Vec<Option<TcpStream>> {
    let num_parties: u16 = addresses
        .len()
        .try_into()
        .expect("Number of parties to be at most 16 bytes long.");
    let my_address = addresses[party_id as usize].clone();

    let accept_incoming = spawn(async move {
        let (_, port) = my_address.split_once(':').unwrap();
        let mut listen_addr = String::from("0.0.0.0:");
        listen_addr.push_str(port);

        let listener = TcpListener::bind(listen_addr).await.unwrap();

        let mut streams: Vec<_> = (0..num_parties).map(|_| None).collect();

        let expected_connections = num_parties - party_id - 1;

        for _ in 0..expected_connections {
            let (mut stream, _) = listener.accept().await.unwrap();
            stream.set_nodelay(true).unwrap();

            let pid = stream
                .read_u16()
                .await
                .expect("Party ID should be readable upon connecting.");
            streams[pid as usize] = Some(stream);
        }

        streams
    });

    let mut connect_to_peers_futures = Vec::with_capacity(party_id as usize);
    for peer_address in addresses.iter().take(party_id as usize).cloned() {
        connect_to_peers_futures.push(spawn(async move {
            let peer_address = peer_address;
            let mut stream = {
                loop {
                    if let Ok(stream) = TcpStream::connect(&peer_address).await {
                        break stream;
                    }

                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            };

            stream.set_nodelay(true).unwrap();
            stream
                .write_u16(party_id)
                .await
                .expect("Party ID should be writeable upon connecting.");
            stream.flush().await.expect("TCP stream is flushble.");
            stream
        }));
    }

    let mut streams = accept_incoming.await.unwrap();
    for (i, future) in connect_to_peers_futures.into_iter().enumerate() {
        let stream = future.await.unwrap();
        streams[i] = Some(stream);
    }
    streams
}

async fn send_router(
    party_id: PartyID,
    num_parties: u16,
    mut net_inp_r: UnboundedReceiver<SendMessage>,
    party_inp_s: Vec<UnboundedSender<SendMessage>>,
    party_out_s: UnboundedSender<ReceivedMessage>,
) {
    while let Some(mssg) = net_inp_r.recv().await {
        match mssg.to {
            Recipient::One(pid) if pid == party_id => {
                party_out_s
                    .send(ReceivedMessage {
                        proto_id: mssg.proto_id,
                        from: party_id,
                        data: mssg.data,
                    })
                    .unwrap();
            }
            Recipient::One(pid) => party_inp_s[pid as usize].send(mssg).unwrap(),
            Recipient::All => {
                for i in 0..num_parties {
                    if i == party_id {
                        party_out_s
                            .send(ReceivedMessage {
                                proto_id: mssg.proto_id.clone(),
                                from: party_id,
                                data: mssg.data.clone(),
                            })
                            .unwrap()
                    } else {
                        party_inp_s[i as usize].send(mssg.clone()).unwrap();
                    }
                }
            }
        }
    }
}

async fn party_sender(
    mut incoming: UnboundedReceiver<SendMessage>,
    mut tcp_stream: OwnedWriteHalf,
) {
    while let Some(mssg) = incoming.recv().await {
        let mssg_len = mssg.proto_id.len() + mssg.data.len() + 1;
        let id_len: u8 = mssg
            .proto_id
            .len()
            .try_into()
            .expect("Protocol ID to be at most 255 bytes.");

        let mut buffer = Vec::with_capacity(mssg_len + vint64::encoded_len(mssg_len as u64));
        buffer.extend_from_slice(vint64::encode(mssg_len as u64).as_ref());
        buffer.push(id_len);
        buffer.extend_from_slice(&mssg.proto_id);
        buffer.extend_from_slice(&mssg.data);

        tcp_stream
            .write_all(&buffer)
            .await
            .expect("TCP stream is writable.");
        tcp_stream.flush().await.expect("TCP stream is flushable.");
    }
}

async fn party_receiver(
    pid: PartyID,
    mut incoming: OwnedReadHalf,
    party_out_s: UnboundedSender<ReceivedMessage>,
) {
    let mut bytes = [0; 9];

    while incoming.read_exact(&mut bytes[..1]).await.is_ok() {
        let mssg_len = {
            let enc_len = vint64::decoded_len(bytes[0]);
            if enc_len != 1 {
                incoming
                    .read_exact(&mut bytes[1..enc_len])
                    .await
                    .expect("TCP stream is readable.");
            }
            vint64::decode(&mut &bytes[..enc_len]).unwrap() as usize
        };

        let mut mssg = vec![0; mssg_len];
        incoming
            .read_exact(&mut mssg)
            .await
            .expect("TCP stream is readable.");

        let id_len = usize::from(mssg[0]);
        let proto_id = mssg[1..(id_len + 1)].to_vec();
        let data = mssg[(id_len + 1)..].to_vec();
        let recv_mssg = ReceivedMessage {
            from: pid,
            proto_id,
            data,
        };

        party_out_s.send(recv_mssg).unwrap();
    }
}

async fn receive_router(
    mut party_out_r: UnboundedReceiver<ReceivedMessage>,
    mut net_out_r: UnboundedReceiver<(ProtocolID, oneshot::Sender<ReceivedMessage>)>,
    stats: Stats,
) {
    let mut mssg_pool: HashMap<ProtocolID, VecDeque<ReceivedMessage>> = HashMap::new();
    let mut handle_pool: HashMap<ProtocolID, oneshot::Sender<ReceivedMessage>> = HashMap::new();

    loop {
        tokio::select! {
            Some(mssg) = party_out_r.recv() => {
                let proto_bytes = mssg.data.len();
                let net_bytes = proto_bytes + mssg.proto_id.len();
                stats.increment(mssg.from, proto_bytes as u64, net_bytes as u64).await;

                match handle_pool.remove(&mssg.proto_id) {
                    Some(handle) => {
                        handle.send(mssg).unwrap();
                    },
                    None => {
                        mssg_pool.entry(mssg.proto_id.clone()).or_insert(VecDeque::new()).push_back(mssg);
                    }
                }
            },
            Some((id, handle)) = net_out_r.recv() => {
                match mssg_pool.get_mut(&id) {
                    Some(buf) => {
                        let mssg = buf.pop_front().unwrap();
                        if buf.is_empty() {
                            mssg_pool.remove(&id);
                        }

                        handle.send(mssg).unwrap();
                    },
                    None => {
                        handle_pool.insert(id, handle);
                    }
                }
            },
            else => break
        }
    }

    while !mssg_pool.is_empty() {
        match net_out_r.recv().await {
            Some((id, handle)) => match mssg_pool.get_mut(&id) {
                Some(buf) => {
                    let mssg = buf.pop_front().unwrap();
                    if buf.is_empty() {
                        mssg_pool.remove(&id);
                    }

                    handle.send(mssg).unwrap();
                }
                None => {
                    panic!("Illegal protocol ID");
                }
            },
            None => break,
        }
    }
}
