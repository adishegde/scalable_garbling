use super::{ProtoChannel, ProtoChannelBuilder, ProtocolID};
use crate::PartyID;
use futures_lite::stream::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::RwLock;
use tokio::task::spawn;
use vint64;

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

#[derive(Debug)]
pub struct RegisterProtocol(pub ProtocolID, pub UnboundedSender<ReceivedMessage>);

#[derive(Clone)]
pub struct Stats(Vec<Arc<RwLock<(u64, u64)>>>);

impl Stats {
    fn new(num_parties: usize) -> Self {
        Stats(
            (0..num_parties)
                .map(|_| Arc::new(RwLock::new((0, 0))))
                .collect(),
        )
    }

    async fn increment(&self, from: PartyID, proto_bytes: u64, net_bytes: u64) {
        let mut val = self.0[from as usize].write().await;
        val.0 += proto_bytes;
        val.1 += net_bytes;
    }

    pub async fn party(&self, from: PartyID) -> (u64, u64) {
        *self.0[from as usize].read().await
    }

    pub async fn total(&self) -> (u64, u64) {
        let mut proto_bytes = 0;
        let mut net_bytes = 0;

        for v in &self.0 {
            let val = v.read().await;
            proto_bytes += val.0;
            net_bytes += val.1;
        }

        (proto_bytes, net_bytes)
    }
}

pub type NetworkChannelBuilder = ProtoChannelBuilder;
pub type NetworkChannel = ProtoChannel<SendMessage, ReceivedMessage>;

pub async fn setup_tcp_network(
    party_id: PartyID,
    addresses: &[String],
) -> (Stats, NetworkChannelBuilder) {
    let num_parties: PartyID = addresses
        .len()
        .try_into()
        .expect("Number of parties to be at most 16 bytes long.");
    let (read_streams, write_streams): (Vec<_>, Vec<_>) = connect_to_peers(party_id, &addresses)
        .await
        .into_iter()
        .map(|stream| match stream {
            Some(stream) => {
                let (read_half, write_half) = stream.into_split();
                (Some(read_half), Some(write_half))
            }
            None => (None, None),
        })
        .unzip();

    // For a message being sent to the i-th party, the flow is as follows:
    //   msg -> net_inp (using channel)
    //   net_inp -> tcp_stream[i] (router task)
    //   tcp_stream[sender] -> proto_channel (per party receiver task)
    //   proto_channel -> mssg (using channel).
    let (net_inp_s, net_inp_r) = unbounded_channel();
    let (net_register_s, mut net_register_r) = unbounded_channel();
    let (proto_router_s, mut proto_router_r) = unbounded_channel();
    let channel_builder = ProtoChannelBuilder::new(net_inp_s, net_register_s);
    let stats = Stats::new(num_parties as usize);

    // net_inp -> tcp_stream[i].
    spawn(send_router(
        party_id,
        net_inp_r,
        write_streams,
        proto_router_s.clone(),
    ));

    // Start receiver task for each party.
    for (pid, stream) in read_streams.into_iter().enumerate() {
        if let Some(stream) = stream {
            // tcp_stream[pid] -> proto_channel.
            spawn(party_receive_router(
                pid.try_into().unwrap(),
                stream,
                proto_router_s.clone(),
                stats.clone(),
            ));
        } else if pid != (party_id as usize) {
            panic!("All parties did not connect.");
        }
    }

    spawn(async move {
        let mut buffer = HashMap::new();
        let mut registered: HashMap<Vec<u8>, UnboundedSender<ReceivedMessage>> = HashMap::new();

        loop {
            tokio::select! {
                Some(mssg) = proto_router_r.recv() => {
                    match registered.get(&mssg.proto_id) {
                        Some(sender) => {
                            if sender.send(mssg.clone()).is_err() {
                                registered.remove(&mssg.proto_id);
                                buffer
                                    .entry(mssg.proto_id.clone())
                                    .or_insert(Vec::new())
                                    .push(mssg);
                            }
                        },
                        None => {
                            buffer.entry(mssg.proto_id.clone()).or_insert(Vec::new()).push(mssg);
                        }
                    }
                },
                Some(RegisterProtocol(id, sender)) = net_register_r.recv() => {
                    if let Some(mssgs) = buffer.remove(&id) {
                        for mssg in mssgs {
                            sender.send(mssg).unwrap();
                        }
                    }
                    registered.insert(id, sender);
                },
                else => { break }
            }
        }
    });

    (stats, channel_builder)
}

pub async fn setup_local_network(num_parties: usize) -> Vec<(Stats, NetworkChannelBuilder)> {
    let (net_inp_s, mut net_inp_r) = unbounded_channel();
    let (register_s, mut register_r) = unbounded_channel();

    let stats: Vec<_> = (0..num_parties).map(|_| Stats::new(num_parties)).collect();

    let mut net_builders = Vec::with_capacity(num_parties);
    for i in 0..num_parties {
        let (party_s, mut party_r) = unbounded_channel::<SendMessage>();
        let (reg_party_s, mut reg_party_r) = unbounded_channel();
        let net_inp_s = net_inp_s.clone();
        let register_s = register_s.clone();

        spawn(async move {
            while let Some(mssg) = party_r.recv().await {
                net_inp_s.send((i, mssg)).unwrap();
            }
        });

        spawn(async move {
            while let Some(RegisterProtocol(id, sender)) = reg_party_r.recv().await {
                register_s.send((i, id, sender)).unwrap();
            }
        });

        net_builders.push(ProtoChannelBuilder::new(party_s, reg_party_s));
    }

    let res_stats = stats.clone();

    spawn(async move {
        let mut buffers: Vec<_> = (0..num_parties).map(|_| HashMap::new()).collect();
        let mut registered: Vec<HashMap<ProtocolID, UnboundedSender<ReceivedMessage>>> =
            (0..num_parties).map(|_| HashMap::new()).collect();

        loop {
            tokio::select! {
                Some((from, mssg)) = net_inp_r.recv() => {
                    let recv_mssg = ReceivedMessage {
                        proto_id: mssg.proto_id,
                        from: from.try_into().unwrap(),
                        data: mssg.data
                    };

                    match mssg.to {
                        Recipient::One(to) => {
                            if to != recv_mssg.from {
                                stats[to as usize]
                                    .increment(
                                        from.try_into().unwrap(),
                                        recv_mssg.data.len() as u64,
                                        (recv_mssg.data.len() + recv_mssg.proto_id.len()) as u64
                                    )
                                    .await;
                            }

                            match registered[to as usize].get(&recv_mssg.proto_id) {
                                Some(sender) => {
                                    if sender.send(recv_mssg.clone()).is_err() {
                                        registered[to as usize].remove(&recv_mssg.proto_id);
                                        buffers[to as usize]
                                            .entry(recv_mssg.proto_id.clone())
                                            .or_insert(Vec::new())
                                            .push(recv_mssg);
                                    }
                                },
                                None => {
                                    buffers[to as usize]
                                        .entry(recv_mssg.proto_id.clone())
                                        .or_insert(Vec::new())
                                        .push(recv_mssg);
                                }
                            }
                        },
                        Recipient::All => {
                            for to in 0..num_parties {
                                let recv_mssg = recv_mssg.clone();

                                if to != from {
                                    stats[to as usize]
                                        .increment(
                                            from.try_into().unwrap(),
                                            recv_mssg.data.len() as u64,
                                            (recv_mssg.data.len() + recv_mssg.proto_id.len()) as u64
                                        )
                                        .await;
                                }

                                match registered[to as usize].get(&recv_mssg.proto_id) {
                                    Some(sender) => {
                                        if sender.send(recv_mssg.clone()).is_err() {
                                            registered[to as usize].remove(&recv_mssg.proto_id);
                                            buffers[to as usize]
                                                .entry(recv_mssg.proto_id.clone())
                                                .or_insert(Vec::new())
                                                .push(recv_mssg);
                                        }
                                    },
                                    None => {
                                        buffers[to as usize]
                                            .entry(recv_mssg.proto_id.clone())
                                            .or_insert(Vec::new())
                                            .push(recv_mssg);
                                    }
                                }
                            }
                        }
                    }
                },
                Some((i, id, sender)) = register_r.recv() => {
                    if let Some(mssgs) = buffers[i].remove(&id) {
                        for mssg in mssgs {
                            sender.send(mssg).unwrap();
                        }
                    }
                    registered[i].insert(id, sender);
                },
                else => { break }
            }
        }
    });

    res_stats
        .into_iter()
        .zip(net_builders.into_iter())
        .collect()
}

pub async fn sync(proto_id: ProtocolID, chan: &mut NetworkChannel, num_parties: usize) {
    let data = b"sync".to_vec();

    chan.send(SendMessage {
        to: Recipient::All,
        proto_id,
        data,
    });

    message_from_each_party(chan, num_parties).await;
}

/// Waits to receive a message from each party and then returns the list of messages in order of
/// the party ID.
/// If multiple messages are received from the same party, the first one is returned and the latter
/// ones are dropped.
pub async fn message_from_each_party(
    chan: &mut NetworkChannel,
    num_parties: usize,
) -> Vec<Vec<u8>> {
    let mut mssgs = vec![Vec::new(); num_parties];
    let mut has_sent = vec![false; num_parties];
    let mut counter = 0;

    while counter != num_parties {
        let mssg = chan.recv().await;
        if !has_sent[mssg.from as usize] {
            mssgs[mssg.from as usize] = mssg.data;
            counter += 1;
            has_sent[mssg.from as usize] = true;
        }
    }

    mssgs
}

// Establish TCP connection with every peer.
async fn connect_to_peers(party_id: PartyID, addresses: &[String]) -> Vec<Option<TcpStream>> {
    const NUM_RETRIES: usize = 1000;
    const SEC_BETWEEN_RETRIES: u64 = 2;

    let num_parties: PartyID = addresses
        .len()
        .try_into()
        .expect("Number of parties to be at most 16 bytes long.");
    let my_address = addresses[usize::from(party_id)].clone();

    let accept_incoming = spawn(async move {
        let (_, port) = my_address.split_once(':').unwrap();
        let mut listen_addr = String::from("0.0.0.0:");
        listen_addr.push_str(port);
        let listener = TcpListener::bind(listen_addr).await.unwrap();

        let mut streams: Vec<_> = (0..num_parties).map(|_| None).collect();

        let expected_connections = num_parties - party_id - 1;
        let mut incoming = futures_lite::stream::iter(0..expected_connections)
            .then(|_| async {
                let (stream, _) = listener.accept().await.unwrap();
                stream
            })
            .boxed();

        while let Some(mut stream) = incoming.next().await {
            stream.set_nodelay(true).unwrap();

            let mut id_bytes = [0; 9];
            stream
                .read_exact(&mut id_bytes[..1])
                .await
                .expect("TCP stream is readable.");

            let connector_id = {
                let enc_len = vint64::decoded_len(id_bytes[0]);
                if enc_len != 1 {
                    stream
                        .read_exact(&mut id_bytes[1..enc_len])
                        .await
                        .expect("TCP stream is readable.");
                }
                vint64::decode(&mut &id_bytes[..enc_len]).unwrap() as PartyID
            };
            streams[usize::from(connector_id)] = Some(stream);
        }

        streams
    });

    let mut connect_to_peers_futures = Vec::with_capacity(usize::from(party_id));
    for i in 0..usize::from(party_id) {
        let peer_address = addresses[i].clone();
        connect_to_peers_futures.push(spawn(async move {
            let mut stream = {
                let mut num_tries = 0;
                loop {
                    if let Ok(stream) = TcpStream::connect(&peer_address).await {
                        break stream;
                    }

                    num_tries += 1;
                    if num_tries == NUM_RETRIES {
                        break TcpStream::connect(&peer_address)
                            .await
                            .expect("Peer is listening.");
                    }

                    tokio::time::sleep(Duration::from_secs(SEC_BETWEEN_RETRIES)).await;
                }
            };

            stream.set_nodelay(true).unwrap();
            stream
                .write_all(vint64::encode(party_id as u64).as_ref())
                .await
                .expect("TCP stream is writeable.");
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

// Send messages to each party over the appropriate TCP socket.
async fn send_router(
    party_id: PartyID,
    mut incoming: UnboundedReceiver<SendMessage>,
    mut tcp_streams: Vec<Option<OwnedWriteHalf>>,
    proto_router_s: UnboundedSender<ReceivedMessage>,
) {
    let num_parties = tcp_streams.len();

    while let Some(mssg) = incoming.recv().await {
        match mssg.to {
            Recipient::One(rid) => {
                if rid == party_id {
                    proto_router_s
                        .send(ReceivedMessage {
                            from: party_id,
                            proto_id: mssg.proto_id,
                            data: mssg.data,
                        })
                        .expect("Protocol channel sender is open.");
                } else {
                    send_to_party(
                        &mssg.proto_id,
                        &mssg.data,
                        tcp_streams[usize::from(rid)].as_mut().unwrap(),
                    )
                    .await;
                }
            }
            Recipient::All => {
                // TODO: Spawn tasks for each party if it provides more efficiency.
                let proto_id = mssg.proto_id;
                let data = mssg.data;

                for rid in 0..num_parties {
                    let proto_id = proto_id.clone();
                    let data = data.clone();

                    if rid as PartyID == party_id {
                        proto_router_s
                            .send(ReceivedMessage {
                                from: party_id,
                                proto_id: proto_id.clone(),
                                data: data.clone(),
                            })
                            .expect("Protocol channel sender is open.");
                    } else {
                        send_to_party(
                            &proto_id,
                            &data,
                            tcp_streams[usize::from(rid)].as_mut().unwrap(),
                        )
                        .await;
                    }
                }
            }
        }
    }
}

// Send data over stream for protocol with ID proto_id.
async fn send_to_party(proto_id: &ProtocolID, data: &[u8], stream: &mut OwnedWriteHalf) {
    let mssg_len = proto_id.len() + data.len() + 1;
    let id_len: u8 = proto_id
        .len()
        .try_into()
        .expect("Protocol ID to be at most 255 bytes.");

    let mut mssg = Vec::with_capacity(usize::from(mssg_len) + vint64::encoded_len(mssg_len as u64));
    mssg.extend_from_slice(vint64::encode(mssg_len as u64).as_ref());
    mssg.push(id_len);
    mssg.extend_from_slice(&proto_id);
    mssg.extend_from_slice(&data);

    stream
        .write_all(&mssg)
        .await
        .expect("TCP stream is writable.");
    stream.flush().await.expect("TCP stream is flushable.");
}

// Receive messages sent over the TCP socket and send them to the appropriate protocol.
async fn party_receive_router(
    pid: PartyID,
    mut incoming: OwnedReadHalf,
    proto_router_s: UnboundedSender<ReceivedMessage>,
    stats: Stats,
) {
    let mut bytes = [0; 9];

    while let Ok(_) = incoming.read_exact(&mut bytes[..1]).await {
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

        if let Err(_) = proto_router_s.send(recv_mssg) {
            break;
        }

        stats
            .increment(pid, (mssg_len - id_len - 1) as u64, (mssg_len + 1) as u64)
            .await;
    }
}
