use super::{ProtoChannelBuilder, ProtocolID};
use crate::spawn;
use crate::PartyID;
use smol;
use smol::channel::{unbounded, Receiver};
use smol::io::{AsyncReadExt, AsyncWriteExt};
use smol::lock::RwLock;
use smol::net::{TcpListener, TcpStream};
use smol::stream::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use vint64;

#[derive(Clone, Copy)]
pub enum Recipient {
    One(PartyID),
    All,
}

#[derive(Clone)]
pub struct SendMessage {
    pub proto_id: ProtocolID,
    pub to: Recipient,
    pub data: Vec<u8>,
}

#[derive(Clone)]
pub struct ReceivedMessage {
    pub proto_id: ProtocolID,
    pub from: PartyID,
    pub data: Vec<u8>,
}

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

pub type NetworkChannelBuilder = ProtoChannelBuilder<SendMessage, ReceivedMessage>;

pub async fn setup_tcp_network(
    party_id: PartyID,
    addresses: &[String],
) -> (Stats, NetworkChannelBuilder) {
    let num_parties: PartyID = addresses
        .len()
        .try_into()
        .expect("Number of parties to be at most 16 bytes long.");
    let mut tcp_streams = connect_to_peers(party_id, &addresses).await;

    // For a message being sent to the i-th party, the flow is as follows:
    //   msg -> net_inp (using channel)
    //   net_inp -> tcp_stream[i] (router task)
    //   tcp_stream[sender] -> proto_channel (per party receiver task)
    //   proto_channel -> mssg (using channel).
    let (net_inp_s, net_inp_r) = unbounded::<SendMessage>();
    let channel_builder = ProtoChannelBuilder::new(net_inp_s);
    let stats = Stats::new(num_parties as usize);

    // net_inp -> tcp_stream[i].
    spawn(send_router(
        party_id,
        net_inp_r,
        tcp_streams.clone(),
        channel_builder.clone(),
    ))
    .detach();

    // Start receiver task for each party.
    for pid in 0..num_parties {
        if let Some(stream) = tcp_streams[usize::from(pid)].take() {
            // tcp_stream[pid] -> proto_channel.
            spawn(party_receive_router(
                pid,
                stream,
                channel_builder.clone(),
                stats.clone(),
            ))
            .detach();
        } else if pid != party_id {
            panic!("All parties did not connect.");
        }
    }

    (stats, channel_builder)
}

pub async fn setup_local_network(num_parties: usize) -> Vec<(Stats, NetworkChannelBuilder)> {
    let mut stats = Vec::with_capacity(num_parties);
    let mut net_builders = Vec::with_capacity(num_parties);
    let mut router_r = Vec::with_capacity(num_parties);

    for _ in 0..num_parties {
        let (s, r) = unbounded::<SendMessage>();
        router_r.push(r);
        net_builders.push(ProtoChannelBuilder::new(s));
        stats.push(Stats::new(num_parties));
    }

    for pid in 0..num_parties {
        let net_builders = net_builders.clone();
        let mut party_router_r = router_r[pid].clone();
        let stats = stats.clone();
        let pid: PartyID = pid.try_into().unwrap();

        spawn(async move {
            while let Some(mssg) = party_router_r.next().await {
                let data_len = mssg.data.len();

                match mssg.to {
                    Recipient::One(rid) => {
                        if rid != pid {
                            stats[rid as usize]
                                .increment(pid, data_len as u64, data_len as u64)
                                .await;
                        }

                        let sender = net_builders[rid as usize]
                            .receiver_handle(&mssg.proto_id)
                            .await;
                        sender
                            .send(ReceivedMessage {
                                from: pid,
                                proto_id: mssg.proto_id,
                                data: mssg.data,
                            })
                            .await
                            .expect("Protocol channel sender is open.");
                    }
                    Recipient::All => {
                        for rid in 0..num_parties {
                            if rid != pid as usize {
                                stats[rid]
                                    .increment(pid, data_len as u64, data_len as u64)
                                    .await;
                            }

                            let sender = net_builders[rid].receiver_handle(&mssg.proto_id).await;
                            sender
                                .send(ReceivedMessage {
                                    from: pid.try_into().unwrap(),
                                    proto_id: mssg.proto_id.clone(),
                                    data: mssg.data.clone(),
                                })
                                .await
                                .expect("Protocol channel sender is open.");
                        }
                    }
                }
            }
        })
        .detach();
    }

    stats.into_iter().zip(net_builders.into_iter()).collect()
}

/// Waits to receive a message from each party and then returns the list of messages in order of
/// the party ID.
/// If multiple messages are received from the same party, the first one is returned and the latter
/// ones are dropped.
pub async fn message_from_each_party(
    receiver: Receiver<ReceivedMessage>,
    num_parties: usize,
) -> Vec<Vec<u8>> {
    let mut mssgs = vec![Vec::new(); num_parties];
    let mut has_sent = vec![false; num_parties];
    let mut counter = 0;

    while counter != num_parties {
        let mssg = receiver.recv().await.unwrap();
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
    const NUM_RETRIES: usize = 10;
    const SEC_BETWEEN_RETRIES: u64 = 1;

    let num_parties: PartyID = addresses
        .len()
        .try_into()
        .expect("Number of parties to be at most 16 bytes long.");
    let my_address = addresses[usize::from(party_id)].clone();

    let accept_incoming = spawn(async move {
        let listener = TcpListener::bind(my_address).await.unwrap();

        let mut streams = vec![None; usize::from(num_parties)];

        let expected_connections = num_parties - party_id - 1;
        let mut incoming = listener.incoming().take(usize::from(expected_connections));

        while let Some(stream) = incoming.next().await {
            let mut stream = stream.unwrap();
            stream.set_nodelay(true).unwrap();

            let mut id_bytes = [0; std::mem::size_of::<PartyID>()];
            stream
                .read_exact(&mut id_bytes)
                .await
                .expect("TCP stream is readable.");

            let connector_id = PartyID::from_be_bytes(id_bytes);
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

                    smol::Timer::after(Duration::from_secs(SEC_BETWEEN_RETRIES)).await;
                }
            };

            stream.set_nodelay(true).unwrap();
            stream
                .write_all(&party_id.to_be_bytes())
                .await
                .expect("TCP stream is writeable.");
            stream.flush().await.expect("TCP stream is flushble.");
            (i, stream)
        }));
    }

    spawn(async move {
        let mut streams = accept_incoming.await;
        for future in connect_to_peers_futures {
            let (pid, stream) = future.await;
            streams[pid] = Some(stream);
        }
        streams
    })
    .await
}

// Send messages to each party over the appropriate TCP socket.
async fn send_router(
    party_id: PartyID,
    mut incoming: Receiver<SendMessage>,
    tcp_streams: Vec<Option<TcpStream>>,
    channel_builder: NetworkChannelBuilder,
) {
    while let Some(mssg) = incoming.next().await {
        match mssg.to {
            Recipient::One(rid) => {
                if rid == party_id {
                    let sender = channel_builder.receiver_handle(&mssg.proto_id).await;
                    sender
                        .send(ReceivedMessage {
                            from: party_id,
                            proto_id: mssg.proto_id,
                            data: mssg.data,
                        })
                        .await
                        .expect("Protocol channel sender is open.");
                } else {
                    send_to_party(
                        &mssg.proto_id,
                        &mssg.data,
                        tcp_streams[usize::from(rid)].as_ref().unwrap().clone(),
                    )
                    .await;
                }
            }
            Recipient::All => {
                let mut tasks = Vec::with_capacity(tcp_streams.len());
                let proto_id = Arc::new(mssg.proto_id);
                let data = Arc::new(mssg.data);

                for rid in 0..tcp_streams.len() {
                    let proto_id = proto_id.clone();
                    let data = data.clone();

                    if rid as PartyID == party_id {
                        let channel_builder = channel_builder.clone();
                        tasks.push(spawn(async move {
                            let sender = channel_builder.receiver_handle(&proto_id).await;
                            sender
                                .send(ReceivedMessage {
                                    from: party_id,
                                    proto_id: proto_id.to_vec(),
                                    data: data.to_vec(),
                                })
                                .await
                                .expect("Protocol channel sender is open.");
                        }));
                    } else {
                        let stream = tcp_streams[usize::from(rid)].as_ref().unwrap().clone();
                        tasks.push(spawn(async move {
                            send_to_party(proto_id.as_ref(), data.as_ref(), stream).await;
                        }));
                    }
                }

                // Wait for all sends to complete.
                spawn(async move {
                    for task in tasks {
                        task.await;
                    }
                })
                .await;
            }
        }
    }
}

// Send data over stream for protocol with ID proto_id.
async fn send_to_party(proto_id: &ProtocolID, data: &[u8], mut stream: TcpStream) {
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
    mut incoming: TcpStream,
    channel_builder: NetworkChannelBuilder,
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

        let sender = channel_builder.receiver_handle(&recv_mssg.proto_id).await;

        if let Err(_) = sender.send(recv_mssg).await {
            break;
        }

        stats
            .increment(pid, (mssg_len - id_len - 1) as u64, (mssg_len + 1) as u64)
            .await;
    }
}
