use super::{ProtoChannelBuilder, ProtocolID};
use crate::utils::spawn;
use crate::PartyID;
use smol;
use smol::io::{AsyncReadExt, AsyncWriteExt};
use smol::net::{TcpListener, TcpStream};
use smol::stream::StreamExt;
use std::time::Duration;

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

pub async fn setup_tcp_network(
    party_id: PartyID,
    addresses: &[String],
) -> ProtoChannelBuilder<SendMessage, ReceivedMessage> {
    let num_parties: PartyID = addresses
        .len()
        .try_into()
        .expect("Number of parties to be at most 16 bytes long.");
    let mut tcp_streams = connect_to_peers(party_id, &addresses).await;

    // For a message being sent to the i-th party, the flow is as follows:
    //   msg -> net_inp (using network struct)
    //   net_inp -> party_router[i] (router task)
    //   party_router[i] -> tcp_stream[i] (per party sender task)
    //   tcp_stream[sender] -> proto_channel (per party receiver task)
    //   proto_channel -> mssg (using network struct).
    let (net_inp_s, mut net_inp_r) = smol::channel::unbounded::<SendMessage>();
    let mut party_router_s = Vec::with_capacity(addresses.len());
    let net_channel_builder = ProtoChannelBuilder::new(net_inp_s);

    // Start sender and listener tasks for each party.
    for pid in 0..num_parties {
        if pid == party_id {
            party_router_s.push(None);
        } else if let Some(mut stream) = tcp_streams[usize::from(pid)].take() {
            let mut send_stream = stream.clone();

            let (router_s, mut router_r) = smol::channel::unbounded::<SendMessage>();
            party_router_s.push(Some(router_s));

            // party_router[pid] -> tcp_stream[pid].
            spawn(async move {
                while let Some(mssg) = router_r.next().await {
                    let data_len: u8 = (mssg.proto_id.len() + mssg.data.len() + 1)
                        .try_into()
                        .expect("Message length to be at most 255 bytes.");
                    let id_len: u8 = mssg
                        .proto_id
                        .len()
                        .try_into()
                        .expect("Protocol ID to be at most 255 bytes.");

                    let mut data = Vec::with_capacity(usize::from(data_len) + 1);
                    data.push(data_len);
                    data.push(id_len);
                    data.extend_from_slice(&mssg.proto_id);
                    data.extend_from_slice(&mssg.data);

                    send_stream
                        .write_all(&data)
                        .await
                        .expect("TCP stream is writable.");
                    send_stream.flush().await.expect("TCP stream is flushable.");
                }
            })
            .detach();

            let channel_builder = net_channel_builder.clone();

            // tcp_stream[i] -> proto_channel.
            spawn(async move {
                let mut byte = [0; 1];

                while let Ok(_) = stream.read_exact(&mut byte).await {
                    let data_len = u8::from_be_bytes(byte);
                    let mut payload = vec![0; usize::from(data_len)];
                    stream
                        .read_exact(&mut payload)
                        .await
                        .expect("TCP stream is readable.");

                    let id_len = usize::from(payload[0]);
                    let proto_id = payload[1..(id_len + 1)].to_vec();
                    let data = payload[(id_len + 1)..].to_vec();
                    let mssg = ReceivedMessage {
                        from: pid,
                        proto_id,
                        data,
                    };

                    let sender = channel_builder.receiver_handle(&mssg.proto_id).await;

                    if let Err(_) = sender.send(mssg).await {
                        break;
                    }
                }
            })
            .detach();
        } else {
            panic!("All parties did not connect.");
        }
    }

    // net_inp -> party_router[i].
    let channel_builder = net_channel_builder.clone();
    spawn(async move {
        while let Some(mssg) = net_inp_r.next().await {
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
                        party_router_s[usize::from(rid)]
                            .as_ref()
                            .unwrap()
                            .send(mssg)
                            .await
                            .expect("Party router sender is open.");
                    }
                }
                Recipient::All => {
                    for i in 0..num_parties {
                        if i == party_id {
                            let sender = channel_builder.receiver_handle(&mssg.proto_id).await;
                            sender
                                .send(ReceivedMessage {
                                    from: party_id,
                                    proto_id: mssg.proto_id.clone(),
                                    data: mssg.data.clone(),
                                })
                                .await
                                .expect("Protocol channel sender is open.");
                        } else {
                            party_router_s[usize::from(i)]
                                .as_ref()
                                .unwrap()
                                .send(mssg.clone())
                                .await
                                .expect("Party router sender is open.");
                        }
                    }
                }
            }
        }

        for s in party_router_s {
            if let Some(stream) = s {
                stream.close();
            }
        }
    })
    .detach();

    net_channel_builder
}

pub async fn setup_local_network(
    num: usize,
) -> Vec<ProtoChannelBuilder<SendMessage, ReceivedMessage>> {
    let mut net_builders = Vec::with_capacity(num);
    let mut router_r = Vec::with_capacity(num);

    for _ in 0..num {
        let (s, r) = smol::channel::unbounded::<SendMessage>();
        router_r.push(r);
        net_builders.push(ProtoChannelBuilder::new(s));
    }

    for pid in 0..num {
        let net_builders = net_builders.clone();
        let mut party_router_r = router_r[pid].clone();

        spawn(async move {
            while let Some(mssg) = party_router_r.next().await {
                match mssg.to {
                    Recipient::One(rid) => {
                        let sender = net_builders[usize::from(rid)]
                            .receiver_handle(&mssg.proto_id)
                            .await;
                        sender
                            .send(ReceivedMessage {
                                from: pid.try_into().unwrap(),
                                proto_id: mssg.proto_id,
                                data: mssg.data,
                            })
                            .await
                            .expect("Protocol channel sender is open.");
                    }
                    Recipient::All => {
                        for rid in 0..num {
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

    net_builders
}
