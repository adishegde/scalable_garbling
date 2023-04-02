use crate::utils::spawn;
use crate::PartyID;
use smol::channel::{unbounded, Receiver, Sender};
use smol::io::{AsyncReadExt, AsyncWriteExt};
use smol::net::{TcpListener, TcpStream};
use smol::stream::StreamExt;

struct Message {
    sender: PartyID,
    receiver: PartyID,
    payload: Vec<u8>,
}

pub struct Network {
    pid: PartyID,
    num: usize,
    sender: Sender<Message>,
    receiver: Receiver<Message>,
}

impl Network {
    pub async fn new_tcp(party_id: PartyID, addresses: &[String]) -> Network {
        let num_parties: PartyID = addresses.len().try_into().unwrap();
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
                stream.read_exact(&mut id_bytes).await.unwrap();

                let connector_id = PartyID::from_be_bytes(id_bytes);
                streams[usize::from(connector_id)] = Some(stream);
            }

            streams
        });

        let mut connect_to_peers_futures = Vec::with_capacity(usize::from(party_id));
        for i in 0..usize::from(party_id) {
            let peer_address = addresses[i].clone();
            connect_to_peers_futures.push(spawn(async move {
                let mut stream = TcpStream::connect(&peer_address).await.unwrap();
                stream.set_nodelay(true).unwrap();
                stream.write_all(&party_id.to_be_bytes()).await.unwrap();
                (i, stream)
            }));
        }

        let mut tcp_streams = spawn(async move {
            let mut streams = accept_incoming.await;
            for future in connect_to_peers_futures {
                let (pid, stream) = future.await;
                streams[pid] = Some(stream);
            }
            streams
        })
        .await;

        let num_parties: PartyID = addresses.len().try_into().unwrap();

        // (net_s, net_r) form the network.
        // For a message being sent to the i-th party, the flow is as follows:
        //   msg -> net_inp (using network struct)
        //   net_inp -> party_router[i] (router task)
        //   party_router[i] -> tcp_stream[i] (per party sender task)
        //   tcp_stream[sender] -> net_out (per party receiver task)
        //   net_out -> mssg (using network struct).
        let (net_inp_s, mut net_inp_r) = unbounded::<Message>();
        let (net_out_s, net_out_r) = unbounded::<Message>();
        let mut party_router_s = Vec::with_capacity(addresses.len());

        // Start sender and listener tasks for each party.
        for pid in 0..num_parties {
            if pid == party_id {
                party_router_s.push(None);
            } else if let Some(mut stream) = tcp_streams[usize::from(pid)].take() {
                let mut send_stream = stream.clone();

                let (router_s, mut router_r) = unbounded::<Message>();
                party_router_s.push(Some(router_s));

                // party_router[pid] -> tcp_stream[pid].
                spawn(async move {
                    while let Some(mssg) = router_r.next().await {
                        // We're assuming each message consists of at most 255 bytes. It might be a
                        // better idea to encode the length as a varint to support longer messages.
                        let num_bytes: u8 = mssg.payload.len().try_into().unwrap();
                        send_stream
                            .write_all(&num_bytes.to_be_bytes())
                            .await
                            .unwrap();
                        send_stream.write_all(&mssg.payload).await.unwrap();
                        send_stream.flush().await.unwrap();
                    }

                    send_stream.close().await.unwrap();
                })
                .detach();

                let cloned_net_out_s = net_out_s.clone();

                // tcp_stream[i] -> net_out.
                spawn(async move {
                    let mut data_len = [0; 1];
                    while let Ok(_) = stream.read_exact(&mut data_len).await {
                        let mut payload = vec![0; usize::from(data_len[0])];
                        stream.read_exact(&mut payload).await.unwrap();
                        if let Err(_) = cloned_net_out_s
                            .send(Message {
                                sender: pid,
                                receiver: party_id,
                                payload,
                            })
                            .await
                        {
                            break;
                        }
                    }

                    cloned_net_out_s.close();
                })
                .detach();
            } else {
                panic!("All parties did not connect.");
            }
        }

        // net_inp -> party_router[i].
        let my_router_s = net_out_s.clone();
        spawn(async move {
            while let Some(mssg) = net_inp_r.next().await {
                if mssg.receiver == party_id {
                    my_router_s.send(mssg).await.unwrap();
                } else {
                    party_router_s[usize::from(mssg.receiver)]
                        .as_ref()
                        .unwrap()
                        .send(mssg)
                        .await
                        .unwrap();
                }
            }

            for s in party_router_s {
                if let Some(stream) = s {
                    stream.close();
                }
            }
        })
        .detach();

        Network {
            pid: party_id,
            num: addresses.len(),
            sender: net_inp_s,
            receiver: net_out_r,
        }
    }

    pub fn new_local(num: usize) -> Vec<Network> {
        let (net_inp_s, mut net_inp_r) = unbounded::<Message>();
        let mut router_s = Vec::with_capacity(num);
        let mut router_r = Vec::with_capacity(num);

        for _ in 0..num {
            let (s, r) = unbounded();
            router_s.push(s);
            router_r.push(r);
        }

        spawn(async move {
            while let Some(mssg) = net_inp_r.next().await {
                router_s[usize::from(mssg.receiver)]
                    .send(mssg)
                    .await
                    .unwrap();
            }

            for s in router_s {
                s.close();
            }
        })
        .detach();

        router_r
            .into_iter()
            .enumerate()
            .map(|(i, r)| Network {
                pid: i.try_into().unwrap(),
                num,
                sender: net_inp_s.clone(),
                receiver: r,
            })
            .collect()
    }

    pub async fn send(&self, receiver: PartyID, payload: Vec<u8>) {
        self.sender
            .send(Message {
                sender: self.pid,
                receiver,
                payload,
            })
            .await
            .unwrap();
    }

    pub async fn send_all(&self, payload: Vec<u8>) {
        for receiver in 0..(self.num).try_into().unwrap() {
            self.send(receiver, payload.clone()).await;
        }
    }

    pub async fn recv(&self) -> (PartyID, Vec<u8>) {
        let mssg = self.receiver.recv().await.unwrap();
        (mssg.sender, mssg.payload)
    }

    pub fn close(&self) {
        self.sender.close();
    }
}
