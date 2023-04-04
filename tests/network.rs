use scalable_mpc::protocol::network;
use scalable_mpc::{block_on, spawn, PartyID};
use serial_test::serial;

#[test]
fn net_local_two_party() {
    block_on(async move {
        let data = b"hello world".to_vec();
        let proto_id = b"root protocol".to_vec();

        let mut comms: Vec<_> = network::setup_local_network(2)
            .await
            .into_iter()
            .map(|val| Some(val))
            .collect();
        let comms1 = comms[0].take().unwrap();
        let comms2 = comms[1].take().unwrap();

        let party1 = {
            let proto_id = proto_id.clone();
            let data = data.clone();

            spawn(async move {
                let (stats, net) = comms1;
                let chan = net.channel(&proto_id).await;
                chan.send(network::SendMessage {
                    to: network::Recipient::One(1),
                    proto_id: proto_id.clone(),
                    data: data.clone(),
                })
                .await;

                let mssg = chan.recv().await;
                assert_eq!(mssg.from, 1);
                assert_eq!(mssg.proto_id, proto_id);
                assert_eq!(mssg.data, data);

                let data_len = data.len() as u64;

                assert_eq!(stats.party(1).await, (data_len, data_len));
                assert_eq!(stats.party(0).await, (0, 0));
            })
        };

        let party2 = spawn(async move {
            let (stats, net) = comms2;
            let chan = net.channel(&proto_id).await;
            let mssg = chan.recv().await;
            assert_eq!(mssg.from, 0);
            assert_eq!(mssg.proto_id, proto_id);
            assert_eq!(mssg.data, data);

            let data_len = data.len() as u64;

            assert_eq!(stats.party(0).await, (data_len, data_len));
            assert_eq!(stats.party(1).await, (0, 0));

            chan.send(network::SendMessage {
                to: network::Recipient::One(0),
                proto_id,
                data,
            })
            .await;

            assert_eq!(stats.party(0).await, (data_len, data_len));
            assert_eq!(stats.party(1).await, (0, 0));
        });

        party1.await;
        party2.await;
    });
}

#[test]
fn net_local_many_parties() {
    block_on(async move {
        let num = 10;
        let data = b"hello world".to_vec();
        let proto_id = b"protocol id".to_vec();

        let comms = network::setup_local_network(num).await;
        let mut handles = Vec::new();
        for (pid, (stats, net)) in comms.into_iter().enumerate() {
            let proto_id = proto_id.clone();
            let data = data.clone();
            let pid: PartyID = pid.try_into().unwrap();

            handles.push(spawn(async move {
                let chan = net.channel(&proto_id).await;
                chan.send(network::SendMessage {
                    to: network::Recipient::All,
                    proto_id: proto_id.clone(),
                    data: data.clone(),
                })
                .await;

                let mut recv_ids = vec![false; num];

                for _ in 0..num {
                    let mssg = chan.recv().await;
                    recv_ids[mssg.from as usize] = true;
                    assert_eq!(mssg.proto_id, proto_id);
                    assert_eq!(mssg.data, data);
                }

                for b in recv_ids {
                    assert!(b);
                }

                let data_len = data.len() as u64;

                for i in 0..num.try_into().unwrap() {
                    if i == pid {
                        assert_eq!(stats.party(i).await, (0, 0));
                    } else {
                        assert_eq!(stats.party(i).await, (data_len, data_len));
                    }
                }
            }));
        }

        for handle in handles {
            handle.await;
        }
    });
}

#[test]
fn net_local_many_parties_two_ids() {
    block_on(async move {
        let num = 10;
        let data1 = b"hello".to_vec();
        let data2 = b"world".to_vec();
        let proto_id1 = b"proto_id 1".to_vec();
        let proto_id2 = b"proto_id 2".to_vec();

        let comms = network::setup_local_network(num).await;
        let mut handles = Vec::new();
        for (pid, (stats, net)) in comms.into_iter().enumerate() {
            let proto_id1 = proto_id1.clone();
            let proto_id2 = proto_id2.clone();
            let data1 = data1.clone();
            let data2 = data2.clone();
            let pid: PartyID = pid.try_into().unwrap();

            handles.push(spawn(async move {
                let chan1 = net.channel(&proto_id1).await;
                chan1
                    .send(network::SendMessage {
                        to: network::Recipient::All,
                        proto_id: proto_id1.clone(),
                        data: data1.clone(),
                    })
                    .await;

                let chan2 = net.channel(&proto_id2).await;
                chan2
                    .send(network::SendMessage {
                        to: network::Recipient::All,
                        proto_id: proto_id2.clone(),
                        data: data2.clone(),
                    })
                    .await;

                for _ in 0..num {
                    let mssg2 = chan2.recv().await;
                    let mssg1 = chan1.recv().await;
                    assert_eq!(mssg1.proto_id, proto_id1);
                    assert_eq!(mssg1.data, data1);
                    assert_eq!(mssg2.proto_id, proto_id2);
                    assert_eq!(mssg2.data, data2);
                }

                let data_len = (data1.len() + data2.len()) as u64;

                for i in 0..num.try_into().unwrap() {
                    if i == pid {
                        assert_eq!(stats.party(i).await, (0, 0));
                    } else {
                        assert_eq!(stats.party(i).await, (data_len, data_len));
                    }
                }
            }));
        }

        for handle in handles {
            handle.await;
        }
    });
}

#[test]
#[serial]
fn net_tcp_two_party() {
    block_on(async move {
        let data = b"hello world".to_vec();
        let proto_id = b"root protocol".to_vec();
        let addresses = vec![
            String::from("127.0.0.1:8001"),
            String::from("127.0.0.1:8002"),
        ];

        let party1 = {
            let proto_id = proto_id.clone();
            let data = data.clone();
            let addresses = addresses.clone();

            spawn(async move {
                let (stats, net) = network::setup_tcp_network(0, &addresses).await;
                let chan = net.channel(&proto_id).await;
                chan.send(network::SendMessage {
                    to: network::Recipient::One(1),
                    proto_id: proto_id.clone(),
                    data: data.clone(),
                })
                .await;

                let mssg = chan.recv().await;
                assert_eq!(mssg.from, 1);
                assert_eq!(mssg.proto_id, proto_id);
                assert_eq!(mssg.data, data);

                let data_len = data.len() as u64;

                assert_eq!(stats.party(1).await.0, data_len);
                assert_eq!(stats.party(0).await, (0, 0));
            })
        };

        let party2 = spawn(async move {
            let (stats, net) = network::setup_tcp_network(1, &addresses).await;
            let chan = net.channel(&proto_id).await;
            let mssg = chan.recv().await;
            assert_eq!(mssg.from, 0);
            assert_eq!(mssg.proto_id, proto_id);
            assert_eq!(mssg.data, data);

            let data_len = data.len() as u64;

            assert_eq!(stats.party(0).await.0, data_len);
            assert_eq!(stats.party(1).await, (0, 0));

            chan.send(network::SendMessage {
                to: network::Recipient::One(0),
                proto_id,
                data,
            })
            .await;

            assert_eq!(stats.party(0).await.0, data_len);
            assert_eq!(stats.party(1).await, (0, 0));
        });

        party1.await;
        party2.await;
    });
}

#[test]
#[serial]
fn net_tcp_many_parties() {
    block_on(async move {
        let num = 10;
        let data = b"hello world".to_vec();
        let proto_id = b"protocol id".to_vec();
        let addresses: Vec<_> = (0..num)
            .map(|i| format!("127.0.0.1:{}", 8000 + i))
            .collect();

        let mut handles = Vec::new();
        for pid in 0..num {
            let proto_id = proto_id.clone();
            let data = data.clone();
            let addresses = addresses.clone();

            handles.push(spawn(async move {
                let (stats, net) = network::setup_tcp_network(pid, &addresses).await;
                let chan = net.channel(&proto_id).await;
                chan.send(network::SendMessage {
                    to: network::Recipient::All,
                    proto_id: proto_id.clone(),
                    data: data.clone(),
                })
                .await;

                let mut recv_ids = vec![false; num as usize];

                for _ in 0..num {
                    let mssg = chan.recv().await;
                    recv_ids[mssg.from as usize] = true;
                    assert_eq!(mssg.proto_id, proto_id);
                    assert_eq!(mssg.data, data);
                }

                for b in recv_ids {
                    assert!(b);
                }

                let data_len = data.len() as u64;

                for i in 0..num.try_into().unwrap() {
                    if i == pid {
                        assert_eq!(stats.party(i).await, (0, 0));
                    } else {
                        assert_eq!(stats.party(i).await.0, data_len);
                    }
                }
            }));
        }

        for handle in handles {
            handle.await;
        }
    });
}

#[test]
#[serial]
fn net_tcp_many_parties_two_ids() {
    block_on(async move {
        let num = 10;
        let data1 = b"hello".to_vec();
        let data2 = b"world".to_vec();
        let proto_id1 = b"proto_id 1".to_vec();
        let proto_id2 = b"proto_id 2".to_vec();
        let addresses: Vec<_> = (0..num)
            .map(|i| format!("127.0.0.1:{}", 8000 + i))
            .collect();

        let mut handles = Vec::new();
        for pid in 0..num {
            let proto_id1 = proto_id1.clone();
            let proto_id2 = proto_id2.clone();
            let data1 = data1.clone();
            let data2 = data2.clone();
            let addresses = addresses.clone();

            handles.push(spawn(async move {
                let (stats, net) = network::setup_tcp_network(pid, &addresses).await;
                let chan1 = net.channel(&proto_id1).await;
                chan1
                    .send(network::SendMessage {
                        to: network::Recipient::All,
                        proto_id: proto_id1.clone(),
                        data: data1.clone(),
                    })
                    .await;

                let chan2 = net.channel(&proto_id2).await;
                chan2
                    .send(network::SendMessage {
                        to: network::Recipient::All,
                        proto_id: proto_id2.clone(),
                        data: data2.clone(),
                    })
                    .await;

                for _ in 0..num {
                    let mssg2 = chan2.recv().await;
                    let mssg1 = chan1.recv().await;
                    assert_eq!(mssg1.proto_id, proto_id1);
                    assert_eq!(mssg1.data, data1);
                    assert_eq!(mssg2.proto_id, proto_id2);
                    assert_eq!(mssg2.data, data2);
                }

                let data_len = (data1.len() + data2.len()) as u64;

                for i in 0..num.try_into().unwrap() {
                    if i == pid {
                        assert_eq!(stats.party(i).await.0, 0);
                    } else {
                        assert_eq!(stats.party(i).await.0, data_len);
                    }
                }
            }));
        }

        for handle in handles {
            handle.await;
        }
    });
}
