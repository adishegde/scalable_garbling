use scalable_mpc::protocol::network;
use scalable_mpc::{block_on, spawn};
use serial_test::serial;

#[test]
fn net_local_two_party() {
    block_on(async move {
        let data = b"hello world".to_vec();
        let proto_id = b"root protocol".to_vec();

        let mut nets: Vec<_> = network::setup_local_network(2)
            .await
            .into_iter()
            .map(|net| Some(net))
            .collect();
        let net1 = nets[0].take().unwrap();
        let net2 = nets[1].take().unwrap();

        let party1 = {
            let proto_id = proto_id.clone();
            let data = data.clone();

            spawn(async move {
                let chan = net1.channel(&proto_id).await;
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

                chan.close();
            })
        };

        let party2 = spawn(async move {
            let chan = net2.channel(&proto_id).await;
            let mssg = chan.recv().await;
            assert_eq!(mssg.from, 0);
            assert_eq!(mssg.proto_id, proto_id);
            assert_eq!(mssg.data, data);

            chan.send(network::SendMessage {
                to: network::Recipient::One(0),
                proto_id,
                data,
            })
            .await;

            chan.close();
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

        let nets = network::setup_local_network(num).await;
        let mut handles = Vec::new();
        for (pid, net) in nets.into_iter().enumerate() {
            let proto_id = proto_id.clone();
            let data = data.clone();

            handles.push(spawn(async move {
                let chan = net.channel(&proto_id).await;

                for i in 0..num {
                    if i == pid {
                        chan.send(network::SendMessage {
                            to: network::Recipient::All,
                            proto_id: proto_id.clone(),
                            data: data.clone(),
                        })
                        .await;
                    }

                    let mssg = chan.recv().await;
                    assert_eq!(mssg.from, i.try_into().unwrap());
                    assert_eq!(mssg.proto_id, proto_id);
                    assert_eq!(mssg.data, data);
                }

                chan.close();
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
                let net = network::setup_tcp_network(0, &addresses).await;
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

                chan.close();
            })
        };

        let party2 = spawn(async move {
            let net = network::setup_tcp_network(1, &addresses).await;
            let chan = net.channel(&proto_id).await;
            let mssg = chan.recv().await;
            assert_eq!(mssg.from, 0);
            assert_eq!(mssg.proto_id, proto_id);
            assert_eq!(mssg.data, data);

            chan.send(network::SendMessage {
                to: network::Recipient::One(0),
                proto_id,
                data,
            })
            .await;

            chan.close();
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
                let net = network::setup_tcp_network(pid, &addresses).await;
                let chan = net.channel(&proto_id).await;

                for i in 0..num {
                    if i == pid {
                        chan.send(network::SendMessage {
                            to: network::Recipient::All,
                            proto_id: proto_id.clone(),
                            data: data.clone(),
                        })
                        .await;
                    }

                    let mssg = chan.recv().await;
                    assert_eq!(mssg.from, i.try_into().unwrap());
                    assert_eq!(mssg.proto_id, proto_id);
                    assert_eq!(mssg.data, data);
                }

                chan.close();
            }));
        }

        for handle in handles {
            handle.await;
        }
    });
}
