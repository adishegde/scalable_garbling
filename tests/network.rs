use async_global_executor as async_exec;
use scalable_mpc::protocol::network::Network;
use smol;

#[test]
fn net_local_two_party() {
    async_exec::block_on(async move {
        let mssg = b"hello world";

        let mut nets: Vec<_> = Network::new_local(2)
            .into_iter()
            .map(|net| Some(net))
            .collect();
        let net1 = nets[0].take().unwrap();
        let net2 = nets[1].take().unwrap();

        let party1 = async_exec::spawn(async move {
            net1.send(1, mssg.to_vec()).await;
            let (pid, data) = net1.recv().await;
            assert_eq!(pid, 1);
            assert_eq!(data, mssg);
            net1.close();
        });

        let party2 = async_exec::spawn(async move {
            let (pid, data) = net2.recv().await;
            assert_eq!(pid, 0);
            assert_eq!(data, mssg);
            net2.send(0, mssg.to_vec()).await;
            net2.close();
        });

        party1.await;
        party2.await;
    });
}

#[test]
fn net_tcp_two_party() {
    async_exec::block_on(async move {
        let mssg = b"hello world";
        let addresses = vec![
            String::from("127.0.0.1:8001"),
            String::from("127.0.0.1:8002"),
        ];

        let setup_net_1 = {
            let cloned_addresses = addresses.clone();
            async_exec::spawn(async move { Network::new_tcp(0, &cloned_addresses).await })
        };
        let setup_net_2 = async_exec::spawn(async move { Network::new_tcp(1, &addresses).await });
        let (net1, net2) = smol::future::zip(setup_net_1, setup_net_2).await;

        let party1 = async_exec::spawn(async move {
            net1.send(1, mssg.to_vec()).await;
            let (pid, data) = net1.recv().await;
            assert_eq!(pid, 1);
            assert_eq!(data, mssg);
            net1.close();
        });

        let party2 = async_exec::spawn(async move {
            let (pid, data) = net2.recv().await;
            assert_eq!(pid, 0);
            assert_eq!(data, mssg);
            net2.send(0, mssg.to_vec()).await;
            net2.close();
        });

        party1.await;
        party2.await;
    });
}

#[test]
fn net_local_many_parties() {
    async_exec::block_on(async move {
        let num = 10;
        let mssg = b"hello world";

        let nets: Vec<_> = Network::new_local(num);
        let mut handles = Vec::new();
        for (pid, net) in nets.into_iter().enumerate() {
            handles.push(async_exec::spawn(async move {
                for i in 0..num {
                    if i == pid {
                        net.send_all(mssg.to_vec()).await;
                    }

                    let (rid, data) = net.recv().await;
                    assert_eq!(rid, i.try_into().unwrap());
                    assert_eq!(data, mssg);
                }
                net.close();
            }));
        }

        for handle in handles {
            handle.await;
        }
    });
}

#[test]
fn net_tcp_many_parties() {
    async_exec::block_on(async move {
        let num = 5;
        let mssg = b"hello world";
        let addresses: Vec<_> = (0..num)
            .map(|i| format!("127.0.0.1:{}", 5000 + i))
            .collect();

        let mut handles = Vec::new();
        for pid in 0..num {
            let cloned_addresses = addresses.clone();
            handles.push(async_exec::spawn(async move {
                let net = Network::new_tcp(pid, &cloned_addresses).await;
                for i in 0..num {
                    if i == pid {
                        net.send_all(mssg.to_vec()).await;
                    }

                    let (rid, data) = net.recv().await;
                    assert_eq!(rid, i.try_into().unwrap());
                    assert_eq!(data, mssg);
                }
                net.close();
            }));
        }

        for handle in handles {
            handle.await;
        }
    });
}
