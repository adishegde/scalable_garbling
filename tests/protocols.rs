use rand::rngs::StdRng;
use rand::SeedableRng;
use scalable_mpc::circuit::Circuit;
use scalable_mpc::math;
use scalable_mpc::math::galois;
use scalable_mpc::math::Combination;
use scalable_mpc::protocol::{core, garble, network, preproc, MPCContext};
use scalable_mpc::sharing;
use scalable_mpc::{block_on, spawn};
use serial_test::serial;
use std::path::PathBuf;
use std::sync::Arc;

const GF_WIDTH: u8 = 18;
const N: usize = 16;
const T: usize = 5;
const L: usize = 3;
const LPN_TAU: usize = 4;
const LPN_KEY_LEN: usize = 5;
const LPN_MSSG_LEN: usize = 7;

async fn setup() -> (
    Arc<galois::GF>,
    Arc<sharing::PackedSharing>,
    Vec<MPCContext>,
) {
    let gf = Arc::new(galois::GF::new(GF_WIDTH).unwrap());
    let pss = Arc::new(sharing::PackedSharing::new(
        N.try_into().unwrap(),
        T.try_into().unwrap(),
        L.try_into().unwrap(),
        gf.as_ref(),
    ));
    let comms = network::setup_local_network(N as usize).await;

    let mut contexts = Vec::new();

    for (pid, (_, net)) in comms.into_iter().enumerate() {
        let gf = gf.clone();
        let pss = pss.clone();

        contexts.push(MPCContext {
            id: pid.try_into().unwrap(),
            n: N,
            t: T,
            l: L,
            lpn_tau: LPN_TAU,
            lpn_key_len: LPN_KEY_LEN,
            lpn_mssg_len: LPN_MSSG_LEN,
            gf,
            pss,
            net_builder: net,
        });
    }

    (gf, pss, contexts)
}

#[test]
#[serial]
fn rand() {
    block_on(async {
        let proto_id = b"protocol".to_vec();

        let (gf, pss, contexts) = setup().await;
        let super_inv_matrix = Arc::new(math::super_inv_matrix(N, N - T, gf.as_ref()));
        let mut handles = Vec::new();

        for context in contexts {
            let context = preproc::RandContext::new(super_inv_matrix.clone(), &context);
            handles.push(spawn(preproc::rand(proto_id.clone(), context)));
        }

        let mut sharings = vec![Vec::with_capacity(N); N - T];

        for handle in handles {
            let shares = handle.await;

            assert_eq!(shares.len(), N - T);

            for (i, share) in shares.into_iter().enumerate() {
                sharings[i].push(share);
            }
        }

        for sharing in sharings {
            pss.recon(&sharing, gf.as_ref()).unwrap();
        }
    });
}

#[test]
#[serial]
fn zero() {
    block_on(async {
        let proto_id = b"protocol".to_vec();

        let (gf, pss, contexts) = setup().await;
        let super_inv_matrix = Arc::new(math::super_inv_matrix(N, N - T, gf.as_ref()));
        let mut handles = Vec::new();

        for context in contexts {
            let context = preproc::ZeroContext::new(super_inv_matrix.clone(), &context);
            handles.push(spawn(preproc::zero(proto_id.clone(), context)));
        }

        let mut sharings = vec![Vec::with_capacity(N); N - T];

        for handle in handles {
            let shares = handle.await;

            assert_eq!(shares.len(), N - T);

            for (i, share) in shares.into_iter().enumerate() {
                sharings[i].push(share);
            }
        }

        let exp_out = vec![gf.zero(); L];

        for sharing in sharings {
            assert_eq!(exp_out, pss.recon_n(&sharing, gf.as_ref()));
        }
    });
}

#[test]
#[serial]
fn randbit() {
    block_on(async {
        let proto_id = b"protocol".to_vec();

        let (gf, pss, contexts) = setup().await;

        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data/n16_t5.txt");
        let bin_supinv_matrix = Arc::new(math::binary_super_inv_matrix(&path, &gf));

        let pos = vec![gf.get(101), gf.get(102), gf.get(103)];
        let share_coeffs = Arc::new(pss.share_coeffs(&pos, &gf));

        let mut handles = Vec::new();
        for context in contexts {
            let context = preproc::RandBitContext::new(bin_supinv_matrix.clone(), &context);
            handles.push(spawn(preproc::randbit(
                proto_id.clone(),
                share_coeffs.clone(),
                context,
            )));
        }

        let mut sharings = vec![Vec::with_capacity(N); bin_supinv_matrix.len()];

        for handle in handles {
            let shares = handle.await;

            assert_eq!(shares.len(), bin_supinv_matrix.len());

            for (i, share) in shares.into_iter().enumerate() {
                sharings[i].push(share);
            }
        }

        let recon_coeffs = pss.recon_coeffs_n(&pos, &gf);
        for sharing in sharings {
            let secrets = math::utils::matrix_vector_prod(&recon_coeffs, &sharing, &gf);
            for secret in secrets {
                assert!((secret == gf.one()) || (secret == gf.zero()));
            }
        }
    });
}

#[test]
#[serial]
fn reduce_degree() {
    block_on(async {
        let mut rng = StdRng::seed_from_u64(200);
        let proto_id = b"protocol".to_vec();

        let (gf, pss, contexts) = setup().await;

        let secrets: Vec<_> = gf.get_range(1..(L + 1).try_into().unwrap()).collect();
        let x_shares = pss.share(&secrets, gf.as_ref(), &mut rng);
        let zero_shares = pss.share(&vec![gf.zero(); L], gf.as_ref(), &mut rng);
        let rand_shares = pss.rand(gf.as_ref(), &mut rng);

        let mut handles = Vec::new();
        for (i, context) in contexts.into_iter().enumerate() {
            handles.push(spawn(core::reduce_degree(
                proto_id.clone(),
                x_shares[i],
                rand_shares[i],
                zero_shares[i],
                context,
            )));
        }

        let mut out_shares = Vec::with_capacity(N);
        for handle in handles {
            out_shares.push(handle.await);
        }

        let out_secrets = pss.recon(&out_shares, gf.as_ref()).unwrap();

        assert_eq!(out_secrets, secrets);
    });
}

#[test]
#[serial]
fn mult() {
    block_on(async {
        let mut rng = StdRng::seed_from_u64(200);
        let proto_id = b"protocol".to_vec();

        let (gf, pss, contexts) = setup().await;

        let x_secrets: Vec<_> = gf.get_range(1..(L + 1).try_into().unwrap()).collect();
        let y_secrets: Vec<_> = gf
            .get_range((L + 1).try_into().unwrap()..(2 * L + 1).try_into().unwrap())
            .collect();
        let x_shares = pss.share(&x_secrets, gf.as_ref(), &mut rng);
        let y_shares = pss.share(&y_secrets, gf.as_ref(), &mut rng);
        let zero_shares = pss.share(&vec![gf.zero(); L], gf.as_ref(), &mut rng);
        let rand_shares = pss.rand(gf.as_ref(), &mut rng);

        let mut handles = Vec::new();
        for (i, context) in contexts.into_iter().enumerate() {
            handles.push(spawn(core::mult(
                proto_id.clone(),
                x_shares[i],
                y_shares[i],
                rand_shares[i],
                zero_shares[i],
                context,
            )));
        }

        let mut out_shares = Vec::with_capacity(N);
        for handle in handles {
            out_shares.push(handle.await);
        }

        let out_secrets = pss.recon(&out_shares, gf.as_ref()).unwrap();
        let exp: Vec<_> = x_secrets
            .into_iter()
            .zip(y_secrets.into_iter())
            .map(|(xval, yval)| xval * yval)
            .collect();

        assert_eq!(out_secrets, exp);
    });
}

#[test]
#[serial]
fn trans() {
    block_on(async {
        let mut rng = StdRng::seed_from_u64(200);
        let proto_id = b"protocol".to_vec();

        let (gf, pss, contexts) = setup().await;

        let l: u32 = L.try_into().unwrap();
        let secrets: Vec<_> = gf.get_range(1..(l + 1)).collect();
        let rand_secrets: Vec<_> = (0..l).map(|_| gf.rand(&mut rng)).collect();
        let old_pos: Vec<_> = gf.get_range(200..(200 + l)).collect();
        let new_pos: Vec<_> = gf.get_range(300..(300 + l)).collect();
        let f_trans = Combination::new(vec![1, 2, 1]);

        let old_shares = {
            let coeffs = pss.share_coeffs(&old_pos, gf.as_ref());
            pss.share_using_coeffs(secrets.clone(), &coeffs, gf.as_ref(), &mut rng)
        };
        let random_n = {
            let coeffs = pss.share_coeffs_n(&old_pos, gf.as_ref());
            pss.share_using_coeffs(rand_secrets.clone(), &coeffs, gf.as_ref(), &mut rng)
        };
        let random = {
            let coeffs = pss.share_coeffs(&new_pos, gf.as_ref());
            pss.share_using_coeffs(f_trans.apply(&rand_secrets), &coeffs, gf.as_ref(), &mut rng)
        };

        let mut handles = Vec::new();
        for (i, context) in contexts.into_iter().enumerate() {
            let transform =
                core::SharingTransform::new(&old_pos, &new_pos, f_trans.clone(), &pss, &gf);

            handles.push(spawn(core::trans(
                proto_id.clone(),
                old_shares[i],
                random[i],
                random_n[i],
                transform,
                context,
            )));
        }

        let mut out_shares = Vec::with_capacity(N);
        for handle in handles {
            out_shares.push(handle.await);
        }

        let recon_coeffs = pss.recon_coeffs_n(&new_pos, gf.as_ref());
        let out_secrets: Vec<_> =
            math::utils::matrix_vector_prod(&recon_coeffs, &out_shares, gf.as_ref()).collect();

        assert_eq!(out_secrets, f_trans.apply(&secrets));
    });
}

#[test]
#[serial]
fn trans_incomplete_block() {
    block_on(async {
        let mut rng = StdRng::seed_from_u64(200);
        let proto_id = b"protocol".to_vec();

        let (gf, pss, contexts) = setup().await;

        let secrets = vec![gf.get(37)];
        let rand_secrets = vec![gf.rand(&mut rng)];
        let old_pos = vec![gf.get(200)];
        let new_pos = vec![gf.get(300), gf.get(301)];
        let f_trans = Combination::new(vec![0, 0]);

        let old_shares = {
            let coeffs = pss.share_coeffs(&old_pos, gf.as_ref());
            pss.share_using_coeffs(secrets.clone(), &coeffs, gf.as_ref(), &mut rng)
        };
        let random_n = {
            let coeffs = pss.share_coeffs_n(&old_pos, gf.as_ref());
            pss.share_using_coeffs(rand_secrets.clone(), &coeffs, gf.as_ref(), &mut rng)
        };
        let random = {
            let coeffs = pss.share_coeffs(&new_pos, gf.as_ref());
            pss.share_using_coeffs(f_trans.apply(&rand_secrets), &coeffs, gf.as_ref(), &mut rng)
        };

        let mut handles = Vec::new();
        for (i, context) in contexts.into_iter().enumerate() {
            let transform =
                core::SharingTransform::new(&old_pos, &new_pos, f_trans.clone(), &pss, &gf);

            handles.push(spawn(core::trans(
                proto_id.clone(),
                old_shares[i],
                random[i],
                random_n[i],
                transform,
                context,
            )));
        }

        let mut out_shares = Vec::with_capacity(N);
        for handle in handles {
            out_shares.push(handle.await);
        }

        let recon_coeffs = pss.recon_coeffs_n(&new_pos, gf.as_ref());
        let out_secrets: Vec<_> =
            math::utils::matrix_vector_prod(&recon_coeffs, &out_shares, gf.as_ref()).collect();

        assert_eq!(out_secrets, f_trans.apply(&secrets));
    });
}

#[test]
#[serial]
fn randtrans() {
    block_on(async {
        let mut rng = StdRng::seed_from_u64(200);
        let proto_id = b"protocol".to_vec();

        let (gf, pss, contexts) = setup().await;

        let l: u32 = L.try_into().unwrap();
        let old_pos: Vec<Vec<galois::GFElement>> = vec![
            gf.get_range(200..(200 + l)).collect(),
            gf.get_range(250..(250 + l)).collect(),
            gf.get_range(225..(225 + l)).collect(),
        ];
        let new_pos: Vec<_> = vec![
            gf.get_range(300..(300 + l)).collect(),
            gf.get_range(350..(350 + l)).collect(),
            gf.get_range(325..(325 + l)).collect(),
        ];
        let f_trans = vec![
            Combination::new((0..L).collect()),
            Combination::new(vec![2, 1, 0]),
            Combination::new(vec![2, 0, 2]),
        ];

        let randoms: Vec<_> = (0..(N + T))
            .map(|_| pss.rand(gf.as_ref(), &mut rng))
            .collect();
        let zeros: Vec<_> = {
            let secrets = vec![gf.zero(); L];
            (0..(2 * N))
                .map(|_| pss.share_n(&secrets, gf.as_ref(), &mut rng))
                .collect()
        };

        let mut handles = Vec::new();
        for (i, context) in contexts.into_iter().enumerate() {
            let randoms: Vec<_> = randoms.iter().map(|v| v[i]).collect();
            let zeros: Vec<_> = zeros.iter().map(|v| v[i]).collect();
            let transform = core::RandSharingTransform::new(
                i.try_into().unwrap(),
                &old_pos,
                &new_pos,
                &f_trans,
                pss.as_ref(),
                gf.as_ref(),
            );

            handles.push(spawn(core::randtrans(
                proto_id.clone(),
                randoms,
                zeros,
                transform,
                context,
            )));
        }

        let mut sharings = vec![Vec::with_capacity(N); L];
        let mut sharings_n = vec![Vec::with_capacity(N); L];
        for handle in handles {
            let shares = handle.await;
            for (i, share) in shares.into_iter().enumerate() {
                sharings[i].push(share.0);
                sharings_n[i].push(share.1);
            }
        }

        for (i, (sharing, sharing_n)) in
            sharings.into_iter().zip(sharings_n.into_iter()).enumerate()
        {
            let secrets: Vec<_> = {
                let coeffs = pss.recon_coeffs_n(&new_pos[i], gf.as_ref());
                math::utils::matrix_vector_prod(&coeffs, &sharing, gf.as_ref()).collect()
            };

            let secrets_n: Vec<_> = {
                let coeffs = pss.recon_coeffs_n(&old_pos[i], gf.as_ref());
                math::utils::matrix_vector_prod(&coeffs, &sharing_n, gf.as_ref()).collect()
            };

            assert_eq!(secrets, f_trans[i].apply(&secrets_n));
        }
    });
}

#[test]
#[serial]
fn randtrans_incomplete_block() {
    block_on(async {
        let mut rng = StdRng::seed_from_u64(200);
        let proto_id = b"protocol".to_vec();

        let (gf, pss, contexts) = setup().await;

        let l: u32 = L.try_into().unwrap();
        let old_pos: Vec<Vec<galois::GFElement>> =
            vec![vec![gf.get(200)], gf.get_range(225..(225 + l)).collect()];
        let new_pos: Vec<_> = vec![
            vec![gf.get(300), gf.get(301), gf.get(302)],
            vec![gf.get(325)],
        ];
        let f_trans = vec![Combination::new(vec![0, 0, 0]), Combination::new(vec![1])];

        let randoms: Vec<_> = (0..(N + T))
            .map(|_| pss.rand(gf.as_ref(), &mut rng))
            .collect();
        let zeros: Vec<_> = {
            let secrets = vec![gf.zero(); L];
            (0..(2 * N))
                .map(|_| pss.share_n(&secrets, gf.as_ref(), &mut rng))
                .collect()
        };

        let mut handles = Vec::new();
        for (i, context) in contexts.into_iter().enumerate() {
            let randoms: Vec<_> = randoms.iter().map(|v| v[i]).collect();
            let zeros: Vec<_> = zeros.iter().map(|v| v[i]).collect();
            let transform = core::RandSharingTransform::new(
                i.try_into().unwrap(),
                &old_pos,
                &new_pos,
                &f_trans,
                pss.as_ref(),
                gf.as_ref(),
            );

            handles.push(spawn(core::randtrans(
                proto_id.clone(),
                randoms,
                zeros,
                transform,
                context,
            )));
        }

        let mut sharings = vec![Vec::with_capacity(N); 2];
        let mut sharings_n = vec![Vec::with_capacity(N); 2];
        for handle in handles {
            let shares = handle.await;
            for i in 0..2 {
                sharings[i].push(shares[i].0);
                sharings_n[i].push(shares[i].1);
            }
        }

        for (i, (sharing, sharing_n)) in
            sharings.into_iter().zip(sharings_n.into_iter()).enumerate()
        {
            if i >= 2 {
                assert_eq!(sharing, vec![gf.get(0); N]);
                assert_eq!(sharing_n, vec![gf.get(0); N]);
            }

            let secrets: Vec<_> = {
                let coeffs = pss.recon_coeffs_n(&new_pos[i], gf.as_ref());
                math::utils::matrix_vector_prod(&coeffs, &sharing, gf.as_ref()).collect()
            };

            let secrets_n: Vec<_> = {
                let coeffs = pss.recon_coeffs_n(&old_pos[i], gf.as_ref());
                math::utils::matrix_vector_prod(&coeffs, &sharing_n, gf.as_ref()).collect()
            };

            assert_eq!(secrets, f_trans[i].apply(&secrets_n));
        }
    });
}

#[test]
#[serial]
fn garble() {
    block_on(async {
        let circ = {
            let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.push("tests/data/sub64.txt");
            let circ = Circuit::from_bristol_fashion(&path);
            circ.pack(L as u32)
        };

        let (_, _, contexts) = setup().await;

        let mut handles = Vec::new();
        for (i, context) in contexts.into_iter().enumerate() {
            let circ = circ.clone();
            let preproc = preproc::PreProc::dummy(i.try_into().unwrap(), 200, &circ, &context);
            let context = Arc::new(garble::GarbleContextData::new(circ, context));
            handles.push(spawn(async move {
                garble::garble(b"".to_vec(), preproc, context).await
            }))
        }

        for handle in handles {
            let gc = handle.await;

            assert_eq!(gc.gates.len(), circ.gates().len());

            for gate in gc.gates {
                match gate {
                    garble::GarbledGate::And(table) => {
                        assert_eq!(table.ctxs.len(), 4);
                        for ctx in table.ctxs {
                            assert_eq!(ctx.len(), LPN_MSSG_LEN);
                        }
                    }
                    garble::GarbledGate::Xor(table) => {
                        assert_eq!(table.ctxs.len(), 4);
                        for ctx in table.ctxs {
                            assert_eq!(ctx.len(), LPN_MSSG_LEN);
                        }
                    }
                    garble::GarbledGate::Inv(table) => {
                        assert_eq!(table.ctxs.len(), 2);
                        for ctx in table.ctxs {
                            assert_eq!(ctx.len(), LPN_MSSG_LEN);
                        }
                    }
                }
            }
        }
    });
}
