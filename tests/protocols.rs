use rand::rngs::mock::StepRng;
use scalable_mpc::math;
use scalable_mpc::math::galois;
use scalable_mpc::protocol::{core, network, preproc, MPCContext};
use scalable_mpc::sharing;
use scalable_mpc::{block_on, spawn};
use serial_test::serial;
use std::path::PathBuf;
use std::sync::Arc;

const GF_WIDTH: u8 = 18;
const N: usize = 16;
const T: usize = 5;
const L: usize = 3;

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
            n: N as usize,
            t: T as usize,
            l: L as usize,
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

        let mut handles = Vec::new();
        for context in contexts {
            let context = preproc::RandBitContext::new(bin_supinv_matrix.clone(), &context);
            handles.push(spawn(preproc::randbit(proto_id.clone(), context)));
        }

        let mut sharings = vec![Vec::with_capacity(N); bin_supinv_matrix.len()];

        for handle in handles {
            let shares = handle.await;

            assert_eq!(shares.len(), bin_supinv_matrix.len());

            for (i, share) in shares.into_iter().enumerate() {
                sharings[i].push(share);
            }
        }

        for sharing in sharings {
            let secrets = pss.recon_n(&sharing, gf.as_ref());
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
        let mut rng = StepRng::new(1, 1);
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
        let mut rng = StepRng::new(1, 1);
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
