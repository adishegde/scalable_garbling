use scalable_mpc::math;
use scalable_mpc::math::galois;
use scalable_mpc::protocol::{network, preproc, MPCContext};
use scalable_mpc::sharing;
use scalable_mpc::{block_on, spawn};
use serial_test::serial;
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
        let proto_id = b"rand protocol".to_vec();

        let (gf, pss, contexts) = setup().await;
        let super_inv_matrix = Arc::new(math::super_inv_matrix(N, N - T, gf.as_ref()));
        let mut handles = Vec::new();

        for context in contexts {
            let rand_context = preproc::RandContext::new(super_inv_matrix.clone(), &context);
            handles.push(spawn(preproc::rand(proto_id.clone(), rand_context)));
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
        let proto_id = b"rand protocol".to_vec();

        let (gf, pss, contexts) = setup().await;
        let super_inv_matrix = Arc::new(math::super_inv_matrix(N, N - T, gf.as_ref()));
        let mut handles = Vec::new();

        for context in contexts {
            let zero_context = preproc::ZeroContext::new(super_inv_matrix.clone(), &context);
            handles.push(spawn(preproc::zero(proto_id.clone(), zero_context)));
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