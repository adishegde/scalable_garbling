use ndarray::{s, Array2, Array3, ArrayView1, ArrayView2, Axis};
use rand::{rngs::StdRng, Rng, SeedableRng};
use scalable_mpc::circuit::Circuit;
use scalable_mpc::math;
use scalable_mpc::math::{galois::GF, Combination};
use scalable_mpc::protocol::{core, network, preproc, MPCContext};
use scalable_mpc::sharing::PackedSharing;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task::spawn;

const W: u8 = 18;
const N: usize = 16;
const T: usize = 5;
const L: usize = 3;
const LPN_TAU: usize = 2;
const LPN_KEY_LEN: usize = 5;
const LPN_MSSG_LEN: usize = 7;
const NUM: usize = 10;

async fn setup() -> (
    Arc<PackedSharing<W>>,
    Arc<PackedSharing<W>>,
    Vec<(network::Network, MPCContext<W>)>,
) {
    GF::<W>::init().unwrap();

    let n: u32 = N.try_into().unwrap();
    let t: u32 = T.try_into().unwrap();
    let l: u32 = L.try_into().unwrap();

    let defpos = PackedSharing::default_pos(n, l);
    let pss = Arc::new(PackedSharing::new(t + l - 1, n, &defpos));
    let pss_n = Arc::new(PackedSharing::new(n - 1, n, &defpos));
    let comms = network::setup_local_network(n.try_into().unwrap()).await;

    let mut res = Vec::with_capacity(N);

    for (pid, (_, net)) in comms.into_iter().enumerate() {
        let pss = pss.clone();
        let pss_n = pss_n.clone();

        let context = MPCContext {
            id: pid.try_into().unwrap(),
            n: N,
            t: T,
            l: L,
            lpn_tau: LPN_TAU,
            lpn_key_len: LPN_KEY_LEN,
            lpn_mssg_len: LPN_MSSG_LEN,
            pss,
            pss_n,
        };

        res.push((net, context));
    }

    (pss, pss_n, res)
}

fn rand_shares<R: Rng, const W: u8>(
    num: usize,
    pss: &PackedSharing<W>,
    rng: &mut R,
) -> Array2<GF<W>> {
    let shares = (0..num).flat_map(|_| pss.rand(rng)).collect();
    Array2::from_shape_vec((num, pss.num_parties() as usize), shares).unwrap()
}

fn shares<R: Rng, const W: u8>(
    secrets: ArrayView2<GF<W>>,
    pss: &PackedSharing<W>,
    rng: &mut R,
) -> Array2<GF<W>> {
    assert_eq!(secrets.shape()[1], pss.num_secrets() as usize);

    let shares = secrets
        .axis_iter(Axis(0))
        .flat_map(|row| pss.share(row, rng))
        .collect();
    Array2::from_shape_vec((secrets.shape()[0], pss.num_parties() as usize), shares).unwrap()
}

#[tokio::test]
async fn reduce_degree() {
    let mut rng = StdRng::seed_from_u64(200);
    let proto_id = b"protocol".to_vec();

    let (pss, pss_n, contexts) = setup().await;

    let secrets = Array2::from_shape_vec(
        (NUM, L),
        (0..(NUM * L)).map(|_| GF::<W>::rand(&mut rng)).collect(),
    )
    .unwrap();
    let vx = shares(secrets.view(), &pss, &mut rng);
    let zero_shares = shares(Array2::zeros((NUM, L)).view(), &pss_n, &mut rng);
    let rand_shares = rand_shares(NUM, &pss, &mut rng);

    let mut handles = Vec::new();
    for (i, (net, context)) in contexts.into_iter().enumerate() {
        handles.push(spawn(core::reduce_degree(
            proto_id.clone(),
            vx.slice(s![.., i]).to_owned(),
            rand_shares.slice(s![.., i]).to_owned(),
            zero_shares.slice(s![.., i]).to_owned(),
            None,
            context,
            net,
        )));
    }

    let mut out_shares = Vec::with_capacity(N * NUM);
    for handle in handles {
        out_shares.extend_from_slice(handle.await.unwrap().view().to_slice().unwrap());
    }
    let out_secrets = Array2::from_shape_vec(
        (NUM, L),
        Array2::from_shape_vec((N, NUM), out_shares)
            .unwrap()
            .axis_iter(Axis(1))
            .flat_map(|s| pss.semihon_recon(s))
            .collect(),
    )
    .unwrap();
    assert_eq!(out_secrets, secrets);
}

#[tokio::test]
async fn mult() {
    let mut rng = StdRng::seed_from_u64(200);
    let proto_id = b"protocol".to_vec();

    let (pss, pss_n, contexts) = setup().await;

    let x_secrets = Array2::from_shape_vec(
        (NUM, L),
        (0..(NUM * L)).map(|_| GF::<W>::rand(&mut rng)).collect(),
    )
    .unwrap();
    let y_secrets = Array2::from_shape_vec(
        (NUM, L),
        (0..(NUM * L)).map(|_| GF::<W>::rand(&mut rng)).collect(),
    )
    .unwrap();
    let vx = shares(x_secrets.view(), &pss, &mut rng);
    let vy = shares(y_secrets.view(), &pss, &mut rng);
    let zero_shares = shares(Array2::zeros((NUM, L)).view(), &pss_n, &mut rng);
    let rand_shares = rand_shares(NUM, &pss, &mut rng);

    let mut handles = Vec::new();
    for (i, (net, context)) in contexts.into_iter().enumerate() {
        handles.push(spawn(core::mult(
            proto_id.clone(),
            vx.slice(s![.., i]).to_owned(),
            vy.slice(s![.., i]).to_owned(),
            rand_shares.slice(s![.., i]).to_owned(),
            zero_shares.slice(s![.., i]).to_owned(),
            None,
            context,
            net,
        )));
    }

    let mut out_shares = Vec::with_capacity(N * NUM);
    for handle in handles {
        out_shares.extend_from_slice(handle.await.unwrap().view().to_slice().unwrap());
    }
    let out_secrets = Array2::from_shape_vec(
        (NUM, L),
        Array2::from_shape_vec((N, NUM), out_shares)
            .unwrap()
            .axis_iter(Axis(1))
            .flat_map(|s| pss.semihon_recon(s))
            .collect(),
    )
    .unwrap();
    assert_eq!(out_secrets, x_secrets * y_secrets);
}

#[tokio::test]
async fn rand() {
    let proto_id = b"protocol".to_vec();

    let (pss, _, contexts) = setup().await;
    let super_inv_matrix = Arc::new(math::super_inv_matrix(N, N - T));
    let mut handles = Vec::new();

    for (net, context) in contexts {
        let context = preproc::RandContext::new(super_inv_matrix.clone(), &context);
        handles.push(spawn(preproc::rand(
            proto_id.clone(),
            NUM * (N - T),
            context,
            net,
        )));
    }

    let mut sharings = Vec::with_capacity(N * NUM * (N - T));
    for handle in handles {
        let shares = handle.await.unwrap();
        assert_eq!(shares.len(), NUM * (N - T));
        sharings.extend_from_slice(shares.as_slice().unwrap());
    }
    let sharings = Array2::from_shape_vec((N, NUM * (N - T)), sharings).unwrap();

    for s in sharings.axis_iter(Axis(1)) {
        pss.recon(s).unwrap();
    }
}

#[tokio::test]
async fn zero() {
    let proto_id = b"protocol".to_vec();

    let (_, pss_n, contexts) = setup().await;
    let super_inv_matrix = Arc::new(math::super_inv_matrix(N, N - T));
    let mut handles = Vec::new();

    for (net, context) in contexts {
        let context = preproc::ZeroContext::new(super_inv_matrix.clone(), &context);
        handles.push(spawn(preproc::zero(
            proto_id.clone(),
            NUM * (N - T),
            context,
            net,
        )));
    }

    let mut sharings = Vec::with_capacity(N * NUM * (N - T));
    for handle in handles {
        let shares = handle.await.unwrap();
        assert_eq!(shares.len(), NUM * (N - T));
        sharings.extend_from_slice(shares.as_slice().unwrap());
    }
    let sharings = Array2::from_shape_vec((N, NUM * (N - T)), sharings).unwrap();

    let exp_out = vec![GF::ZERO; L];
    for s in sharings.axis_iter(Axis(1)) {
        assert_eq!(pss_n.semihon_recon(s), exp_out);
    }
}

#[tokio::test]
async fn randbit() {
    let proto_id = b"protocol".to_vec();

    let (pss, _, contexts) = setup().await;

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/n16_t5.txt");
    let bin_supinv_matrix = Arc::new(math::binary_super_inv_matrix(&path));
    let batch_size = bin_supinv_matrix.shape()[0];

    let mut handles = Vec::new();
    for (net, context) in contexts {
        let context = preproc::RandBitContext::new(bin_supinv_matrix.clone(), &context);
        handles.push(spawn(preproc::randbit(
            proto_id.clone(),
            NUM * batch_size,
            context,
            net,
        )));
    }

    let mut sharings = Vec::with_capacity(N * NUM * batch_size);
    for handle in handles {
        let shares = handle.await.unwrap();
        assert_eq!(shares.len(), NUM * batch_size);
        sharings.extend_from_slice(shares.as_slice().unwrap());
    }
    let sharings = Array2::from_shape_vec((N, NUM * batch_size), sharings).unwrap();

    for s in sharings.axis_iter(Axis(1)) {
        let secrets = pss.recon(s).unwrap();
        for secret in secrets {
            assert!(secret == GF::ZERO || secret == GF::ONE);
        }
    }
}

#[tokio::test]
async fn preproc() {
    let circ = {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data/sub64.txt");
        let circ = Circuit::from_bristol_fashion(&path);
        circ.pack(L as u32)
    };

    let (_, _, contexts) = setup().await;

    let bin_supinv_matrix = {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data/n16_t5.txt");
        Arc::new(math::binary_super_inv_matrix(&path))
    };

    let super_inv_matrix = Arc::new(math::super_inv_matrix(N, N - T));

    let dummy_preproc = preproc::PreProc::dummy(0, 0, &circ, &contexts[0].1);
    let desc = preproc::PreProc::describe(&circ, &contexts[0].1);

    let mut handles = Vec::new();
    for (net, context) in contexts {
        let rcontext = preproc::RandContext::new(super_inv_matrix.clone(), &context);
        let zcontext = preproc::ZeroContext::new(super_inv_matrix.clone(), &context);
        let bcontext = preproc::RandBitContext::new(bin_supinv_matrix.clone(), &context);

        handles.push(spawn(async move {
            preproc::preproc(
                b"".to_vec(),
                desc,
                context,
                rcontext,
                zcontext,
                bcontext,
                net,
            )
            .await
        }))
    }

    for handle in handles {
        let preproc = handle.await.unwrap();

        assert_eq!(preproc.masks.shape(), dummy_preproc.masks.shape());
        assert_eq!(preproc.keys.shape(), dummy_preproc.keys.shape());
        assert_eq!(preproc.errors.len(), dummy_preproc.errors.len());
        assert!(preproc.randoms.len() >= dummy_preproc.randoms.len());
        assert!(preproc.zeros.len() >= dummy_preproc.zeros.len());
    }
}

#[tokio::test]
async fn trans() {
    let mut rng = StdRng::seed_from_u64(200);
    let proto_id = b"protocol".to_vec();

    let (_, _, contexts) = setup().await;

    let secrets = Array2::from_shape_vec(
        (NUM, L),
        (0..(NUM * L)).map(|_| GF::<W>::rand(&mut rng)).collect(),
    )
    .unwrap();
    let rand_secrets = Array2::from_shape_vec(
        (NUM, L),
        (0..(NUM * L)).map(|_| GF::<W>::rand(&mut rng)).collect(),
    )
    .unwrap();
    let old_pos: Vec<_> = (200..(200 + L).try_into().unwrap()).map(GF::from).collect();
    let new_pos: Vec<_> = (300..(300 + L).try_into().unwrap()).map(GF::from).collect();
    let f_trans = Combination::new(vec![1, 2, 1]);

    let pss_old = PackedSharing::new(
        (T + L - 1).try_into().unwrap(),
        N.try_into().unwrap(),
        &old_pos,
    );
    let pss_n_old =
        PackedSharing::new((N - 1).try_into().unwrap(), N.try_into().unwrap(), &old_pos);
    let pss_new = PackedSharing::new(
        (T + L - 1).try_into().unwrap(),
        N.try_into().unwrap(),
        &new_pos,
    );

    let old_shares = shares(secrets.view(), &pss_old, &mut rng);
    let random_n = shares(rand_secrets.view(), &pss_n_old, &mut rng);
    let random = {
        let tf_rand_secrets = rand_secrets
            .rows()
            .into_iter()
            .flat_map(|row| f_trans.apply(row))
            .collect();
        let tf_rand_secrets = Array2::from_shape_vec((NUM, L), tf_rand_secrets).unwrap();
        shares(tf_rand_secrets.view(), &pss_new, &mut rng)
    };

    let mut handles = Vec::new();
    for (i, (net, context)) in contexts.into_iter().enumerate() {
        let transform = core::SharingTransform::new(&old_pos, &new_pos, f_trans.clone(), &context);

        handles.push(spawn(core::trans(
            proto_id.clone(),
            old_shares.slice(s![.., i]).to_owned(),
            random.slice(s![.., i]).to_owned(),
            random_n.slice(s![.., i]).to_owned(),
            transform,
            context,
            net,
        )));
    }

    let mut out_shares = Vec::with_capacity(N * NUM);
    for handle in handles {
        out_shares.extend_from_slice(handle.await.unwrap().view().to_slice().unwrap());
    }

    let exp_secrets = {
        let secrets = secrets
            .rows()
            .into_iter()
            .flat_map(|row| f_trans.apply(row))
            .collect();
        Array2::from_shape_vec((NUM, L), secrets).unwrap()
    };

    let out_secrets = Array2::from_shape_vec(
        (NUM, L),
        Array2::from_shape_vec((N, NUM), out_shares)
            .unwrap()
            .axis_iter(Axis(1))
            .flat_map(|s| pss_new.recon(s).unwrap())
            .collect(),
    )
    .unwrap();
    assert_eq!(out_secrets, exp_secrets);
}

#[tokio::test]
async fn trans_incomplete_block() {
    let mut rng = StdRng::seed_from_u64(200);
    let proto_id = b"protocol".to_vec();

    let (_, _, contexts) = setup().await;

    let secrets = Array2::from_shape_vec(
        (NUM, 1),
        (0..NUM).map(|_| GF::<W>::rand(&mut rng)).collect(),
    )
    .unwrap();
    let rand_secrets = Array2::from_shape_vec(
        (NUM, 1),
        (0..NUM).map(|_| GF::<W>::rand(&mut rng)).collect(),
    )
    .unwrap();
    let old_pos = vec![GF::from(200u32)];
    let new_pos = vec![GF::from(300u32), GF::from(301u32)];
    let f_trans = Combination::new(vec![0, 0]);

    let pss_old = PackedSharing::new(
        (T + L - 1).try_into().unwrap(),
        N.try_into().unwrap(),
        &old_pos,
    );
    let pss_n_old =
        PackedSharing::new((N - 1).try_into().unwrap(), N.try_into().unwrap(), &old_pos);
    let pss_new = PackedSharing::new(
        (T + L - 1).try_into().unwrap(),
        N.try_into().unwrap(),
        &new_pos,
    );

    let old_shares = shares(secrets.view(), &pss_old, &mut rng);
    let random_n = shares(rand_secrets.view(), &pss_n_old, &mut rng);
    let random = {
        let tf_rand_secrets = rand_secrets
            .rows()
            .into_iter()
            .flat_map(|row| f_trans.apply(row))
            .collect();
        let tf_rand_secrets = Array2::from_shape_vec((NUM, 2), tf_rand_secrets).unwrap();
        shares(tf_rand_secrets.view(), &pss_new, &mut rng)
    };

    let mut handles = Vec::new();
    for (i, (net, context)) in contexts.into_iter().enumerate() {
        let transform = core::SharingTransform::new(&old_pos, &new_pos, f_trans.clone(), &context);

        handles.push(spawn(core::trans(
            proto_id.clone(),
            old_shares.slice(s![.., i]).to_owned(),
            random.slice(s![.., i]).to_owned(),
            random_n.slice(s![.., i]).to_owned(),
            transform,
            context,
            net,
        )));
    }

    let mut out_shares = Vec::with_capacity(N * NUM);
    for handle in handles {
        out_shares.extend_from_slice(handle.await.unwrap().view().to_slice().unwrap());
    }

    let exp_secrets = {
        let secrets = secrets
            .rows()
            .into_iter()
            .flat_map(|row| f_trans.apply(row))
            .collect();
        Array2::from_shape_vec((NUM, 2), secrets).unwrap()
    };

    let out_secrets = Array2::from_shape_vec(
        (NUM, 2),
        Array2::from_shape_vec((N, NUM), out_shares)
            .unwrap()
            .axis_iter(Axis(1))
            .flat_map(|s| pss_new.recon(s).unwrap())
            .collect(),
    )
    .unwrap();
    assert_eq!(out_secrets, exp_secrets);
}

#[tokio::test]
async fn randtrans() {
    let mut rng = StdRng::seed_from_u64(200);
    let proto_id = b"protocol".to_vec();

    let (pss, pss_n, contexts) = setup().await;

    let l: u32 = L.try_into().unwrap();
    let old_pos: Vec<Vec<GF<W>>> = vec![
        (200..(200 + l)).map(GF::from).collect(),
        (250..(250 + l)).map(GF::from).collect(),
        (225..(225 + l)).map(GF::from).collect(),
    ];
    let new_pos: Vec<Vec<GF<W>>> = vec![
        (300..(300 + l)).map(GF::from).collect(),
        (350..(350 + l)).map(GF::from).collect(),
        (325..(325 + l)).map(GF::from).collect(),
    ];
    let f_trans = vec![
        Combination::new((0..L).collect()),
        Combination::new(vec![2, 1, 0]),
        Combination::new(vec![2, 0, 2]),
    ];

    let rand_shares = rand_shares(NUM * (N + T), &pss, &mut rng);
    let zero_shares = shares(Array2::zeros((NUM * 2 * N, L)).view(), &pss_n, &mut rng);

    let mut handles = Vec::new();
    for (i, (net, context)) in contexts.into_iter().enumerate() {
        let transform = core::RandSharingTransform::new(&old_pos, &new_pos, &f_trans, &context);

        let mut transforms = Vec::with_capacity(NUM);
        for _ in 0..NUM {
            transforms.push(transform.clone());
        }

        handles.push(spawn(core::randtrans(
            proto_id.clone(),
            rand_shares.slice(s![.., i]).to_vec(),
            zero_shares.slice(s![.., i]).to_vec(),
            transforms,
            context,
            net,
        )));
    }

    let mut sharings = Array3::zeros((0, NUM, L));
    let mut sharings_n = Array3::zeros((0, NUM, L));
    for handle in handles {
        let (shares, shares_n) = handle.await.unwrap();
        sharings.push(Axis(0), shares.view()).unwrap();
        sharings_n.push(Axis(0), shares_n.view()).unwrap();
    }

    for i in 0..L {
        let orec_coeffs = PackedSharing::compute_recon_coeffs(
            (N - 1).try_into().unwrap(),
            N.try_into().unwrap(),
            &old_pos[i],
        );
        let nrec_coeffs = PackedSharing::compute_recon_coeffs(
            (T + L - 1).try_into().unwrap(),
            N.try_into().unwrap(),
            &new_pos[i],
        );

        for j in 0..NUM {
            let osec = PackedSharing::recon_using_coeffs(
                sharings_n.slice(s![.., j, i]),
                orec_coeffs.view(),
            );
            let nsec =
                PackedSharing::recon_using_coeffs(sharings.slice(s![.., j, i]), nrec_coeffs.view());
            debug_assert_eq!(nsec.to_vec(), f_trans[i].apply(ArrayView1::from(&osec)));
        }
    }
}

#[tokio::test]
async fn randtrans_incomplete_block() {
    let mut rng = StdRng::seed_from_u64(200);
    let proto_id = b"protocol".to_vec();

    let (pss, pss_n, contexts) = setup().await;

    let l: u32 = L.try_into().unwrap();
    let old_pos: Vec<Vec<GF<W>>> = vec![
        vec![GF::from(200)],
        (225..(225 + l)).map(GF::from).collect(),
    ];
    let new_pos: Vec<Vec<GF<W>>> = vec![
        vec![GF::from(300), GF::from(301), GF::from(302)],
        vec![GF::from(325)],
    ];
    let f_trans = vec![Combination::new(vec![0, 0, 0]), Combination::new(vec![1])];

    let rand_shares = rand_shares(NUM * (N + T), &pss, &mut rng);
    let zero_shares = shares(Array2::zeros((NUM * 2 * N, L)).view(), &pss_n, &mut rng);

    let mut handles = Vec::new();
    for (i, (net, context)) in contexts.into_iter().enumerate() {
        let transform = core::RandSharingTransform::new(&old_pos, &new_pos, &f_trans, &context);

        let mut transforms = Vec::with_capacity(NUM);
        for _ in 0..NUM {
            transforms.push(transform.clone());
        }

        handles.push(spawn(core::randtrans(
            proto_id.clone(),
            rand_shares.slice(s![.., i]).to_vec(),
            zero_shares.slice(s![.., i]).to_vec(),
            transforms,
            context,
            net,
        )));
    }

    let mut sharings = Array3::zeros((0, NUM, L));
    let mut sharings_n = Array3::zeros((0, NUM, L));
    for handle in handles {
        let (shares, shares_n) = handle.await.unwrap();
        sharings.push(Axis(0), shares.view()).unwrap();
        sharings_n.push(Axis(0), shares_n.view()).unwrap();
    }

    for i in 0..2 {
        let orec_coeffs = PackedSharing::compute_recon_coeffs(
            (N - 1).try_into().unwrap(),
            N.try_into().unwrap(),
            &old_pos[i],
        );
        let nrec_coeffs = PackedSharing::compute_recon_coeffs(
            (T + L - 1).try_into().unwrap(),
            N.try_into().unwrap(),
            &new_pos[i],
        );

        for j in 0..NUM {
            let osec = PackedSharing::recon_using_coeffs(
                sharings_n.slice(s![.., j, i]),
                orec_coeffs.view(),
            );
            let nsec =
                PackedSharing::recon_using_coeffs(sharings.slice(s![.., j, i]), nrec_coeffs.view());
            debug_assert_eq!(nsec.to_vec(), f_trans[i].apply(ArrayView1::from(&osec)));
        }
    }
}
