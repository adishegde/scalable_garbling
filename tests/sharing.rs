use rand::rngs::StdRng;
use rand::SeedableRng;
use scalable_mpc::math::galois::GF;
use scalable_mpc::math::utils;
use scalable_mpc::sharing::PackedSharing;
use scalable_mpc::ProtoErrorKind;
use serial_test::serial;

const GF_WIDTH: u8 = 18;

// Creating a from field is not thread safe and so the tests can't be run in parallel.
fn setup() -> GF {
    GF::new(GF_WIDTH).unwrap()
}

#[test]
#[serial]
fn share_and_recon() {
    let gf = setup();
    let pss = PackedSharing::new(20, 5, 3, &gf);
    let mut rng = StdRng::seed_from_u64(200);

    let secrets = vec![gf.get(7), gf.get(3), gf.get(12)];
    let shares = pss.share(&secrets, &gf, &mut rng);

    assert_eq!(shares.len(), 20);

    let recon_secrets = pss.semihon_recon(&shares, &gf);

    assert_eq!(recon_secrets.len(), 3);
    assert_eq!(secrets, recon_secrets);
}

#[test]
#[serial]
fn share_and_recon_linear_ops() {
    let gf = setup();
    let pss = PackedSharing::new(20, 5, 3, &gf);
    let mut rng = StdRng::seed_from_u64(200);

    let secrets1 = vec![gf.get(7), gf.get(3), gf.get(12)];
    let secrets2 = vec![gf.get(4), gf.get(0), gf.get(100)];

    let shares1 = pss.share(&secrets1, &gf, &mut rng);
    let shares2 = pss.share(&secrets2, &gf, &mut rng);

    let shares_sum: Vec<_> = shares1.iter().zip(shares2).map(|(&x, y)| x + y).collect();
    let recon_secrets = pss.semihon_recon(&shares_sum, &gf);

    let exp_secrets: Vec<_> = secrets1.iter().zip(secrets2).map(|(&x, y)| x + y).collect();
    assert_eq!(exp_secrets, recon_secrets);
}

#[test]
#[serial]
fn share_and_recon_malicious() {
    let gf = setup();
    let pss = PackedSharing::new(20, 5, 3, &gf);
    let mut rng = StdRng::seed_from_u64(200);

    let secrets = vec![gf.get(7), gf.get(3), gf.get(12)];
    let mut shares = pss.share(&secrets, &gf, &mut rng);

    shares[0] += gf.one();

    let resp = pss.recon(&shares, &gf);
    assert_eq!(resp, Err(ProtoErrorKind::MaliciousBehavior));
}

#[test]
#[serial]
fn share_and_recon_n() {
    let gf = setup();
    let pss = PackedSharing::new(20, 5, 3, &gf);
    let mut rng = StdRng::seed_from_u64(200);

    let secrets = vec![gf.get(7), gf.get(3), gf.get(12)];
    let shares = pss.share_n(&secrets, &gf, &mut rng);

    assert_eq!(shares.len(), 20);

    let recon_secrets = pss.recon_n(&shares, &gf);

    assert_eq!(recon_secrets.len(), 3);
    assert_eq!(secrets, recon_secrets);
}

#[test]
#[serial]
fn share_and_recon_n_linear_ops() {
    let gf = setup();
    let pss = PackedSharing::new(20, 5, 3, &gf);
    let mut rng = StdRng::seed_from_u64(200);

    let secrets1 = vec![gf.get(7), gf.get(3), gf.get(12)];
    let secrets2 = vec![gf.get(4), gf.get(0), gf.get(100)];

    let shares1 = pss.share_n(&secrets1, &gf, &mut rng);
    let shares2 = pss.share_n(&secrets2, &gf, &mut rng);

    let shares_sum: Vec<_> = shares1.iter().zip(shares2).map(|(&x, y)| x + y).collect();
    let recon_secrets = pss.recon_n(&shares_sum, &gf);

    let exp_secrets: Vec<_> = secrets1.iter().zip(secrets2).map(|(&x, y)| x + y).collect();
    assert_eq!(exp_secrets, recon_secrets);
}

#[test]
#[serial]
fn product_of_shares_with_recon_n() {
    let gf = setup();
    let pss = PackedSharing::new(20, 5, 3, &gf);
    let mut rng = StdRng::seed_from_u64(200);

    let secrets1 = vec![gf.get(7), gf.get(3), gf.get(12)];
    let secrets2 = vec![gf.get(4), gf.get(0), gf.get(100)];

    let shares1 = pss.share(&secrets1, &gf, &mut rng);
    let shares2 = pss.share(&secrets2, &gf, &mut rng);

    let shares_prod: Vec<_> = shares1.iter().zip(shares2).map(|(&x, y)| x * y).collect();
    let recon_secrets = pss.recon_n(&shares_prod, &gf);

    let exp_secrets: Vec<_> = secrets1.iter().zip(secrets2).map(|(&x, y)| x * y).collect();
    assert_eq!(exp_secrets, recon_secrets);
}

#[test]
#[serial]
fn default_share_with_recon_n() {
    let gf = setup();
    let pss = PackedSharing::new(20, 5, 3, &gf);
    let mut rng = StdRng::seed_from_u64(200);

    let secrets = vec![gf.get(7), gf.get(3), gf.get(12)];
    let shares = pss.share(&secrets, &gf, &mut rng);

    let recon_secrets = pss.recon_n(&shares, &gf);
    assert_eq!(secrets, recon_secrets);
}

#[test]
#[serial]
fn const_to_share() {
    let gf = setup();
    let pss = PackedSharing::new(20, 5, 3, &gf);

    let const_vals = vec![gf.get(7), gf.get(3), gf.get(12)];

    let shares: Vec<_> = (0..20)
        .map(|i| pss.const_to_share(&const_vals, i, &gf))
        .collect();

    let recon_vals = pss.semihon_recon(&shares, &gf);

    assert_eq!(recon_vals, const_vals);
}

#[test]
#[serial]
fn share_coeffs() {
    let gf = setup();
    let pss = PackedSharing::new(20, 5, 3, &gf);

    let secrets = vec![gf.get(7), gf.get(3), gf.get(12)];
    let pos = vec![gf.get(100), gf.get(103), gf.get(109)];

    let share_coeffs = pss.share_coeffs(&pos, &gf);
    let recon_coeffs = pss.recon_coeffs_n(&pos, &gf);

    assert_eq!(share_coeffs.len(), 15);
    assert_eq!(recon_coeffs.len(), 3);

    let shares = {
        let mut shares: Vec<_> = gf.get_range(1..6).collect();
        let points: Vec<_> = secrets.iter().chain(shares.iter()).copied().collect();
        shares.extend(utils::matrix_vector_prod(&share_coeffs, &points, &gf));
        shares
    };
    let recon_vals: Vec<_> = utils::matrix_vector_prod(&recon_coeffs, &shares, &gf).collect();

    assert_eq!(recon_vals, secrets);
}

#[test]
#[serial]
fn share_coeffs_n() {
    let gf = setup();
    let pss = PackedSharing::new(20, 5, 3, &gf);

    let secrets = vec![gf.get(7), gf.get(3), gf.get(12)];
    let pos = vec![gf.get(100), gf.get(103), gf.get(109)];

    let share_coeffs = pss.share_coeffs_n(&pos, &gf);
    let recon_coeffs = pss.recon_coeffs_n(&pos, &gf);

    assert_eq!(share_coeffs.len(), 3);
    assert_eq!(recon_coeffs.len(), 3);

    let shares = {
        let mut shares: Vec<_> = gf.get_range(1..18).collect();
        let points: Vec<_> = secrets.iter().chain(shares.iter()).copied().collect();
        shares.extend(utils::matrix_vector_prod(&share_coeffs, &points, &gf));
        shares
    };
    let recon_vals: Vec<_> = utils::matrix_vector_prod(&recon_coeffs, &shares, &gf).collect();

    assert_eq!(recon_vals, secrets);
}

#[test]
#[serial]
fn const_coeffs() {
    let gf = setup();
    let pss = PackedSharing::new(20, 5, 3, &gf);

    let secrets = vec![gf.get(7), gf.get(3), gf.get(12)];
    let pos = vec![gf.get(100), gf.get(103), gf.get(109)];

    let shares: Vec<_> = (0..20)
        .map(|i| {
            let coeffs = pss.const_coeffs(&pos, i, &gf);
            utils::iprod(&coeffs, &secrets, &gf)
        })
        .collect();

    let recon_coeffs = pss.recon_coeffs_n(&pos, &gf);
    let recon_vals: Vec<_> = utils::matrix_vector_prod(&recon_coeffs, &shares, &gf).collect();

    assert_eq!(recon_vals, secrets);
}
