use ndarray::ArrayView;
use rand::rngs::StdRng;
use rand::SeedableRng;
use scalable_mpc::math::galois::GF;
use scalable_mpc::sharing::PackedSharing;
use scalable_mpc::ProtoErrorKind;

const W: u8 = 18;

#[test]
fn share_and_recon() {
    GF::<W>::init().unwrap();
    let n = 20;
    let t = 5;
    let l = 3;

    let pss = PackedSharing::new(t + l - 1, n, &PackedSharing::default_pos(n, l));
    let mut rng = StdRng::seed_from_u64(200);

    let secrets: [GF<W>; 3] = [7, 3, 12].map(GF::from);
    let shares = pss.share(ArrayView::from(&secrets), &mut rng);

    assert_eq!(shares.len(), n as usize);

    let recon_secrets = pss.semihon_recon(ArrayView::from(&shares));

    assert_eq!(recon_secrets.len(), l as usize);
    assert_eq!(secrets[..], recon_secrets[..]);
}

#[test]
fn share_and_recon_linear_ops() {
    GF::<W>::init().unwrap();
    let n = 20;
    let t = 5;
    let l = 3;

    let pss = PackedSharing::new(t + l - 1, n, &PackedSharing::default_pos(n, l));
    let mut rng = StdRng::seed_from_u64(200);

    let secrets1: [GF<W>; 3] = [7, 3, 12].map(GF::from);
    let secrets2: [GF<W>; 3] = [4, 0, 100].map(GF::from);

    let shares1 = pss.share(ArrayView::from(&secrets1), &mut rng);
    let shares2 = pss.share(ArrayView::from(&secrets2), &mut rng);

    let shares_sum: Vec<_> = shares1.iter().zip(shares2).map(|(x, y)| x + y).collect();
    let recon_secrets = pss.semihon_recon(ArrayView::from(&shares_sum));

    let exp_secrets: Vec<_> = secrets1.iter().zip(secrets2).map(|(x, y)| x + y).collect();
    assert_eq!(exp_secrets, recon_secrets);
}

#[test]
fn share_and_recon_malicious() {
    GF::<W>::init().unwrap();
    let n = 20;
    let t = 5;
    let l = 3;

    let pss = PackedSharing::new(t + l - 1, n, &PackedSharing::default_pos(n, l));
    let mut rng = StdRng::seed_from_u64(200);

    let secrets: [GF<W>; 3] = [7, 3, 12].map(GF::from);
    let mut shares = pss.share(ArrayView::from(&secrets), &mut rng);

    shares[0] += GF::ONE;

    let resp = pss.recon(ArrayView::from(&shares));
    assert_eq!(resp, Err(ProtoErrorKind::MaliciousBehavior));
}

#[test]
fn share_and_recon_n() {
    GF::<W>::init().unwrap();
    let n = 20;
    let l = 3;

    let pss = PackedSharing::new(n - 1, n, &PackedSharing::default_pos(n, l));
    let mut rng = StdRng::seed_from_u64(200);

    let secrets: [GF<W>; 3] = [7, 3, 12].map(GF::from);
    let shares = pss.share(ArrayView::from(&secrets), &mut rng);

    assert_eq!(shares.len(), n as usize);

    let recon_secrets = pss.semihon_recon(ArrayView::from(&shares));

    assert_eq!(recon_secrets.len(), l as usize);
    assert_eq!(&secrets[..], &recon_secrets[..]);
}

#[test]
fn share_and_recon_n_linear_ops() {
    GF::<W>::init().unwrap();
    let n = 20;
    let l = 3;

    let pss = PackedSharing::new(n - 1, n, &PackedSharing::default_pos(n, l));
    let mut rng = StdRng::seed_from_u64(200);

    let secrets1: [GF<W>; 3] = [7, 3, 12].map(GF::from);
    let secrets2: [GF<W>; 3] = [4, 0, 100].map(GF::from);

    let shares1 = pss.share(ArrayView::from(&secrets1), &mut rng);
    let shares2 = pss.share(ArrayView::from(&secrets2), &mut rng);

    let shares_sum: Vec<_> = shares1.iter().zip(shares2).map(|(x, y)| x + y).collect();
    let recon_secrets = pss.semihon_recon(ArrayView::from(&shares_sum));

    let exp_secrets: Vec<_> = secrets1.iter().zip(secrets2).map(|(x, y)| x + y).collect();
    assert_eq!(&exp_secrets[..], &recon_secrets[..]);
}

#[test]
fn product_of_shares_with_recon_n() {
    GF::<W>::init().unwrap();
    let n = 20;
    let t = 5;
    let l = 3;

    let pos = PackedSharing::default_pos(n, l);
    let pss = PackedSharing::new(t + l - 1, n, &pos);
    let pss_n = PackedSharing::new(n - 1, n, &pos);
    let mut rng = StdRng::seed_from_u64(200);

    let secrets1: [GF<W>; 3] = [7, 3, 12].map(GF::from);
    let secrets2: [GF<W>; 3] = [4, 0, 100].map(GF::from);

    let shares1 = pss.share(ArrayView::from(&secrets1), &mut rng);
    let shares2 = pss.share(ArrayView::from(&secrets2), &mut rng);

    let shares_prod: Vec<_> = shares1.iter().zip(shares2).map(|(x, y)| x * y).collect();
    let recon_secrets = pss_n.semihon_recon(ArrayView::from(&shares_prod));

    let exp_secrets: Vec<_> = secrets1.iter().zip(secrets2).map(|(x, y)| x * y).collect();
    assert_eq!(exp_secrets, recon_secrets);
}

#[test]
fn const_to_share() {
    GF::<W>::init().unwrap();
    let n = 20;
    let l = 3;

    let pss = PackedSharing::new(l - 1, n, &PackedSharing::default_pos(n, l));
    let mut rng = StdRng::seed_from_u64(200);

    let const_vals: [GF<W>; 3] = [7, 3, 12].map(GF::from);
    let shares = pss.share(ArrayView::from(&const_vals), &mut rng);
    let recon_vals = pss.semihon_recon(ArrayView::from(&shares));

    assert_eq!(recon_vals, const_vals);
}

#[test]
fn share_and_recon_at_pos() {
    GF::<W>::init().unwrap();
    let n = 20;
    let t = 5;
    let l = 3;

    let pos: [GF<W>; 3] = [223, 33, 101].map(GF::from);
    let pss = PackedSharing::new(t + l - 1, n, &pos);
    let mut rng = StdRng::seed_from_u64(200);

    let secrets: [GF<W>; 3] = [7, 3, 12].map(GF::from);
    let shares = pss.share(ArrayView::from(&secrets), &mut rng);

    assert_eq!(shares.len(), n as usize);

    let recon_secrets = pss.semihon_recon(ArrayView::from(&shares));

    assert_eq!(recon_secrets.len(), l as usize);
    assert_eq!(secrets[..], recon_secrets[..]);
}

#[test]
fn product_of_shares_with_recon_n_at_pos() {
    GF::<W>::init().unwrap();
    let n = 20;
    let t = 5;
    let l = 3;

    let pos: [GF<W>; 3] = [223, 33, 101].map(GF::from);
    let pss = PackedSharing::new(t + l - 1, n, &pos);
    let pss_n = PackedSharing::new(n - 1, n, &pos);
    let mut rng = StdRng::seed_from_u64(200);

    let secrets1: [GF<W>; 3] = [7, 3, 12].map(GF::from);
    let secrets2: [GF<W>; 3] = [4, 0, 100].map(GF::from);

    let shares1 = pss.share(ArrayView::from(&secrets1), &mut rng);
    let shares2 = pss.share(ArrayView::from(&secrets2), &mut rng);

    let shares_prod: Vec<_> = shares1.iter().zip(shares2).map(|(x, y)| x * y).collect();
    let recon_secrets = pss_n.semihon_recon(ArrayView::from(&shares_prod));

    let exp_secrets: Vec<_> = secrets1.iter().zip(secrets2).map(|(x, y)| x * y).collect();
    assert_eq!(exp_secrets, recon_secrets);
}

#[test]
fn share_coeffs() {
    GF::<W>::init().unwrap();
    let n = 20;
    let t = 5;
    let l = 3;

    let pos: [GF<W>; 3] = [223, 33, 101].map(GF::from);
    let pss = PackedSharing::new(t + l - 1, n, &pos);
    let share_coeffs = PackedSharing::compute_share_coeffs(t + l - 1, n, &pos);
    let mut rng = StdRng::seed_from_u64(200);

    let mut shares: Vec<_> = (0..t).map(|_| GF::rand(&mut rng)).collect();
    let secrets: Vec<_> = (0..l).map(|_| GF::rand(&mut rng)).collect();
    let points: Vec<_> = secrets.iter().chain(shares.iter()).cloned().collect();
    shares.extend(share_coeffs.dot(&ArrayView::from(&points)));

    let recon_vals = pss.semihon_recon(ArrayView::from(&shares));
    assert_eq!(recon_vals, secrets);
}
