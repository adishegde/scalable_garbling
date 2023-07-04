use ndarray::{Array, ArrayView1, Zip};
use num_traits::{One, Zero};
use scalable_mpc::math;
use scalable_mpc::math::{galois::GF, Combination};
use std::path::PathBuf;

const W: u8 = 18;

#[test]
fn add() {
    GF::<W>::init().unwrap();

    let a = GF::from(5u32);
    let b = GF::from(3u32);
    let c = GF::<W>::from(6u32);

    assert_eq!(a + b, c);
    assert_eq!(&a + b, c);
    assert_eq!(&a + &b, c);
    assert_eq!(a + &b, c);
}

#[test]
fn add_assign() {
    GF::<W>::init().unwrap();
    let mut a = GF::from(5u32);
    let b = GF::from(3u32);
    let c = GF::<W>::from(6u32);

    a += b;
    assert_eq!(a, c);

    let mut a = GF::from(5u32);
    a += &b;
    assert_eq!(a, c);
}

#[test]
fn mul() {
    GF::<W>::init().unwrap();

    let a = GF::from(7u32);
    let b = GF::from(9u32);
    let c = GF::<W>::from(63u32);

    assert_eq!(a * b, c);
    assert_eq!(&a * b, c);
    assert_eq!(&a * &b, c);
    assert_eq!(a * &b, c);
}

#[test]
fn mul_assign() {
    GF::<W>::init().unwrap();
    let mut a = GF::from(7u32);
    let b = GF::from(9u32);
    let c = GF::<W>::from(63u32);

    a *= b;
    assert_eq!(a, c);

    let mut a = GF::from(7u32);
    a *= &b;
    assert_eq!(a, c);
}

#[test]
fn div() {
    GF::<W>::init().unwrap();

    let a = GF::from(63u32);
    let b = GF::from(7u32);
    let c = GF::<W>::from(9u32);

    assert_eq!(a / b, c);
    assert_eq!(&a / b, c);
    assert_eq!(&a / &b, c);
    assert_eq!(a / &b, c);
}

#[test]
fn div_assign() {
    GF::<W>::init().unwrap();
    let mut a = GF::from(63u32);
    let b = GF::from(7u32);
    let c = GF::<W>::from(9u32);

    a /= b;
    assert_eq!(a, c);

    let mut a = GF::from(63u32);
    a /= &b;
    assert_eq!(a, c);
}

#[test]
fn lagrange_coeffs_on_same_evaluations() {
    GF::<W>::init().unwrap();

    let pos: Vec<GF<W>> = vec![37u32, 71u32, 97u32]
        .into_iter()
        .map(GF::from)
        .collect();
    let all_coeffs = math::lagrange_coeffs(&pos, &pos);

    assert_eq!(all_coeffs.shape(), &[pos.len(), pos.len()]);
    assert_eq!(all_coeffs, Array::eye(pos.len()));
}

#[test]
fn lagrange_coeffs_on_diff_evaluations() {
    GF::<W>::init().unwrap();

    let poly = |x| GF::from(5u32) * x * x + GF::from(2u32) * x + GF::from(7u32);

    let pos: Vec<GF<W>> = vec![37u32, 71u32, 97u32]
        .into_iter()
        .map(GF::from)
        .collect();
    let evals = Array::from_vec(pos.iter().map(poly).collect());

    let npos: Vec<GF<W>> = vec![71u32, 4u32].into_iter().map(GF::from).collect();
    let exp = Array::from_vec(npos.iter().map(poly).collect());

    let coeffs = math::lagrange_coeffs(&pos, &npos);

    assert_eq!(coeffs.shape(), &[npos.len(), pos.len()]);

    let out = coeffs.dot(&evals);

    assert_eq!(exp, out);
}

#[test]
fn binary_super_inv_matrix() {
    GF::<W>::init().unwrap();

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/n16_t5.txt");

    let matrix = math::binary_super_inv_matrix::<W>(&path);

    Zip::from(&matrix).for_each(|x| {
        assert!(x.is_zero() || x.is_one());
    });
}

#[test]
fn super_invertible_matrix() {
    GF::<W>::init().unwrap();
    let mat = math::super_inv_matrix::<W>(10, 7);
    assert_eq!(mat.shape(), &[7, 10]);
}

#[test]
fn reed_solomon_generator_matrix() {
    GF::<W>::init().unwrap();
    let mat = math::rs_gen_mat::<W>(12, 40);
    assert_eq!(mat.shape(), &[40, 12]);
}

#[test]
fn combination() {
    let comb = {
        let inp: [u32; 3] = [101, 3, 23];
        let out: [u32; 5] = [23, 3, 101, 3, 23];
        Combination::from_instance(&inp, &out)
    };

    let inp = [-2, 99, 1];
    let exp = [1, 99, -2, 99, 1];
    let out = comb.apply(ArrayView1::from(&inp));

    assert_eq!(out, exp);
}

#[test]
fn serialize() {
    GF::<W>::init().unwrap();

    let a: GF<W> = GF::from(1242u32);
    let bytes_a = bincode::serialize(&a).unwrap();

    assert_eq!(bytes_a.len(), GF::<W>::NUM_BYTES);
}

#[test]
fn deserialize() {
    GF::<W>::init().unwrap();

    let a: GF<W> = GF::from(1242u32);
    let bytes_a = bincode::serialize(&a).unwrap();
    let b: GF<W> = bincode::deserialize(&bytes_a).unwrap();

    assert_eq!(a, b);
}
