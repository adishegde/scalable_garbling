use scalable_mpc::galois::{GFElement, GF};
use scalable_mpc::utils;
use serial_test::serial;

const GF_WIDTH: u8 = 18;

// Creating a from field is not thread safe and so the tests can't be run in parallel.
fn setup() -> GF {
    GF::new(GF_WIDTH).unwrap()
}

#[test]
#[serial]
fn lagrange_coeffs_on_same_evaluations() {
    let gf = setup();

    let pos: Vec<GFElement> = vec![37, 71, 97].iter().map(|&x| gf.get(x)).collect();
    let all_coeffs = utils::lagrange_coeffs(&pos, &pos, &gf);

    assert_eq!(all_coeffs.len(), pos.len());
    for coeffs in &all_coeffs {
        assert_eq!(coeffs.len(), pos.len());
    }

    let gf_one = gf.one();
    let gf_zero = gf.zero();

    for (i, coeffs) in all_coeffs.iter().enumerate() {
        for (j, &v) in coeffs.iter().enumerate() {
            if i == j {
                assert_eq!(v, gf_one);
            } else {
                assert_eq!(v, gf_zero);
            }
        }
    }
}

#[test]
#[serial]
fn lagrange_coeffs_on_diff_evaluations() {
    let gf = setup();
    let poly = |x: GFElement| gf.get(5) * x * x + gf.get(2) * x + gf.get(7);

    let pos: Vec<GFElement> = vec![37, 71, 97].iter().map(|&x| gf.get(x)).collect();
    let evals: Vec<GFElement> = pos.iter().map(|&x| poly(x)).collect();

    let npos: Vec<GFElement> = vec![71, 4].iter().map(|&x| gf.get(x)).collect();
    let exp: Vec<GFElement> = vec![evals[1], gf.get(95)];

    let all_coeffs = utils::lagrange_coeffs(&pos, &npos, &gf);

    assert_eq!(all_coeffs.len(), npos.len());
    for coeffs in &all_coeffs {
        assert_eq!(coeffs.len(), pos.len());
    }

    let out: Vec<GFElement> = all_coeffs
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .zip(evals.iter())
                .fold(gf.get(0), |acc, (&x, &y)| acc + (x * y))
        })
        .collect();

    println!(" exp: {:?}\n out: {:?}", exp, out);

    for (&e, &o) in exp.iter().zip(out.iter()) {
        assert_eq!(e, o);
    }
}
