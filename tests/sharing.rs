use scalable_mpc::galois::GFElement;
use scalable_mpc::sharing;
use serial_test::serial;

const GF_WIDTH: u8 = 18;

// Creating a from field is not thread safe and so the tests can't be run in parallel.
fn setup() {
    GFElement::setup(GF_WIDTH).unwrap()
}

#[test]
#[serial]
fn lagrange_coeffs_on_same_evaluations() {
    setup();

    let pos: Vec<GFElement> = vec![37, 71, 97]
        .iter()
        .map(|&x| GFElement::from(x, GF_WIDTH))
        .collect();
    let all_coeffs = sharing::lagrange_coeffs(&pos, &pos);

    assert_eq!(all_coeffs.len(), pos.len());
    for coeffs in &all_coeffs {
        assert_eq!(coeffs.len(), pos.len());
    }

    let gf_one = GFElement::from(1, GF_WIDTH);
    let gf_zero = GFElement::from(0, GF_WIDTH);

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
    setup();
    let poly = |x: GFElement| {
        GFElement::from(5, GF_WIDTH) * x * x
            + GFElement::from(2, GF_WIDTH) * x
            + GFElement::from(7, GF_WIDTH)
    };

    let pos: Vec<GFElement> = vec![37, 71, 97]
        .iter()
        .map(|&x| GFElement::from(x, GF_WIDTH))
        .collect();
    let evals: Vec<GFElement> = pos.iter().map(|&x| poly(x)).collect();

    let npos: Vec<GFElement> = vec![71, 4]
        .iter()
        .map(|&x| GFElement::from(x, GF_WIDTH))
        .collect();
    let exp: Vec<GFElement> = vec![evals[1], GFElement::from(95, GF_WIDTH)];

    let all_coeffs = sharing::lagrange_coeffs(&pos, &npos);

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
                .fold(GFElement::from(0, GF_WIDTH), |acc, (&x, &y)| acc + (x * y))
        })
        .collect();

    println!(" exp: {:?}\n out: {:?}", exp, out);

    for (&e, &o) in exp.iter().zip(out.iter()) {
        assert_eq!(e, o);
    }
}
