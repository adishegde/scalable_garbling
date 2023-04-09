use scalable_mpc::math;
use scalable_mpc::math::galois::{GFElement, GF};
use scalable_mpc::math::lagrange_coeffs;
use serial_test::serial;
use std::path::PathBuf;
use std::thread;

const GF_WIDTH: u8 = 18;

// Creating a from field is not thread safe and so the tests can't be run in parallel.
fn setup() -> GF {
    GF::new(GF_WIDTH).unwrap()
}

#[test]
#[serial]
fn add() {
    let gf = setup();
    let a = gf.get(5);
    let b = gf.get(3);
    let c = a + b;

    assert_eq!(c, gf.get(6));
}

#[test]
#[serial]
fn add_assign() {
    let gf = setup();
    let mut a = gf.get(5);
    let b = gf.get(3);
    a += b;

    assert_eq!(a, gf.get(6));
}

#[test]
#[serial]
fn mul() {
    let gf = setup();
    let a = gf.get(7);
    let b = gf.get(9);
    let c = a * b;

    assert_eq!(c, gf.get(63));
}

#[test]
#[serial]
fn mul_assign() {
    let gf = setup();
    let mut a = gf.get(7);
    let b = gf.get(9);
    a *= b;

    assert_eq!(a, gf.get(63));
}

#[test]
#[serial]
fn div() {
    let gf = setup();
    let a = gf.get(63);
    let b = gf.get(7);
    let c = a / b;

    assert_eq!(c, gf.get(9));
}

#[test]
#[serial]
fn div_assign() {
    let gf = setup();
    let mut a = gf.get(63);
    let b = gf.get(7);
    a /= b;

    assert_eq!(a, gf.get(9));
}

#[test]
#[serial]
fn get_range() {
    let gf = setup();
    let vals = gf.get_range(0..10);

    for (i, v) in vals.enumerate() {
        assert_eq!(gf.get(i.try_into().unwrap()), v);
    }
}

#[test]
#[serial]
fn ops_in_parallel() {
    let gf = setup();
    let one = gf.one();

    let prod = thread::scope(|s| {
        let h1 = s.spawn(|| {
            let beg = 1;
            let end = gf.order() / 2;
            (beg..end).fold(one, |acc, x| acc * gf.get(x))
        });

        let h2 = s.spawn(|| {
            let beg = gf.order() / 2;
            let end = gf.order();
            (beg..end).fold(one, |acc, x| acc * gf.get(x))
        });

        h1.join().unwrap() * h2.join().unwrap()
    });

    let exp = (1..2_u32.pow(GF_WIDTH as u32)).fold(one, |acc, x| acc * gf.get(x));
    assert_eq!(prod, exp);
}

#[test]
#[serial]
fn serialize_and_deserialize_element() {
    let gf = setup();
    let a = gf.get(53);
    let bytes = gf.serialize_element(&a);
    let b = gf.deserialize_element(&bytes);

    assert_eq!(bytes.len(), 3);
    assert_eq!(a, b);
}

#[test]
#[serial]
fn serialize_and_deserialize_vec() {
    let gf = setup();
    let a = vec![gf.get(53), gf.get(133), gf.get(23)];
    let bytes = gf.serialize_vec(&a);
    let b = gf.deserialize_vec(&bytes);

    assert_eq!(bytes.len(), 9);
    assert_eq!(a, b);
}

#[test]
#[serial]
fn lagrange_coeffs_on_same_evaluations() {
    let gf = setup();

    let pos: Vec<GFElement> = vec![37, 71, 97].iter().map(|&x| gf.get(x)).collect();
    let all_coeffs = lagrange_coeffs(&pos, &pos, &gf);

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

    let all_coeffs = lagrange_coeffs(&pos, &npos, &gf);

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

#[test]
#[serial]
fn binary_super_inv_matrix() {
    let gf = setup();
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/n16_t5.txt");

    let matrix = math::binary_super_inv_matrix(&path, &gf);

    for row in matrix {
        for val in row {
            assert!((val == gf.zero()) || (val == gf.one()));
        }
    }
}
