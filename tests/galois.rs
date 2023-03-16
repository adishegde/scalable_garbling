use scalable_mpc::galois::GF;
use serial_test::serial;
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
