use scalable_mpc::galois::GFElement;
use serial_test::serial;
use std::thread;

const GF_WIDTH: u8 = 18;

// Creating a from field is not thread safe and so the tests can't be run in parallel.
fn setup() {
    GFElement::setup(GF_WIDTH).unwrap()
}

#[test]
#[serial]
fn add() {
    setup();
    let a = GFElement::from(5, GF_WIDTH);
    let b = GFElement::from(3, GF_WIDTH);
    let c = a + b;

    assert_eq!(c, GFElement::from(6, GF_WIDTH));
}

#[test]
#[serial]
fn add_assign() {
    setup();
    let mut a = GFElement::from(5, GF_WIDTH);
    let b = GFElement::from(3, GF_WIDTH);
    a += b;

    assert_eq!(a, GFElement::from(6, GF_WIDTH));
}

#[test]
#[serial]
fn mul() {
    setup();
    let a = GFElement::from(7, GF_WIDTH);
    let b = GFElement::from(9, GF_WIDTH);
    let c = a * b;

    assert_eq!(c, GFElement::from(63, GF_WIDTH));
}

#[test]
#[serial]
fn mul_assign() {
    setup();
    let mut a = GFElement::from(7, GF_WIDTH);
    let b = GFElement::from(9, GF_WIDTH);
    a *= b;

    assert_eq!(a, GFElement::from(63, GF_WIDTH));
}

#[test]
#[serial]
fn div() {
    setup();
    let a = GFElement::from(63, GF_WIDTH);
    let b = GFElement::from(7, GF_WIDTH);
    let c = a / b;

    assert_eq!(c, GFElement::from(9, GF_WIDTH));
}

#[test]
#[serial]
fn div_assign() {
    setup();
    let mut a = GFElement::from(63, GF_WIDTH);
    let b = GFElement::from(7, GF_WIDTH);
    a /= b;

    assert_eq!(a, GFElement::from(9, GF_WIDTH));
}

#[test]
#[serial]
fn ops_in_parallel() {
    setup();

    let one = GFElement::from(1, GF_WIDTH);

    let h1 = thread::spawn(move || {
        let beg = 1;
        let end = 2_i32.pow((GF_WIDTH as u32) - 1);
        (beg..end).fold(one, |acc, x| acc * GFElement::from(x, GF_WIDTH))
    });

    let h2 = thread::spawn(move || {
        let beg = 2_i32.pow((GF_WIDTH as u32) - 1);
        let end = 2_i32.pow(GF_WIDTH as u32);
        (beg..end).fold(one, |acc, x| acc * GFElement::from(x, GF_WIDTH))
    });

    let prod = h1.join().unwrap() * h2.join().unwrap();

    let exp =
        (1..2_i32.pow(GF_WIDTH as u32)).fold(one, |acc, x| acc * GFElement::from(x, GF_WIDTH));
    assert_eq!(prod, exp);
}
