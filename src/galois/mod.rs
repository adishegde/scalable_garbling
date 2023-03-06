mod bindings;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

pub struct GF {
    width: u8,
    rand_dist: Uniform<u32>,
}

impl GF {
    // Setup field with specified width by pre-computing necessary data.
    pub fn new(width: u8) -> Result<GF, &'static str> {
        if width > 30 {
            return Err("Width is too large.");
        }

        let ret = unsafe { bindings::galois_create_log_tables(width.into()) };

        if ret != 0 {
            Err("Could not create log tables.")
        } else {
            Ok(GF {
                width,
                rand_dist: Uniform::new(0, 2_u32.pow(width.into())),
            })
        }
    }

    // Get field element with value `value`.
    pub fn get(&self, value: u32) -> GFElement {
        GFElement {
            value,
            width: self.width,
        }
    }

    // Get a random field element.
    pub fn rand<R: Rng>(&self, rng: &mut R) -> GFElement {
        GFElement {
            value: self.rand_dist.sample(rng),
            width: self.width,
        }
    }

    pub fn one(&self) -> GFElement {
        self.get(1)
    }

    pub fn zero(&self) -> GFElement {
        self.get(0)
    }

    pub fn order(&self) -> u32 {
        2_u32.pow(self.width as u32)
    }
}

// Represents a Galois field element.
// Serves as a thin wrapper around the C library.
// The implemented operations assume that input field elements have the same width.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GFElement {
    value: u32,
    width: u8,
}

impl AddAssign for GFElement {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: Self) {
        self.value ^= rhs.value;
    }
}

impl Add for GFElement {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, other: Self) -> Self {
        let mut sum = self;
        sum += other;
        sum
    }
}

impl SubAssign for GFElement {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn sub_assign(&mut self, rhs: Self) {
        self.add_assign(rhs)
    }
}

impl Sub for GFElement {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, other: Self) -> Self {
        let mut sum = self;
        sum += other;
        sum
    }
}

impl MulAssign for GFElement {
    fn mul_assign(&mut self, rhs: Self) {
        self.value = unsafe {
            bindings::galois_logtable_multiply(
                self.value as i32,
                rhs.value as i32,
                self.width.into(),
            ) as u32
        }
    }
}

impl Mul for GFElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut prod = self;
        prod *= other;
        prod
    }
}

impl DivAssign for GFElement {
    fn div_assign(&mut self, rhs: Self) {
        self.value = unsafe {
            bindings::galois_logtable_divide(self.value as i32, rhs.value as i32, self.width.into())
                as u32
        };
    }
}

impl Div for GFElement {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let mut res = self;
        res /= rhs;
        res
    }
}
