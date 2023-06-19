mod bindings;
use num_traits::identities::{One, Zero};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::sync::{Once, OnceLock};

/// Galois field elements where the order of the field is 2^W.
///
/// Note that W is expected to be less than 30.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GF<const W: u8>(u32);

impl<const W: u8> GF<W> {
    /// Order/size of the field.
    pub const ORDER: u32 = 2_u32.pow(W as u32);

    /// Number of bytes when serialized.
    pub const NUM_BYTES: usize = ((W as usize) + 7) / 8;

    /// Additive identity.
    pub const ZERO: Self = Self(0);

    /// Multiplicative identity.
    pub const ONE: Self = Self(1);

    fn get_dist() -> &'static Uniform<u32> {
        static DIST: OnceLock<Uniform<u32>> = OnceLock::new();

        DIST.get_or_init(|| Uniform::new(0, 2_u32.pow(W.into())))
    }

    /// Initialize the field by pre-computing data required to carry out field operations.
    pub fn init() -> Result<(), &'static str> {
        static INIT: Once = Once::new();

        let mut flag = true;
        INIT.call_once(|| {
            let res = unsafe { bindings::galois_create_log_tables(W.into()) };
            if res != 0 {
                flag = false;
            }
        });

        if flag {
            Ok(())
        } else {
            Err("Could not initialize the field.")
        }
    }

    /// Sample an element uniformly at random from the field.
    pub fn rand<R: Rng>(rng: &mut R) -> Self {
        Self(Self::get_dist().sample(rng))
    }
}

impl<const W: u8> From<u32> for GF<W> {
    fn from(value: u32) -> Self {
        let res = Self(value);

        if value >= Self::ORDER {
            res * Self::ONE
        } else {
            res
        }
    }
}

impl<const W: u8> From<GF<W>> for u32 {
    fn from(value: GF<W>) -> u32 {
        value.0
    }
}

impl<const W: u8> MulAssign for GF<W> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = unsafe {
            bindings::galois_logtable_multiply(self.0 as i32, rhs.0 as i32, W.into()) as u32
        }
    }
}

impl<const W: u8> MulAssign<&GF<W>> for GF<W> {
    fn mul_assign(&mut self, rhs: &Self) {
        self.0 = unsafe {
            bindings::galois_logtable_multiply(self.0 as i32, rhs.0 as i32, W.into()) as u32
        }
    }
}

impl<const W: u8> Mul for GF<W> {
    type Output = Self;

    fn mul(mut self, other: Self) -> Self {
        self *= other;
        self
    }
}

impl<const W: u8> Mul<&GF<W>> for GF<W> {
    type Output = Self;

    fn mul(mut self, other: &Self) -> Self {
        self *= other;
        self
    }
}

impl<const W: u8> Mul for &GF<W> {
    type Output = GF<W>;

    fn mul(self, other: Self) -> GF<W> {
        let mut prod = GF(self.0);
        prod *= other;
        prod
    }
}

impl<const W: u8> Mul<GF<W>> for &GF<W> {
    type Output = GF<W>;

    fn mul(self, mut other: GF<W>) -> GF<W> {
        other *= self;
        other
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<const W: u8> AddAssign for GF<W> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<const W: u8> AddAssign<&GF<W>> for GF<W> {
    fn add_assign(&mut self, rhs: &Self) {
        self.0 ^= rhs.0;
    }
}

impl<const W: u8> Add for GF<W> {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl<const W: u8> Add<&GF<W>> for GF<W> {
    type Output = Self;

    fn add(mut self, other: &Self) -> Self {
        self += other;
        self
    }
}

impl<const W: u8> Add for &GF<W> {
    type Output = GF<W>;

    fn add(self, other: Self) -> GF<W> {
        let mut sum = GF(self.0);
        sum += other;
        sum
    }
}

impl<const W: u8> Add<GF<W>> for &GF<W> {
    type Output = GF<W>;

    fn add(self, mut other: GF<W>) -> GF<W> {
        other += self;
        other
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<const W: u8> SubAssign for GF<W> {
    fn sub_assign(&mut self, rhs: Self) {
        *self += rhs;
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<const W: u8> SubAssign<&GF<W>> for GF<W> {
    fn sub_assign(&mut self, rhs: &Self) {
        *self += rhs;
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<const W: u8> Sub for GF<W> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + other
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<const W: u8> Sub<&GF<W>> for GF<W> {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        self + other
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<const W: u8> Sub for &GF<W> {
    type Output = GF<W>;

    fn sub(self, other: Self) -> GF<W> {
        self + other
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<const W: u8> Sub<GF<W>> for &GF<W> {
    type Output = GF<W>;

    fn sub(self, other: GF<W>) -> GF<W> {
        self + other
    }
}

impl<const W: u8> DivAssign for GF<W> {
    fn div_assign(&mut self, rhs: Self) {
        self.0 = unsafe {
            bindings::galois_logtable_divide(self.0 as i32, rhs.0 as i32, W.into()) as u32
        };
    }
}

impl<const W: u8> DivAssign<&GF<W>> for GF<W> {
    fn div_assign(&mut self, rhs: &Self) {
        self.0 = unsafe {
            bindings::galois_logtable_divide(self.0 as i32, rhs.0 as i32, W.into()) as u32
        };
    }
}

impl<const W: u8> Div for GF<W> {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self {
        self /= rhs;
        self
    }
}

impl<const W: u8> Div<&GF<W>> for GF<W> {
    type Output = Self;

    fn div(mut self, rhs: &Self) -> Self {
        self /= rhs;
        self
    }
}

impl<const W: u8> Div for &GF<W> {
    type Output = GF<W>;

    fn div(self, rhs: Self) -> GF<W> {
        let mut res = GF(self.0);
        res /= rhs;
        res
    }
}

impl<const W: u8> Div<GF<W>> for &GF<W> {
    type Output = GF<W>;

    fn div(self, rhs: GF<W>) -> GF<W> {
        let mut res = GF(self.0);
        res /= rhs;
        res
    }
}

impl<const W: u8> Zero for GF<W> {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl<const W: u8> One for GF<W> {
    fn one() -> Self {
        Self::ONE
    }
}
