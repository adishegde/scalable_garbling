mod bindings;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

// Represents a Galois field element.
// Serves as a thin wrapper around the C library.
// The implemented operations assume that input field elements have the same width.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GFElement {
    value: i32,
    width: u8
}

impl GFElement {
    // Setup field with specified width by pre-computing necessary data.
    pub fn setup(width: u8) -> Result<(), &'static str> {
        if width > 30 {
            return Err("Width is too large.");
        }

        let ret = unsafe { bindings::galois_create_log_tables(width.into()) };

        if ret != 0 {
            Err("Could not create log tables.")
        } else {
            Ok(())
        }
    }

    pub fn get_field(&self) -> u8 {
        return self.width;
    }

    pub fn from(value: i32, width: u8) -> Self {
        GFElement { value, width }
    }
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
            bindings::galois_logtable_multiply(self.value, rhs.value, self.width.into())
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
            bindings::galois_logtable_divide(self.value, rhs.value, self.width.into())
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
