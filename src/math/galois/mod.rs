mod bindings;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Galois field with fixed order.
pub struct GF {
    width: u8,
    num_bytes: usize,
    rand_dist: Uniform<u32>,
}

impl GF {
    /// Creates a Galois field of order `2^{width}`.
    /// Width is required to be at most 30.
    pub fn new(width: u8) -> Result<GF, &'static str> {
        if width > 30 {
            return Err("Width is too large.");
        }

        let ret = unsafe { bindings::galois_create_log_tables(width.into()) };
        let num_bytes = (width as f64 / 8.0).ceil() as usize;

        if ret != 0 {
            Err("Could not create log tables.")
        } else {
            Ok(GF {
                width,
                num_bytes,
                rand_dist: Uniform::new(0, 2_u32.pow(width.into())),
            })
        }
    }

    /// Creates field element with canonical representation `value`.
    pub fn get(&self, value: u32) -> GFElement {
        GFElement {
            value,
            width: self.width,
        }
    }

    /// Transforms an iterator over values to iterator over corresponding field elements.
    pub fn get_range<'a, I>(&'a self, iter: I) -> impl Iterator<Item = GFElement> + 'a
    where
        I: IntoIterator<Item = u32> + 'a,
    {
        iter.into_iter().map(|x| self.get(x))
    }

    /// Creates a random field element.
    pub fn rand<R: Rng>(&self, rng: &mut R) -> GFElement {
        GFElement {
            value: self.rand_dist.sample(rng),
            width: self.width,
        }
    }

    /// Returns the multiplicative identity.
    pub fn one(&self) -> GFElement {
        self.get(1)
    }

    /// Returns the additive identity.
    pub fn zero(&self) -> GFElement {
        self.get(0)
    }

    /// Returns the order of the field.
    pub fn order(&self) -> u32 {
        2_u32.pow(self.width as u32)
    }

    /// Serialize a field element.
    pub fn serialize_element(&self, element: &GFElement) -> Vec<u8> {
        element.value.to_le_bytes()[..self.num_bytes].to_vec()
    }

    /// Deserialize a field element.
    pub fn deserialize_element(&self, bytes: &[u8]) -> GFElement {
        let mut le_bytes = [0, 0, 0, 0];
        for i in 0..bytes.len() {
            le_bytes[i] = bytes[i];
        }

        GFElement {
            value: u32::from_le_bytes(le_bytes),
            width: self.width,
        }
    }

    /// Serialize a slice of elements into bytes.
    pub fn serialize_vec(&self, elements: &[GFElement]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.num_bytes * elements.len());
        for element in elements {
            bytes.extend_from_slice(&element.value.to_le_bytes()[..self.num_bytes]);
        }

        bytes
    }

    /// Deserialize a slice of elements into bytes.
    pub fn deserialize_vec(&self, bytes: &[u8]) -> Vec<GFElement> {
        let num = bytes.len() / self.num_bytes;
        let mut elements = Vec::with_capacity(num);

        for le_chunk in bytes.chunks(self.num_bytes) {
            let mut le_bytes = [0, 0, 0, 0];
            for (i, &v) in le_chunk.iter().enumerate() {
                le_bytes[i] = v;
            }

            elements.push(GFElement {
                value: u32::from_le_bytes(le_bytes),
                width: self.width,
            });
        }

        elements
    }
}

/// Represents a Galois field element.
///
/// These are constructed using the respective galois field instances.
/// The implemented operations assume that input field elements belong to the same field.
/// The behaviour is undefined when inputs belong to different fields.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

/// Matrix over a Galois field.
pub type GFMatrix = Vec<Vec<GFElement>>;
