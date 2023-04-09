use super::galois::{GFElement, GF};

/// Takes an inner product of two vectors defined implicitly by their iterators.
/// If the vectors are of unequal length, the longer one is truncated to the length of the shorter
/// one.
pub fn iprod<'a, I1, I2>(iter1: I1, iter2: I2, gf: &GF) -> GFElement
where
    I1: IntoIterator<Item = &'a GFElement>,
    I2: IntoIterator<Item = &'a GFElement>,
{
    iter1
        .into_iter()
        .zip(iter2.into_iter())
        .fold(gf.zero(), |acc, (&x, &y)| acc + x * y)
}

/// Computes the product of a matrix and a vector, and provides an iterator over the result.
/// If the number of columns of the matrix or the length of the vector are unequal, the answer is
/// truncated to the shorter one.
pub fn matrix_vector_prod<'a>(
    matrix: &'a [Vec<GFElement>],
    vector: &'a [GFElement],
    gf: &'a GF,
) -> impl Iterator<Item = GFElement> + 'a {
    matrix
        .iter()
        .map(move |row| iprod(row.iter(), vector.iter(), gf))
}
