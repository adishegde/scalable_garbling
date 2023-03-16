use super::galois::{GFElement, GF};

pub fn iprod<'a, I>(iter1: I, iter2: I, gf: &GF) -> GFElement
where
    I: IntoIterator<Item = &'a GFElement>,
{
    iter1
        .into_iter()
        .zip(iter2.into_iter())
        .fold(gf.zero(), |acc, (&x, &y)| acc + x * y)
}
