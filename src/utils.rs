use super::math::galois::{GFElement, GF};
use async_global_executor;
use smol::prelude::Future;
use smol::Task;

pub fn iprod<'a, I>(iter1: I, iter2: I, gf: &GF) -> GFElement
where
    I: IntoIterator<Item = &'a GFElement>,
{
    iter1
        .into_iter()
        .zip(iter2.into_iter())
        .fold(gf.zero(), |acc, (&x, &y)| acc + x * y)
}

// Allows easily switching global executor.
pub fn spawn<F, T>(future: F) -> Task<T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    async_global_executor::spawn(future)
}

pub fn block_on<F: Future<Output = T>, T>(future: F) -> T {
    async_global_executor::block_on(future)
}
