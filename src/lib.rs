use async_global_executor;
use smol::prelude::Future;
use smol::Task;

pub mod circuit;
pub mod math;
pub mod protocol;
pub mod sharing;

#[derive(Debug, PartialEq)]
pub enum ProtoErrorKind {
    MaliciousBehavior,
    Other(&'static str),
}

/// Identifier for each participant.
pub type PartyID = u16;

// Allows easily switching global executor.
pub fn spawn<F, T>(future: F) -> Task<T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    async_global_executor::spawn(future)
}

pub fn block_on<F, T>(future: F) -> T
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    async_global_executor::block_on(future)
}
