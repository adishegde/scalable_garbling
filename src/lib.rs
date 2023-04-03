pub mod math;
pub mod protocol;
pub mod sharing;
mod utils;

pub use utils::block_on;
pub use utils::spawn;

#[derive(Debug, PartialEq)]
pub enum ProtoErrorKind {
    MaliciousBehavior,
    Other(&'static str),
}

/// Identifier for each participant.
pub type PartyID = u16;
