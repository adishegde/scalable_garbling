pub mod math;
pub mod sharing;
mod utils;

#[derive(Debug, PartialEq)]
pub enum ProtoErrorKind {
    MaliciousBehavior,
    Other(&'static str),
}

/// Identifier for each participant.
pub type PartyID = u16;
