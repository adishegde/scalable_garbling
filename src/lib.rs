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
