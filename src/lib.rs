pub mod galois;
pub mod sharing;
mod utils;

#[derive(Debug,PartialEq)]
pub enum ProtoErrorKind {
    MaliciousBehavior,
    Other(&'static str)
}
