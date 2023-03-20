pub mod galois;
pub mod sharing;
pub mod utils;

#[derive(Debug,PartialEq)]
pub enum ProtoErrorKind {
    MaliciousBehavior,
    Other(&'static str)
}
