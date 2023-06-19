use std::time::Instant;

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

pub struct Stats {
    time: Instant,
    comm: Vec<(u64, u64)>,
}

impl Stats {
    pub async fn now(stats: &protocol::network::Stats, n: u32) -> Self {
        let mut comm = Vec::with_capacity(n as usize);
        for i in 0..n {
            comm.push(stats.party(i.try_into().unwrap()).await);
        }

        Self {
            time: Instant::now(),
            comm,
        }
    }

    pub async fn elapsed(&self, stats: &protocol::network::Stats) -> (u64, Vec<(u64, u64)>) {
        let time: u64 = self.time.elapsed().as_millis().try_into().unwrap();
        let mut comm = Vec::with_capacity(self.comm.len());
        for i in 0..self.comm.len() {
            let pcom = stats.party(i.try_into().unwrap()).await;
            comm.push((pcom.0 - self.comm[i].0, pcom.1 - self.comm[i].1));
        }

        (time, comm)
    }
}
