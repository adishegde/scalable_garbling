use crate::circuit::WireID;
use crate::math::galois::GF;
use crate::math::lagrange_coeffs;
use crate::ProtoErrorKind;
use ndarray::{s, Array2, ArrayView, ArrayView1, ArrayView2};
use rand::Rng;

pub type PackedShare<const W: u8> = GF<W>;

pub struct PackedSharing<const W: u8> {
    n: usize,
    np: usize,
    l: usize,
    share_coeffs: Array2<GF<W>>,
    recon_coeffs: Array2<GF<W>>,
    rand_coeffs: Array2<GF<W>>,
}

impl<const W: u8> PackedSharing<W> {
    pub fn default_pos(n: u32, l: u32) -> Vec<GF<W>> {
        (n..(n + l)).map(GF::from).collect()
    }

    pub fn share_pos(n: u32) -> Vec<GF<W>> {
        (0..n).map(GF::from).collect()
    }

    pub fn wire_to_pos<'a, I>(n: u32, l: u32, iter: I) -> impl Iterator<Item = GF<W>> + 'a
    where
        I: IntoIterator<Item = WireID> + 'a,
    {
        let offset = n + l;
        iter.into_iter().map(move |x| GF::from(x + offset))
    }

    pub fn compute_share_coeffs(d: u32, n: u32, pos: &[GF<W>]) -> Array2<GF<W>> {
        let np = (d + 1) as usize;
        let sh_pos = Self::share_pos(n);
        let all_pos: Vec<_> = pos.iter().chain(sh_pos.iter()).cloned().collect();
        lagrange_coeffs(&all_pos[..np], &all_pos[np..])
    }

    pub fn share_using_coeffs<R: Rng>(
        secrets: ArrayView1<GF<W>>,
        coeffs: ArrayView2<GF<W>>,
        l: u32,
        rng: &mut R,
    ) -> Vec<PackedShare<W>> {
        let np = coeffs.shape()[1];
        let l = l as usize;

        let mut shares: Vec<_> = (l..np).map(|_| GF::rand(rng)).collect();
        let points: Vec<_> = secrets
            .iter()
            .cloned()
            .chain(std::iter::from_fn(|| Some(GF::rand(rng))))
            .take(l)
            .chain(shares.iter().cloned())
            .collect();

        shares.extend(coeffs.dot(&ArrayView::from(&points)));
        shares
    }

    pub fn compute_recon_coeffs(d: u32, n: u32, pos: &[GF<W>]) -> Array2<GF<W>> {
        let sh_pos = Self::share_pos(n);

        let n = n as usize;
        let l = pos.len();
        let np = (d + 1) as usize;

        let all_pos: Vec<_> = pos.iter().chain(sh_pos.iter()).cloned().collect();

        lagrange_coeffs(&all_pos[(l + n - np)..], &all_pos[..l])
    }

    pub fn recon_using_coeffs(
        shares: ArrayView1<GF<W>>,
        coeffs: ArrayView2<GF<W>>,
    ) -> Vec<PackedShare<W>> {
        let np = coeffs.shape()[1];
        let n = shares.shape()[0];
        coeffs.dot(&shares.slice(s![(n - np)..])).to_vec()
    }

    pub fn new(d: u32, n: u32, pos: &[GF<W>]) -> Self {
        let sh_pos = Self::share_pos(n);
        let np = d + 1;

        debug_assert!(np <= n);
        debug_assert_eq!(
            pos.iter()
                .map(|&x| if u32::from(x) < n { 1 } else { 0 })
                .sum::<usize>(),
            0
        );

        let l = pos.len();
        let n = n as usize;
        let np = np as usize;

        // n + l length vector consisiting of positions of secret concatenated with positions of
        // shares.
        let all_pos: Vec<_> = pos.iter().chain(sh_pos.iter()).cloned().collect();

        // Polynomial defined by l secrets and first np - l shares.
        let share_coeffs = lagrange_coeffs(&all_pos[..np], &all_pos[np..]);

        // Using the last np shares reconstruct secret and remaining shares.
        // Remaining shares are used to check for malicious behavior.
        let recon_coeffs = lagrange_coeffs(&all_pos[(l + n - np)..], &all_pos[..(l + n - np)]);

        // Random polynomial can be defined by using first np shares.
        let rand_coeffs = lagrange_coeffs(&sh_pos[..np], &sh_pos[np..]);

        Self {
            n,
            np,
            l,
            share_coeffs,
            recon_coeffs,
            rand_coeffs,
        }
    }

    pub fn share<R: Rng>(&self, secrets: ArrayView1<GF<W>>, rng: &mut R) -> Vec<PackedShare<W>> {
        Self::share_using_coeffs(
            secrets,
            self.share_coeffs.view(),
            self.l.try_into().unwrap(),
            rng,
        )
    }

    pub fn rand<R: Rng>(&self, rng: &mut R) -> Vec<PackedShare<W>> {
        let mut shares: Vec<_> = (0..self.np).map(|_| GF::rand(rng)).collect();
        shares.extend(self.rand_coeffs.dot(&ArrayView::from(&shares)));
        shares
    }

    pub fn semihon_recon(&self, shares: ArrayView1<GF<W>>) -> Vec<GF<W>> {
        debug_assert_eq!(shares.len(), self.n);

        self.recon_coeffs
            .slice(s![..self.l, ..])
            .dot(&shares.slice(s![(self.n - self.np)..]))
            .to_vec()
    }

    pub fn recon(&self, shares: ArrayView1<GF<W>>) -> Result<Vec<GF<W>>, ProtoErrorKind> {
        if shares.len() != self.n {
            return Err(ProtoErrorKind::Other("Expected one share from each party."));
        }

        let recon_vals = self
            .recon_coeffs
            .dot(&shares.slice(s![(self.n - self.np)..]))
            .to_vec();

        for (i, &v) in recon_vals[self.l..].iter().enumerate() {
            if v != shares[i] {
                return Err(ProtoErrorKind::MaliciousBehavior);
            }
        }

        Ok(recon_vals[..self.l].to_vec())
    }

    pub fn recon_coeffs(&self) -> ArrayView2<GF<W>> {
        self.recon_coeffs.slice(s![..self.l, ..])
    }

    pub fn num_parties(&self) -> u32 {
        self.n.try_into().unwrap()
    }

    pub fn num_secrets(&self) -> u32 {
        self.l.try_into().unwrap()
    }

    pub fn degree(&self) -> u32 {
        (self.np - 1).try_into().unwrap()
    }
}
