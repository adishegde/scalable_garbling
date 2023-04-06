use crate::math::galois::{GFElement, GFMatrix, GF};
use crate::math::lagrange_coeffs;
use crate::utils::iprod;
use crate::PartyID;
use crate::ProtoErrorKind;
use rand::Rng;

pub type PackedShare = GFElement;

/// Creating and manipulate packed shares for a particular corruption threshold and packing
/// parameter.
pub struct PackedSharing {
    // Corruption threshold.
    t: usize,
    // Packing parameter.
    l: usize,
    // Number of parties/shares. n >= 2t + 2l - 1.
    n: usize,
    // Coefficients to compute `t + l - 1` degree sharings from secrets.
    // Dimensions: `n - t` vectors each of length `t + l`.
    share_coeffs: GFMatrix,
    // Coefficients to compute secrets from `t + l - 1` degree sharings.
    // Dimensions: `n - t` vectors each of length `t + l`.
    recon_coeffs: GFMatrix,
    // Coefficients to compute a random `t + l - 1` degree sharing.
    // Dimensions: `n - t - l` vectors each of length `t + l`.
    rand_coeffs: GFMatrix,
    // Coefficients to compute shares over `n-1` degree polynomials.
    // Dimensions: `l` vectors each of length `n`.
    share_coeffs_n: GFMatrix,
    // Coefficients to compute secrets from `n-1` degree polynomial sharing.
    // Dimensions: `l` vectors each of length `n`.
    recon_coeffs_n: GFMatrix,
    // Coefficiens to compute shares over `l-1` degree polynomials.
    // Dimensions: `n` vectors each of length `l`.
    share_coeffs_l: GFMatrix,
}

impl PackedSharing {
    /// Creates a new packed sharing instance with a fixed threshold `t` and packing parameter `l`.
    pub fn new(n: u32, t: u32, l: u32, gf: &GF) -> Self {
        // Secrets correspond to evaluations at `[0, ..., l - 1]`.
        // Shares correspond to evaluations at `[l, ..., l + n - 1]`.
        // The sharing polynomial is defined by `[0, ..., l + t - 1]` which allows setting the
        // shares of the first `t` parties to a random value.
        let share_coeffs = {
            let cpos: Vec<_> = gf.get_range(0..(l + t)).collect();
            let npos: Vec<_> = gf.get_range((l + t)..(l + n)).collect();
            lagrange_coeffs(&cpos, &npos, gf)
        };

        let recon_coeffs = {
            let cpos: Vec<_> = gf.get_range((n - t)..(l + n)).collect();
            let npos: Vec<_> = gf.get_range(0..(n - t)).collect();
            lagrange_coeffs(&cpos, &npos, gf)
        };

        let rand_coeffs = {
            let cpos: Vec<_> = gf.get_range(l..(2 * l + t)).collect();
            let npos: Vec<_> = gf.get_range((2 * l + t)..(l + n)).collect();
            lagrange_coeffs(&cpos, &npos, gf)
        };

        let share_coeffs_n = {
            let cpos: Vec<_> = gf.get_range(0..n).collect();
            let npos: Vec<_> = gf.get_range(n..(l + n)).collect();
            lagrange_coeffs(&cpos, &npos, gf)
        };

        let recon_coeffs_n = {
            let cpos: Vec<_> = gf.get_range(l..(l + n)).collect();
            let npos: Vec<_> = gf.get_range(0..l).collect();
            lagrange_coeffs(&cpos, &npos, gf)
        };

        let share_coeffs_l = {
            let cpos: Vec<_> = gf.get_range(0..l).collect();
            let npos: Vec<_> = gf.get_range(l..(l + n)).collect();
            lagrange_coeffs(&cpos, &npos, gf)
        };

        PackedSharing {
            t: t as usize,
            l: l as usize,
            n: n as usize,
            share_coeffs,
            recon_coeffs,
            rand_coeffs,
            share_coeffs_n,
            recon_coeffs_n,
            share_coeffs_l,
        }
    }

    /// Secret shares over a `t + l - 1` degree polynomial.
    /// Returns an `n` length vector corresponding to the share of each party.
    ///
    /// If the number of secrets input is lesser than `l`, it pads the secrets to be of length `l`.
    pub fn share<R: Rng>(&self, secrets: &[GFElement], gf: &GF, rng: &mut R) -> Vec<PackedShare> {
        let nevals = self.t + self.l;

        let mut inp = Vec::with_capacity(nevals);
        let mut shares = Vec::with_capacity(self.n);

        // Only take up to first `l` secrets.
        for &x in secrets.iter().take(self.l) {
            inp.push(x);
        }

        // Remaining inputs are random.
        // This loop pads the input with random values to ensure there are `l` secrets as well as
        // samples random shares for the first `t` parties.
        while inp.len() < nevals {
            inp.push(gf.rand(rng));
        }

        shares.extend_from_slice(&inp[self.l..]);
        shares.extend(
            self.share_coeffs
                .iter()
                .map(|c| iprod(c.iter(), inp.iter(), gf)),
        );

        shares
    }

    /// Returns a random `t + l - 1` degree sharing.
    pub fn rand<R: Rng>(&self, gf: &GF, rng: &mut R) -> Vec<PackedShare> {
        let mut inp = Vec::with_capacity(self.t + self.l);
        let mut shares = Vec::with_capacity(self.n);

        inp.extend((0..(self.t + self.l)).map(|_| gf.rand(rng)));

        shares.extend_from_slice(&inp[..]);
        shares.extend(
            self.rand_coeffs
                .iter()
                .map(|c| iprod(c.iter(), inp.iter(), gf)),
        );

        shares
    }

    /// Reconstructs secrets from a `t + l - 1` degree sharing assuming that all shares are
    /// correct.
    /// Returns a vector of length `l` corresponding to the secrets.
    ///
    /// `shares` should be of length `n` but this is not checked within the method.
    pub fn semihon_recon(&self, shares: &[PackedShare], gf: &GF) -> Vec<GFElement> {
        self.recon_coeffs[..self.l]
            .iter()
            .map(|c| iprod(c.iter(), shares[(self.n - self.t - self.l)..].iter(), gf))
            .collect()
    }

    /// Reconstructs secrets from a `t + l - 1` degree sharing while ensuring that all input shares
    /// are consistent and lie on a polynoial of degree `t + l - 1`.
    /// Returns a vector of length `l` corresponding to the secrets.
    pub fn recon(&self, shares: &[PackedShare], gf: &GF) -> Result<Vec<GFElement>, ProtoErrorKind> {
        if shares.len() != self.n {
            return Err(ProtoErrorKind::Other("Expected one share from each party."));
        }

        let recon_vals: Vec<_> = self
            .recon_coeffs
            .iter()
            .map(|c| iprod(c.iter(), shares[(self.n - self.t - self.l)..].iter(), gf))
            .collect();

        for (i, &v) in recon_vals[self.l..].iter().enumerate() {
            if v != shares[i] {
                return Err(ProtoErrorKind::MaliciousBehavior);
            }
        }

        Ok(recon_vals[..self.l].to_vec())
    }

    /// Secret shares over a `n - 1 = 2t + 2l - 2` degree polynomial.
    /// Returns an `n = 2t + 2l - 1` length vector corresponding to the shares of each party.
    ///
    /// If the number of secrets input is lesser than `l`, it pads the secrets to be of length `l`.
    pub fn share_n<R: Rng>(&self, secrets: &[GFElement], gf: &GF, rng: &mut R) -> Vec<PackedShare> {
        let mut inp = Vec::with_capacity(self.n);
        let mut shares = Vec::with_capacity(self.n);

        // Only take up to first `l` secrets.
        for &x in secrets.iter().take(self.l) {
            inp.push(x);
        }

        // Remaining inputs are random.
        // This loop pads the input with random values to ensure there are `l` secrets as well as
        // samples random shares for the first `2t + l - 1` parties.
        while inp.len() < self.n {
            inp.push(gf.rand(rng));
        }

        shares.extend_from_slice(&inp[self.l..]);
        shares.extend(
            self.share_coeffs_n
                .iter()
                .map(|c| iprod(c.iter(), inp.iter(), gf)),
        );

        shares
    }

    /// Returns a random `n - 1 = 2t + 2l - 2` degree sharing.
    pub fn rand_n<R: Rng>(&self, gf: &GF, rng: &mut R) -> Vec<PackedShare> {
        (0..self.n).map(|_| gf.rand(rng)).collect()
    }

    /// Reconstructs secrets from a `n-1` degree sharing.
    /// Returns a vector of length `l` corresponding to the secrets.
    ///
    /// `shares` should be of length `n` but this is not checked within the method.
    pub fn recon_n(&self, shares: &[PackedShare], gf: &GF) -> Vec<GFElement> {
        self.recon_coeffs_n
            .iter()
            .map(|c| iprod(c.iter(), shares.iter(), gf))
            .collect()
    }

    /// Returns the i-th party's share for a `l-1` degree polynomial encoding of `l` secrets.
    ///
    /// `vals` should be of length `l` but this is not checked within the method.
    pub fn const_to_share(&self, vals: &[GFElement], i: PartyID, gf: &GF) -> PackedShare {
        iprod(self.share_coeffs_l[usize::from(i)].iter(), vals.iter(), gf)
    }
}
