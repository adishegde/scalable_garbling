use super::network;
use super::network::{Recipient, SendMessage};
use super::{MPCContext, ProtocolID};
use crate::math::galois::{GFElement, GFMatrix, GF};
use crate::math::utils;
use crate::sharing::{PackedShare, PackedSharing};
use crate::PartyID;
use rand;
use seahash::hash;

pub async fn reduce_degree(
    id: ProtocolID,
    x: PackedShare,
    random: PackedShare,
    zero: PackedShare,
    context: MPCContext,
) -> PackedShare {
    // Instead of having the same leader for every instance, we attempt to do some load balancing
    // by having different parties as leaders.
    // Computing the leader from the hash of the protocol ID avoids requiring external mechanisms
    // to synchronize leader assignment across all parties.
    // The hash need not be cryptographically secure, the only requirement is that it should be
    // deterministic across all parties.
    // The better the hash quality the better the load balancing will be.
    let leader: PartyID = (hash(&id) % (context.n as u64)).try_into().unwrap();

    let chan = context.net_builder.channel(&id).await;

    let x_recon = x + random + zero;
    chan.send(SendMessage {
        to: Recipient::One(leader),
        proto_id: id.clone(),
        data: context.gf.serialize_element(&x_recon),
    })
    .await;

    if context.id == leader {
        let shares: Vec<_> = network::message_from_each_party(chan.receiver(), context.n)
            .await
            .into_iter()
            .map(|d| context.gf.deserialize_element(&d))
            .collect();

        let secrets = context.pss.recon_n(&shares, context.gf.as_ref());
        let shares = {
            let mut rng = rand::thread_rng();
            context.pss.share(&secrets, context.gf.as_ref(), &mut rng)
        };

        for (i, share) in shares.into_iter().enumerate() {
            chan.send(SendMessage {
                to: Recipient::One(i as PartyID),
                proto_id: id.clone(),
                data: context.gf.serialize_element(&share),
            })
            .await;
        }
    }

    let share = context.gf.deserialize_element(&chan.recv().await.data);
    share - random
}

pub async fn mult(
    id: ProtocolID,
    x: PackedShare,
    y: PackedShare,
    random: PackedShare,
    zero: PackedShare,
    context: MPCContext,
) -> PackedShare {
    let product = x * y;
    reduce_degree(id, product, random, zero, context).await
}

/// Describes a sharing transformation.
/// Given a sharing with secrets at a particular position in the underyling polynomial, the
/// transformed sharing will have the secrets at a different set of positions.
pub struct SharingTransform {
    recon_n: GFMatrix,
    share: GFMatrix,
    f: Box<dyn Fn(&[PackedShare]) -> Vec<PackedShare> + Send>,
}

impl SharingTransform {
    pub fn new(
        old_pos: &[GFElement],
        new_pos: &[GFElement],
        f_trans: Box<dyn Fn(&[PackedShare]) -> Vec<PackedShare> + Send>,
        pss: &PackedSharing,
        gf: &GF,
    ) -> Self {
        Self {
            recon_n: pss.recon_coeffs_n(old_pos, gf),
            share: pss.share_coeffs(new_pos, gf),
            f: f_trans,
        }
    }
}

pub async fn trans(
    id: ProtocolID,
    x: PackedShare,
    random: PackedShare,
    random_n: PackedShare,
    transform: SharingTransform,
    context: MPCContext,
) -> PackedShare {
    // Attempt to load balance leader's work across all parties as in the degree reduction
    // protocol.
    let leader: PartyID = (hash(&id) % (context.n as u64)).try_into().unwrap();
    let chan = context.net_builder.channel(&id).await;

    let x_recon = x + random_n;
    chan.send(SendMessage {
        to: Recipient::One(leader),
        proto_id: id.clone(),
        data: context.gf.serialize_element(&x_recon),
    })
    .await;

    if context.id == leader {
        let shares: Vec<_> = network::message_from_each_party(chan.receiver(), context.n)
            .await
            .into_iter()
            .map(|d| context.gf.deserialize_element(&d))
            .collect();

        let secrets: Vec<_> =
            utils::matrix_vector_prod(&transform.recon_n, &shares, context.gf.as_ref()).collect();
        let secrets = transform.f.as_ref()(&secrets);
        let shares = {
            let mut rng = rand::thread_rng();
            let mut shares: Vec<_> = (0..context.t).map(|_| context.gf.rand(&mut rng)).collect();
            let points: Vec<_> = secrets.iter().chain(shares.iter()).copied().collect();
            shares.extend(utils::matrix_vector_prod(
                &transform.share,
                &points,
                context.gf.as_ref(),
            ));
            shares
        };

        for (i, share) in shares.into_iter().enumerate() {
            chan.send(SendMessage {
                to: Recipient::One(i as PartyID),
                proto_id: id.clone(),
                data: context.gf.serialize_element(&share),
            })
            .await;
        }
    }

    let share = context.gf.deserialize_element(&chan.recv().await.data);
    share - random
}

pub struct RandSharingTransform {
    share: GFMatrix,
    share_n: GFMatrix,
}

impl RandSharingTransform {
    pub fn new(
        id: PartyID,
        old_pos: &[Vec<GFElement>],
        new_pos: &[Vec<GFElement>],
        f_trans: &[Box<dyn Fn(&[PackedShare]) -> Vec<PackedShare>>],
        pss: &PackedSharing,
        gf: &GF,
    ) -> Self {
        let n: usize = pss.num_parties() as usize;
        let t: usize = pss.threshold() as usize;
        let l: usize = pss.packing_param() as usize;

        let mut coeffs_list = Vec::with_capacity(l);
        let mut coeffs_list_n = Vec::with_capacity(l);

        // Compute coefficients to interpolate polynomials to old_pos and new_pos.
        for ((opos, npos), trans) in old_pos.iter().zip(new_pos.iter()).zip(f_trans.iter()) {
            coeffs_list_n.push(pss.share_coeffs_n(opos, gf));

            // The matrix to compute the shares should also incorporate the underlying
            // transformation on the secrets.
            // We first compute the coefficient matrix, then multiply it with the matrix
            // corresponding to the linear transformation.
            let coeffs = pss.share_coeffs(npos, gf);
            let mut coeffs_trans = coeffs.clone();

            let mut inp = vec![gf.zero(); l];
            for c in 0..l {
                // inp should correspond to the c-th unit vector to capture the c-th secret's
                // contribution to the transformed secret.
                inp[c] = gf.one();
                if c != 0 {
                    inp[c - 1] = gf.zero();
                }

                // Multiply the coefficient matrix with the c-th column of the linear
                // transformation matrix.
                // The c-th column of the coefficient matrix is then replaced with the answer.
                for (r, v) in utils::matrix_vector_prod(&coeffs, &trans(&inp), gf).enumerate() {
                    coeffs_trans[r][c] = v;
                }
            }

            coeffs_list.push(coeffs_trans);
        }

        // Encode the coefficients as a share (i.e., as the points over a polynomial).

        let share = {
            let num_rows = n - t;
            let num_cols = t + l;

            let mut coeff_shares = Vec::with_capacity(num_rows);
            for r in 0..num_rows {
                let mut row_shares = Vec::with_capacity(num_cols);

                for c in 0..num_cols {
                    let mut vals = Vec::with_capacity(l);
                    for coeffs in coeffs_list.iter() {
                        vals.push(coeffs[r][c]);
                    }
                    // Only vectors of length l can be encoded.
                    // If fewer than l transformations are specified then the remaining are just
                    // zeroed out.
                    vals.resize(l, gf.zero());

                    row_shares.push(pss.const_to_share(&vals, id, gf));
                }
                coeff_shares.push(row_shares);
            }

            coeff_shares
        };

        let share_n = {
            let num_rows = l;
            let num_cols = n;

            let mut coeff_shares = Vec::with_capacity(num_rows);
            for r in 0..num_rows {
                let mut row_shares = Vec::with_capacity(num_cols);

                for c in 0..num_cols {
                    let mut vals = Vec::with_capacity(l);
                    for coeffs in coeffs_list_n.iter() {
                        vals.push(coeffs[r][c]);
                    }
                    // Only vectors of length l can be encoded.
                    // If fewer than l transformations are specified then the remaining are just
                    // zeroed out.
                    vals.resize(l, gf.zero());

                    row_shares.push(pss.const_to_share(&vals, id, gf));
                }
                coeff_shares.push(row_shares);
            }

            coeff_shares
        };

        Self { share, share_n }
    }
}

pub async fn randtrans(
    id: ProtocolID,
    randoms: Vec<GFElement>,
    zeros: Vec<GFElement>,
    transform: RandSharingTransform,
    context: MPCContext,
) -> Vec<(PackedShare, PackedShare)> {
    let chan = context.net_builder.channel(&id).await;

    let secrets = &randoms[..context.l];
    let randtape = &randoms[context.n..(context.n + context.t)];
    let zeros_n = &zeros[..context.n];
    let zeros = &zeros[context.n..];

    let shares_n: Vec<_> = {
        let points = &randoms[..context.n];
        let mut shares = randoms[context.l..context.n].to_vec();
        shares.extend(utils::matrix_vector_prod(
            &transform.share_n,
            points,
            context.gf.as_ref(),
        ));
        shares
            .into_iter()
            .zip(zeros_n.iter())
            .map(|(v, z)| v + *z)
            .collect()
    };

    let shares: Vec<_> = {
        let points: Vec<_> = secrets.iter().chain(randtape.iter()).copied().collect();
        let mut shares = randtape.to_vec();
        shares.extend(utils::matrix_vector_prod(
            &transform.share,
            &points,
            context.gf.as_ref(),
        ));
        shares
            .into_iter()
            .zip(zeros.iter())
            .map(|(v, z)| v + *z)
            .collect()
    };

    for (i, (share, share_n)) in shares.into_iter().zip(shares_n.into_iter()).enumerate() {
        chan.send(SendMessage {
            to: Recipient::One(i as PartyID),
            proto_id: id.clone(),
            data: context.gf.serialize_vec(&[share, share_n]),
        })
        .await;
    }

    let (shares, shares_n): (Vec<_>, Vec<_>) =
        network::message_from_each_party(chan.receiver(), context.n)
            .await
            .into_iter()
            .map(|d| {
                let vals = context.gf.deserialize_vec(&d);
                (vals[0], vals[1])
            })
            .unzip();

    context
        .pss
        .recon_n(&shares, context.gf.as_ref())
        .into_iter()
        .zip(
            context
                .pss
                .recon_n(&shares_n, context.gf.as_ref())
                .into_iter(),
        )
        .collect()
}
