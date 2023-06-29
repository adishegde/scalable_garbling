use super::network;
use super::network::{Network, Recipient, SendMessage};
use super::{MPCContext, ProtocolID};
use crate::math::{galois::GF, lagrange_coeffs, Combination};
use crate::sharing::{PackedShare, PackedSharing};
use crate::PartyID;
use bincode::{deserialize, serialize};
use ndarray::parallel::prelude::*;
use ndarray::{concatenate, s, Array1, Array2, Array3, ArrayView1, Axis, Zip};
use rand;
use seahash::hash;

// Instead of having the same leader for every instance, we attempt to do some load balancing
// by having different parties as leaders.
// Computing the leader from the hash of the protocol ID avoids requiring external mechanisms
// to synchronize leader assignment across all parties.
// The hash need not be cryptographically secure, the only requirement is that it should be
// deterministic across all parties.
// The better the hash quality the better the load balancing will be.
fn leader_from_proto_id(id: &ProtocolID, num_parties: usize) -> PartyID {
    (hash(id) % (num_parties as u64)).try_into().unwrap()
}

pub async fn reduce_degree<const W: u8>(
    id: ProtocolID,
    vx: Array1<PackedShare<W>>,
    random: Array1<PackedShare<W>>,
    zero: Array1<PackedShare<W>>,
    leader: Option<PartyID>,
    context: MPCContext<W>,
    net: Network,
) -> Array1<PackedShare<W>> {
    let leader = match leader {
        Some(pid) => pid,
        None => leader_from_proto_id(&id, context.n),
    };
    let num = vx.len();

    let vrecon = Zip::from(&vx)
        .and(&random)
        .and(&zero)
        .par_map_collect(|x, r, z| x + r + z);
    net.send(SendMessage {
        to: Recipient::One(leader),
        proto_id: id.clone(),
        data: serialize(vrecon.as_slice().unwrap()).unwrap(),
    })
    .await;

    if context.id == leader {
        let shares_n: Vec<GF<W>> = network::message_from_each_party(id.clone(), &net, context.n)
            .await
            .into_par_iter()
            .flat_map(|d| deserialize::<Vec<GF<W>>>(&d).unwrap())
            .collect();
        let shares_n = Array2::from_shape_vec((context.n, num), shares_n).unwrap();

        let shares: Vec<_> = shares_n
            .axis_iter(Axis(1))
            .into_par_iter()
            .map_init(rand::thread_rng, |rng, s| {
                let secrets = context.pss_n.semihon_recon(s);
                context.pss.share(ArrayView1::from(&secrets), rng)
            })
            .flatten_iter()
            .collect();
        let shares = Array2::from_shape_vec((num, context.n), shares).unwrap();

        for i in 0..context.n {
            net.send(SendMessage {
                to: Recipient::One(i.try_into().unwrap()),
                proto_id: id.clone(),
                data: serialize(&shares.slice(s![.., i]).to_vec()).unwrap(),
            })
            .await;
        }
    }

    let mut shares: Array1<GF<W>> =
        Array1::from_vec(deserialize(&net.recv(id).await.data).unwrap());
    Zip::from(&mut shares).and(&random).par_for_each(|s, r| {
        *s -= r;
    });
    shares
}

pub async fn mult<const W: u8>(
    id: ProtocolID,
    mut vx: Array1<PackedShare<W>>,
    vy: Array1<PackedShare<W>>,
    random: Array1<PackedShare<W>>,
    zero: Array1<PackedShare<W>>,
    leader: Option<PartyID>,
    context: MPCContext<W>,
    net: Network,
) -> Array1<PackedShare<W>> {
    Zip::from(&mut vx).and(&vy).par_for_each(|x, y| {
        *x *= y;
    });
    reduce_degree(id, vx, random, zero, leader, context, net).await
}

#[derive(Clone)]
pub struct SharingTransform<const W: u8> {
    recon_n: Array2<GF<W>>,
    share: Array2<GF<W>>,
    npos_len: usize,
    f: Combination,
}

impl<const W: u8> SharingTransform<W> {
    pub fn new(
        old_pos: &[GF<W>],
        new_pos: &[GF<W>],
        f: Combination,
        context: &MPCContext<W>,
    ) -> Self {
        let n: u32 = context.n.try_into().unwrap();
        let t: u32 = context.t.try_into().unwrap();
        let l: u32 = context.l.try_into().unwrap();

        Self {
            recon_n: PackedSharing::compute_recon_coeffs(n - 1, n, old_pos),
            share: PackedSharing::compute_share_coeffs(t + l - 1, n, new_pos),
            npos_len: new_pos.len(),
            f,
        }
    }
}

pub async fn trans<const W: u8>(
    id: ProtocolID,
    vx: Array1<PackedShare<W>>,
    random: Array1<PackedShare<W>>,
    mut random_n: Array1<PackedShare<W>>,
    transform: SharingTransform<W>,
    context: MPCContext<W>,
    net: Network,
) -> Array1<PackedShare<W>> {
    let leader = leader_from_proto_id(&id, context.n);
    let num = vx.len();

    Zip::from(&mut random_n).and(&vx).par_for_each(|r, x| {
        *r += x;
    });

    net.send(SendMessage {
        to: Recipient::One(leader),
        proto_id: id.clone(),
        data: serialize(random_n.as_slice().unwrap()).unwrap(),
    })
    .await;

    if context.id == leader {
        let shares_n: Vec<_> = network::message_from_each_party(id.clone(), &net, context.n)
            .await
            .into_par_iter()
            .flat_map(|d| deserialize::<Vec<GF<W>>>(&d).unwrap())
            .collect();
        let shares_n = Array2::from_shape_vec((context.n, num), shares_n).unwrap();

        let secrets = transform.recon_n.dot(&shares_n);

        let shares = secrets
            .axis_iter(Axis(1))
            .into_par_iter()
            .map_init(rand::thread_rng, |rng, c| {
                let tf_secrets = transform.f.apply(c);
                PackedSharing::share_using_coeffs(
                    ArrayView1::from(&tf_secrets),
                    transform.share.view(),
                    transform.npos_len.try_into().unwrap(),
                    rng,
                )
            })
            .flatten_iter()
            .collect();
        let shares = Array2::from_shape_vec((num, context.n), shares).unwrap();

        for i in 0..context.n {
            net.send(SendMessage {
                to: Recipient::One(i.try_into().unwrap()),
                proto_id: id.clone(),
                data: serialize(&shares.slice(s![.., i]).to_vec()).unwrap(),
            })
            .await;
        }
    }

    let mut shares: Array1<GF<W>> =
        Array1::from_vec(deserialize(&net.recv(id).await.data).unwrap());
    Zip::from(&mut shares).and(&random).par_for_each(|s, r| {
        *s -= r;
    });
    shares
}

#[derive(Clone)]
pub struct RandSharingTransform<const W: u8> {
    share: Array2<GF<W>>,
    share_n: Array2<GF<W>>,
}

impl<const W: u8> RandSharingTransform<W> {
    pub fn new(
        old_pos: &[Vec<GF<W>>],
        new_pos: &[Vec<GF<W>>],
        f_trans: &[Combination],
        context: &MPCContext<W>,
    ) -> Self {
        debug_assert_eq!(old_pos.len(), new_pos.len());
        debug_assert_eq!(new_pos.len(), f_trans.len());
        debug_assert!(f_trans.len() <= context.l);

        let share_coeffs = |d: usize, n: usize, pos: &[GF<W>]| -> Array2<GF<W>> {
            let np = d + 1;
            let sh_pos = PackedSharing::share_pos(n.try_into().unwrap());
            let all_pos: Vec<_> = pos.iter().chain(sh_pos.iter()).cloned().collect();
            lagrange_coeffs(&all_pos[..np], &sh_pos)
        };

        let mut acc_coeffs = Array3::zeros((0, context.n, context.t + context.l));
        let mut acc_coeffs_n = Array3::zeros((0, context.n, context.n));

        // Compute coefficients to interpolate polynomials to old_pos and new_pos.
        for ((opos, npos), trans) in old_pos.iter().zip(new_pos.iter()).zip(f_trans.iter()) {
            debug_assert!(opos.len() <= context.l);
            debug_assert!(npos.len() <= context.l);
            debug_assert_eq!(trans.len(), npos.len());

            let coeffs_n = share_coeffs(context.n - 1, context.n, opos);
            acc_coeffs_n.push(Axis(0), coeffs_n.view()).unwrap();

            let coeffs = share_coeffs(context.t + context.l - 1, context.n, npos);
            let trans_mat = {
                let v: Vec<_> = (0..(context.t + context.l)).collect();
                let mut map = trans.apply(ArrayView1::from(&v[..opos.len()]));
                map.extend_from_slice(&v[npos.len()..]);

                let ext_trans = Combination::new(map);
                let mut mat: Array2<GF<W>> = Array2::eye(context.t + context.l);
                mat.columns_mut().into_iter().for_each(|col| {
                    let tcol = ext_trans.apply(col.view());
                    Zip::from(col).and(&tcol).for_each(|c, &tc| {
                        *c = tc;
                    });
                });
                mat
            };
            acc_coeffs
                .push(Axis(0), coeffs.dot(&trans_mat).view())
                .unwrap();
        }

        let share_l = share_coeffs(
            context.l,
            context.n,
            &PackedSharing::default_pos(
                context.n.try_into().unwrap(),
                context.l.try_into().unwrap(),
            ),
        )
        .slice(s![context.id as usize, ..])
        .to_owned();

        // Encode the coefficients as a share (i.e., as the points over a polynomial).
        let share = {
            let num = acc_coeffs.shape()[0];
            let f = |(r, c)| {
                share_l
                    .slice(s![..num])
                    .dot(&acc_coeffs.slice(s![.., r, c]).view())
            };

            Array2::from_shape_fn((context.n, context.t + context.l), f)
        };

        let share_n = {
            let num = acc_coeffs_n.shape()[0];
            let f = |(r, c)| {
                share_l
                    .slice(s![..num])
                    .dot(&acc_coeffs_n.slice(s![.., r, c]).view())
            };

            Array2::from_shape_fn((context.n, context.n), f)
        };

        let share = share;
        let share_n = share_n;

        Self { share, share_n }
    }
}

pub async fn randtrans<const W: u8>(
    id: ProtocolID,
    mut randoms: Vec<GF<W>>,
    mut zeros: Vec<GF<W>>,
    transform: Vec<RandSharingTransform<W>>,
    context: MPCContext<W>,
    net: Network,
) -> (Array2<GF<W>>, Array2<GF<W>>) {
    let num = transform.len();

    debug_assert_eq!(randoms.len(), num * (context.n + context.t));
    debug_assert_eq!(zeros.len(), 2 * context.n * num);

    let secrets = Array2::from_shape_vec(
        (context.l, num),
        randoms.split_off(randoms.len() - num * context.l),
    )
    .unwrap();
    let secrets_n = secrets.clone();

    let rands = Array2::from_shape_vec(
        (context.t, num),
        randoms.split_off(randoms.len() - num * context.t),
    )
    .unwrap();
    let rands_n = Array2::from_shape_vec((context.n - context.l, num), randoms).unwrap();

    let points = concatenate![Axis(0), secrets, rands];
    let points_n = concatenate![Axis(0), secrets_n, rands_n];

    let zeros_n =
        Array2::from_shape_vec((context.n, num), zeros.split_off(context.n * num)).unwrap();
    let zeros = Array2::from_shape_vec((context.n, num), zeros).unwrap();

    let shares: Vec<_> = transform
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, trans)| {
            (trans.share.dot(&points.slice(s![.., i])) + zeros.slice(s![.., i])).to_vec()
        })
        .collect();
    let shares = Array2::from_shape_vec((num, context.n), shares).unwrap();

    let shares_n: Vec<_> = transform
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, trans)| {
            (trans.share_n.dot(&points_n.slice(s![.., i])) + zeros_n.slice(s![.., i])).to_vec()
        })
        .collect();
    let shares_n = Array2::from_shape_vec((num, context.n), shares_n).unwrap();

    for i in 0..context.n {
        net.send(SendMessage {
            to: Recipient::One(i.try_into().unwrap()),
            proto_id: id.clone(),
            data: serialize(&(
                shares.slice(s![.., i]).to_vec(),
                shares_n.slice(s![.., i]).to_vec(),
            ))
            .unwrap(),
        })
        .await;
    }

    let (shares, shares_n): (Vec<_>, Vec<_>) =
        network::message_from_each_party(id, &net, context.n)
            .await
            .into_par_iter()
            .map(|d| {
                let (shares, shares_n): (Vec<GF<W>>, Vec<GF<W>>) = deserialize(&d).unwrap();
                (shares, shares_n)
            })
            .unzip();

    let shares = shares.into_par_iter().flat_map_iter(|s| s).collect();
    let shares = Array2::from_shape_vec((context.n, num), shares).unwrap();
    let shares_n = shares_n.into_par_iter().flat_map_iter(|s| s).collect();
    let shares_n = Array2::from_shape_vec((context.n, num), shares_n).unwrap();

    let shares = context.pss_n.recon_coeffs().dot(&shares).reversed_axes();
    let shares_n = context.pss_n.recon_coeffs().dot(&shares_n).reversed_axes();

    // Output shapes are (num, l).
    (shares, shares_n)
}
