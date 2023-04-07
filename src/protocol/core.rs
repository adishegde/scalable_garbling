use super::network;
use super::network::{Recipient, SendMessage};
use super::{MPCContext, ProtocolID};
use crate::sharing::PackedShare;
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
    let leader = (hash(&id) % (context.n as u64)).try_into().unwrap();

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
