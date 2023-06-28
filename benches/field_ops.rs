use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use scalable_mpc::math::galois::GF;

const W: u8 = 18;

fn rand_elements() -> (GF<W>, GF<W>) {
    let mut rng = rand::thread_rng();
    (GF::rand(&mut rng), GF::rand(&mut rng))
}

pub fn criterion_benchmark(c: &mut Criterion) {
    GF::<W>::init().unwrap();

    let mut group = c.benchmark_group("field_ops");

    group.bench_function("add", |b| {
        b.iter_batched(rand_elements, |(x, y)| x + y, BatchSize::SmallInput)
    });

    group.bench_function("mult", |b| {
        b.iter_batched(rand_elements, |(x, y)| x * y, BatchSize::SmallInput)
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
