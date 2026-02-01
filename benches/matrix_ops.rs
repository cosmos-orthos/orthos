use criterion::{black_box, criterion_group, criterion_main, Criterion};
use numerix::matrix::Matrix;
use numerix::ops;

fn benchmark_matmul(c: &mut Criterion) {
    let a = Matrix::zeros(100, 100);
    let b = Matrix::zeros(100, 100);

    c.bench_function("matmul 100x100", |bencher| {
        bencher.iter(|| black_box(ops::matmul(&a, &b)))
    });
}

fn benchmark_matmul_large(c: &mut Criterion) {
    let a = Matrix::zeros(500, 500);
    let b = Matrix::zeros(500, 500);

    c.bench_function("matmul 500x500", |bencher| {
        bencher.iter(|| black_box(ops::matmul(&a, &b)))
    });
}

fn benchmark_transpose(c: &mut Criterion) {
    let m = Matrix::zeros(500, 500);

    c.bench_function("transpose 500x500", |bencher| {
        bencher.iter(|| black_box(m.transpose()))
    });
}

fn benchmark_determinant(c: &mut Criterion) {
    let m = Matrix::identity(100);

    c.bench_function("determinant 100x100", |bencher| {
        bencher.iter(|| black_box(m.determinant()))
    });
}

fn benchmark_inverse(c: &mut Criterion) {
    let m = Matrix::identity(100);

    c.bench_function("inverse 100x100", |bencher| {
        bencher.iter(|| black_box(m.inverse()))
    });
}

criterion_group!(
    benches,
    benchmark_matmul,
    benchmark_matmul_large,
    benchmark_transpose,
    benchmark_determinant,
    benchmark_inverse
);
criterion_main!(benches);
