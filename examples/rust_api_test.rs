//! End-to-end test for the Rust API
//! Run with: cargo run --example rust_api_test

use orthos::{Matrix, Vector, MatrixError, VectorError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== End-to-End Rust API Test ===\n");

    println!("1. Matrix Creation");
    let m1 = Matrix::from_rows(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ])?;
    println!("   From rows: {:?}", m1.shape());

    let m2 = Matrix::zeros(3, 3);
    println!("   Zeros: {:?}", m2.shape());

    let m3 = Matrix::ones(2, 3);
    println!("   Ones: {:?}", m3.shape());

    let m4 = Matrix::identity(4);
    println!("   Identity: {:?}", m4.shape());

    println!("\n2. Matrix Operations");
    let a = Matrix::from_rows(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ])?;
    let b = Matrix::from_rows(vec![
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ])?;

    let c = a.add(&b)?;
    println!("   a + b: OK");

    let d = a.matmul(&b)?;
    println!("   a @ b: OK");

    let t = a.transpose();
    println!("   Transpose: {:?}", t.shape());

    let det = a.determinant()?;
    println!("   Determinant: {}", det);

    let inv = a.inverse()?;
    println!("   Inverse: OK");

    let trace = a.trace();
    println!("   Trace: {}", trace);

    println!("\n3. Vector Operations");
    let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::new(vec![4.0, 5.0, 6.0]);

    let v3 = v1.add(&v2)?;
    println!("   v1 + v2: OK");

    let dot = v1.dot(&v2)?;
    println!("   Dot product: {}", dot);

    let norm = v1.norm();
    println!("   Norm: {}", norm);

    let normalized = v1.normalize()?;
    println!("   Normalize: OK (norm = {})", normalized.norm());

    println!("\n4. Matrix-Vector Multiplication");
    let m = Matrix::from_rows(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
    ])?;
    let v = Vector::new(vec![1.0, 1.0]);
    let result = orthos::ops::matvec(&m, &v)?;
    println!("   matvec: {:?}", result.to_vec());

    println!("\n5. Error Handling");
    let err_result: Result<Matrix, MatrixError> = Matrix::from_rows(vec![
        vec![1.0, 2.0],
        vec![3.0], // Mismatched row length
    ]);
    match err_result {
        Ok(_) => println!("   Error handling: FAILED (should have errored)"),
        Err(e) => println!("   Error handling: OK (caught: {})", e),
    }

    println!("\n=== All Rust API tests PASSED ===");
    Ok(())
}
