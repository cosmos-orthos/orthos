# orthos

High-performance linear algebra library written in Rust.

## Installation

```toml
[dependencies]
orthos = "0.1"
```

## Quick Start

```rust
use orthos::{Matrix, Vector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create matrices
    let a = Matrix::from_rows(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ])?;
    let b = Matrix::identity(2);

    // Matrix operations
    let c = a.matmul(&b)?;       // Matrix multiplication
    let d = a.add(&b)?;          // Addition
    let e = a.transpose();       // Transpose
    let f = a.inverse()?;        // Inverse

    // Properties
    println!("Determinant: {}", a.determinant()?);  // -2.0
    println!("Trace: {}", a.trace());               // 5.0
    println!("Shape: {:?}", a.shape());             // (2, 2)

    // Element access
    println!("a[0,0] = {}", a.get(0, 0)?);

    // Vectors
    let v = Vector::new(vec![1.0, 2.0, 3.0]);
    let w = Vector::new(vec![4.0, 5.0, 6.0]);

    println!("Dot product: {}", v.dot(&w)?);   // 32.0
    println!("Norm: {}", v.norm());             // 3.7416...
    let normalized = v.normalize()?;            // Unit vector

    // Matrix-vector multiplication
    let result = orthos::ops::matvec(&a, &Vector::new(vec![1.0, 1.0]))?;

    Ok(())
}
```

## Features

### Matrix Operations
- Creation: `zeros`, `ones`, `identity`, `from_rows`, `new`
- Arithmetic: `add`, `sub`, `mul_elementwise`, `matmul`, `scale`
- Methods: `transpose`, `inverse`, `determinant`, `trace`
- Access: `get`, `set`, `shape`, `rows`, `cols`

### Vector Operations
- Creation: `new`, `zeros`, `ones`
- Arithmetic: `add`, `sub`, `scale`
- Methods: `dot`, `norm`, `norm_squared`, `normalize`
- Access: `get`, `set`, `len`, `is_empty`

### Matrix-Vector Operations
- `orthos::ops::matvec(&matrix, &vector)` - Matrix-vector multiplication
- `orthos::ops::matmul(&a, &b)` - Matrix multiplication

## Error Handling

All fallible operations return `Result` types:

```rust
use orthos::{Matrix, MatrixError};

let result: Result<Matrix, MatrixError> = Matrix::from_rows(vec![
    vec![1.0, 2.0],
    vec![3.0],  // Mismatched length - will error
]);

match result {
    Ok(m) => println!("Created matrix"),
    Err(e) => println!("Error: {}", e),
}
```

Error types:
- `MatrixError::DimensionMismatch` - Incompatible dimensions
- `MatrixError::IndexOutOfBounds` - Invalid index
- `MatrixError::SingularMatrix` - Cannot invert singular matrix
- `MatrixError::InvalidInput` - Invalid input data
- `VectorError::DimensionMismatch` - Incompatible vector lengths
- `VectorError::IndexOutOfBounds` - Invalid index

## Feature Flags

| Feature | Description |
|---------|-------------|
| `python` | Enable Python bindings via PyO3 (optional) |

```toml
# With Python bindings
[dependencies]
orthos = { version = "0.1", features = ["python"] }
```

## Performance

Built on [faer](https://github.com/sarah-ek/faer-rs), optimized for:
- Cache-friendly memory layouts
- SIMD vectorization
- Multi-threaded operations for large matrices

## Links

- [Documentation](https://docs.rs/orthos)
- [Source Code](https://github.com/cosmos-orthos/orthos)
- [Issue Tracker](https://github.com/cosmos-orthos/orthos/issues)
- [Python package](https://pypi.org/project/orthos/)

## License

Apache-2.0
