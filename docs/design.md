# Design Document: numerix Linear Algebra Library

## 1. Introduction

### 1.1 Purpose

This document describes the design of a Rust-based linear algebra library that exposes its functionality to Python. The project serves as a reference implementation for building high-performance Rust libraries consumable by Python applications.

### 1.2 Scope

The library provides basic linear algebra operations including:
- Matrix creation and manipulation
- Matrix arithmetic (addition, subtraction, multiplication)
- Vector operations
- Common matrix operations (transpose, inverse, determinant)

### 1.3 Terminology

| Term | Definition |
|------|------------|
| PyO3 | Rust bindings for Python, enabling Rust code to be called from Python |
| Maturin | Build system for creating Python packages from Rust code |
| Wheel | Python package distribution format |
| FFI | Foreign Function Interface |

## 2. Goals and Non-Goals

### 2.1 Goals

- Create a performant linear algebra library in Rust
- Expose all functionality through a clean Python API
- Support installation via pip, poetry, and uv
- Provide type hints for Python IDE support
- Maintain memory safety through Rust's ownership model
- Zero-copy data transfer where possible between Python and Rust

### 2.2 Non-Goals

- GPU acceleration (future consideration)
- Sparse matrix support in initial version
- Replacing NumPy for general-purpose numerical computing
- Support for Python versions below 3.9

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Python Application                       │
├─────────────────────────────────────────────────────────────┤
│                     Python API Layer                         │
│                   (Type hints, Docstrings)                   │
├─────────────────────────────────────────────────────────────┤
│                      PyO3 Bindings                           │
│            (Python ↔ Rust type conversions)                  │
├─────────────────────────────────────────────────────────────┤
│                    Rust Core Library                         │
│              (Linear algebra operations)                     │
├─────────────────────────────────────────────────────────────┤
│                   nalgebra / ndarray                         │
│              (Underlying math library)                       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Component Overview

```
numerix/
├── Cargo.toml              # Rust package manifest
├── pyproject.toml          # Python package configuration
├── src/
│   ├── lib.rs              # Library entry point and PyO3 module
│   ├── matrix.rs           # Matrix type and operations
│   ├── vector.rs           # Vector type and operations
│   ├── ops.rs              # Mathematical operations
│   └── error.rs            # Error types and conversions
├── python/
│   └── numerix/
│       ├── __init__.py     # Python package init
│       └── __init__.pyi    # Type stub file
├── tests/
│   ├── rust/               # Rust unit tests
│   └── python/             # Python integration tests
└── docs/
    ├── objective.md
    └── design.md
```

## 4. Technical Design

### 4.1 Rust Core Library

#### 4.1.1 Matrix Type

The `Matrix` struct wraps the underlying linear algebra library and provides a consistent interface.

```rust
use nalgebra::DMatrix;

pub struct Matrix {
    inner: DMatrix<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, MatrixError>;
    pub fn zeros(rows: usize, cols: usize) -> Self;
    pub fn identity(size: usize) -> Self;
    pub fn from_rows(data: Vec<Vec<f64>>) -> Result<Self, MatrixError>;

    pub fn rows(&self) -> usize;
    pub fn cols(&self) -> usize;
    pub fn get(&self, row: usize, col: usize) -> Option<f64>;
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), MatrixError>;

    pub fn transpose(&self) -> Self;
    pub fn inverse(&self) -> Result<Self, MatrixError>;
    pub fn determinant(&self) -> f64;
    pub fn trace(&self) -> f64;
}
```

#### 4.1.2 Vector Type

```rust
use nalgebra::DVector;

pub struct Vector {
    inner: DVector<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self;
    pub fn zeros(size: usize) -> Self;

    pub fn len(&self) -> usize;
    pub fn get(&self, index: usize) -> Option<f64>;
    pub fn set(&mut self, index: usize, value: f64) -> Result<(), VectorError>;

    pub fn dot(&self, other: &Vector) -> Result<f64, VectorError>;
    pub fn norm(&self) -> f64;
    pub fn normalize(&self) -> Self;
}
```

#### 4.1.3 Operations Module

```rust
// Matrix-Matrix operations
pub fn matmul(a: &Matrix, b: &Matrix) -> Result<Matrix, MatrixError>;
pub fn add(a: &Matrix, b: &Matrix) -> Result<Matrix, MatrixError>;
pub fn subtract(a: &Matrix, b: &Matrix) -> Result<Matrix, MatrixError>;
pub fn elementwise_multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, MatrixError>;

// Matrix-Scalar operations
pub fn scale(matrix: &Matrix, scalar: f64) -> Matrix;

// Matrix-Vector operations
pub fn matvec(matrix: &Matrix, vector: &Vector) -> Result<Vector, MatrixError>;
```

#### 4.1.4 Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum MatrixError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("index out of bounds: ({row}, {col}) for matrix of size ({rows}, {cols})")]
    IndexOutOfBounds { row: usize, col: usize, rows: usize, cols: usize },

    #[error("matrix is singular and cannot be inverted")]
    SingularMatrix,

    #[error("invalid input: {message}")]
    InvalidInput { message: String },
}
```

### 4.2 PyO3 Bindings

#### 4.2.1 Python Module Structure

```rust
use pyo3::prelude::*;

#[pymodule]
fn numerix(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMatrix>()?;
    m.add_class::<PyVector>()?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    Ok(())
}
```

#### 4.2.2 PyMatrix Class

```rust
#[pyclass(name = "Matrix")]
pub struct PyMatrix {
    inner: Matrix,
}

#[pymethods]
impl PyMatrix {
    #[new]
    fn new(data: Vec<Vec<f64>>) -> PyResult<Self>;

    #[staticmethod]
    fn zeros(rows: usize, cols: usize) -> Self;

    #[staticmethod]
    fn identity(size: usize) -> Self;

    #[getter]
    fn shape(&self) -> (usize, usize);

    fn __repr__(&self) -> String;
    fn __str__(&self) -> String;

    fn __getitem__(&self, idx: (usize, usize)) -> PyResult<f64>;
    fn __setitem__(&mut self, idx: (usize, usize), value: f64) -> PyResult<()>;

    fn __add__(&self, other: &PyMatrix) -> PyResult<PyMatrix>;
    fn __sub__(&self, other: &PyMatrix) -> PyResult<PyMatrix>;
    fn __mul__(&self, other: &PyMatrix) -> PyResult<PyMatrix>;
    fn __matmul__(&self, other: &PyMatrix) -> PyResult<PyMatrix>;

    fn transpose(&self) -> PyMatrix;
    fn inverse(&self) -> PyResult<PyMatrix>;
    fn determinant(&self) -> f64;
    fn trace(&self) -> f64;

    fn to_list(&self) -> Vec<Vec<f64>>;
}
```

#### 4.2.3 NumPy Integration

```rust
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};

#[pymethods]
impl PyMatrix {
    #[staticmethod]
    fn from_numpy(py: Python, array: PyReadonlyArray2<f64>) -> PyResult<Self>;

    fn to_numpy<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64>;
}
```

### 4.3 Error Mapping

Rust errors are converted to Python exceptions:

| Rust Error | Python Exception |
|------------|------------------|
| `MatrixError::DimensionMismatch` | `ValueError` |
| `MatrixError::IndexOutOfBounds` | `IndexError` |
| `MatrixError::SingularMatrix` | `ValueError` |
| `MatrixError::InvalidInput` | `ValueError` |

## 5. Python API Design

### 5.1 Usage Examples

```python
from numerix import Matrix, Vector

# Create matrices
a = Matrix([[1.0, 2.0], [3.0, 4.0]])
b = Matrix.zeros(2, 2)
c = Matrix.identity(3)

# Matrix operations
d = a + b           # Addition
e = a - b           # Subtraction
f = a * b           # Element-wise multiplication
g = a @ b           # Matrix multiplication

# Properties
print(a.shape)      # (2, 2)
print(a[0, 1])      # 2.0
print(a.determinant())
print(a.transpose())

# NumPy interop
import numpy as np
np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
m = Matrix.from_numpy(np_array)
back_to_numpy = m.to_numpy()
```

### 5.2 Type Stubs

```python
# python/numerix/__init__.pyi
from typing import List, Tuple, overload
import numpy as np
from numpy.typing import NDArray

class Matrix:
    def __init__(self, data: List[List[float]]) -> None: ...

    @staticmethod
    def zeros(rows: int, cols: int) -> Matrix: ...

    @staticmethod
    def identity(size: int) -> Matrix: ...

    @staticmethod
    def from_numpy(array: NDArray[np.float64]) -> Matrix: ...

    @property
    def shape(self) -> Tuple[int, int]: ...

    def __getitem__(self, idx: Tuple[int, int]) -> float: ...
    def __setitem__(self, idx: Tuple[int, int], value: float) -> None: ...

    def __add__(self, other: Matrix) -> Matrix: ...
    def __sub__(self, other: Matrix) -> Matrix: ...
    def __mul__(self, other: Matrix) -> Matrix: ...
    def __matmul__(self, other: Matrix) -> Matrix: ...

    def transpose(self) -> Matrix: ...
    def inverse(self) -> Matrix: ...
    def determinant(self) -> float: ...
    def trace(self) -> float: ...

    def to_list(self) -> List[List[float]]: ...
    def to_numpy(self) -> NDArray[np.float64]: ...

class Vector:
    def __init__(self, data: List[float]) -> None: ...

    @staticmethod
    def zeros(size: int) -> Vector: ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> float: ...
    def __setitem__(self, idx: int, value: float) -> None: ...

    def dot(self, other: Vector) -> float: ...
    def norm(self) -> float: ...
    def normalize(self) -> Vector: ...

    def to_list(self) -> List[float]: ...
```

## 6. Build and Packaging

### 6.1 Cargo.toml

```toml
[package]
name = "numerix"
version = "0.1.0"
edition = "2021"

[lib]
name = "numerix"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
nalgebra = "0.32"
thiserror = "1.0"

[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "matrix_ops"
harness = false
```

### 6.2 pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "numerix"
version = "0.1.0"
description = "High-performance linear algebra library written in Rust"
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "numpy>=1.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-benchmark",
]

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
module-name = "numerix._numerix"
```

### 6.3 Build Commands

```bash
# Development build (debug mode, installed in current venv)
maturin develop

# Release build
maturin build --release

# Build wheel for specific Python version
maturin build --release -i python3.11

# Build wheels for multiple platforms (CI)
maturin build --release --target x86_64-unknown-linux-gnu
maturin build --release --target aarch64-apple-darwin

# Publish to PyPI
maturin publish
```

## 7. Testing Strategy

### 7.1 Rust Unit Tests

Located in `src/*.rs` using `#[cfg(test)]` modules:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication() {
        let a = Matrix::from_rows(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]).unwrap();
        let b = Matrix::identity(2);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(a, c);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(4, 2);
        assert!(matmul(&a, &b).is_err());
    }
}
```

### 7.2 Python Integration Tests

Located in `tests/python/`:

```python
# tests/python/test_matrix.py
import pytest
import numpy as np
from numerix import Matrix

def test_matrix_creation():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert m.shape == (2, 2)
    assert m[0, 0] == 1.0

def test_matrix_multiplication():
    a = Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = Matrix.identity(2)
    c = a @ b
    assert c.to_list() == a.to_list()

def test_numpy_interop():
    np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = Matrix.from_numpy(np_array)
    result = m.to_numpy()
    np.testing.assert_array_equal(np_array, result)

def test_dimension_mismatch_raises():
    a = Matrix.zeros(2, 3)
    b = Matrix.zeros(4, 2)
    with pytest.raises(ValueError):
        a @ b
```

### 7.3 Benchmark Tests

```rust
// benches/matrix_ops.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use numerix::Matrix;

fn benchmark_matmul(c: &mut Criterion) {
    let a = Matrix::zeros(100, 100);
    let b = Matrix::zeros(100, 100);

    c.bench_function("matmul 100x100", |bencher| {
        bencher.iter(|| {
            black_box(numerix::ops::matmul(&a, &b))
        })
    });
}

criterion_group!(benches, benchmark_matmul);
criterion_main!(benches);
```

## 8. Performance Considerations

### 8.1 Memory Layout

- Use column-major order (Fortran-style) to match nalgebra's internal representation
- Provide row-major constructors for Python convenience with internal conversion
- Document memory layout for users working with raw buffers

### 8.2 Zero-Copy Where Possible

- NumPy arrays with compatible memory layout can be wrapped without copying
- Use `PyReadonlyArray` for read-only access to NumPy data
- Document when copies occur vs. zero-copy views

### 8.3 SIMD Optimization

- nalgebra automatically uses SIMD when available
- Compile with appropriate target features for production:
  ```bash
  RUSTFLAGS="-C target-cpu=native" maturin build --release
  ```

## 9. Future Considerations

### 9.1 Potential Extensions

- Sparse matrix support using `nalgebra-sparse`
- GPU acceleration via `cudarc` or OpenCL bindings
- Additional decompositions (LU, QR, SVD, Eigendecomposition)
- Complex number support
- Parallel operations using `rayon`

### 9.2 API Stability

- Follow semantic versioning
- Mark experimental APIs with `_experimental` suffix
- Provide deprecation warnings for breaking changes

## 10. References

- [PyO3 User Guide](https://pyo3.rs/)
- [Maturin Documentation](https://www.maturin.rs/)
- [nalgebra Documentation](https://nalgebra.org/)
- [NumPy C-API and Memory Layout](https://numpy.org/doc/stable/reference/c-api/)
