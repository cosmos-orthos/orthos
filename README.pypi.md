# orthos

High-performance linear algebra library written in Rust with Python bindings.

## Installation

```bash
pip install orthos
```

## Quick Start

```python
from orthos import Matrix, Vector

# Create matrices
a = Matrix([[1.0, 2.0], [3.0, 4.0]])
b = Matrix.identity(2)

# Matrix operations
c = a @ b           # Matrix multiplication
d = a + b           # Addition
e = a.transpose()   # Transpose
f = a.inverse()     # Inverse

# Properties
print(a.determinant())  # -2.0
print(a.trace())        # 5.0
print(a.shape)          # (2, 2)

# Element access
print(a[0, 0])      # 1.0
a[0, 0] = 10.0      # Set element

# Vectors
v = Vector([1.0, 2.0, 3.0])
w = Vector([4.0, 5.0, 6.0])
print(v.dot(w))         # 32.0
print(v.norm())         # 3.7416...
print(v.normalize())    # Unit vector

# NumPy interoperability
import numpy as np
np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
m = Matrix.from_numpy(np_array)
back_to_numpy = m.to_numpy()
```

## Features

### Matrix Operations
- Creation: `zeros`, `ones`, `identity`, from lists, from NumPy
- Arithmetic: `+`, `-`, `*` (element-wise), `@` (matrix multiply)
- Methods: `transpose()`, `inverse()`, `determinant()`, `trace()`, `scale()`
- Indexing: `matrix[row, col]`

### Vector Operations
- Creation: `zeros`, `ones`, from lists, from NumPy
- Arithmetic: `+`, `-`, scalar multiply
- Methods: `dot()`, `norm()`, `norm_squared()`, `normalize()`, `scale()`
- Indexing: `vector[index]`

### Matrix-Vector Operations
- `matvec(matrix, vector)` - Matrix-vector multiplication
- `matrix.matvec(vector)` - Method form

### NumPy Interoperability
- `Matrix.from_numpy(array)` - Create from NumPy array
- `matrix.to_numpy()` - Convert to NumPy array
- `Vector.from_numpy(array)` - Create from NumPy array
- `vector.to_numpy()` - Convert to NumPy array

## Supported Platforms

| Platform | Architectures |
|----------|---------------|
| Linux | x86_64, aarch64 |
| macOS | x86_64, Apple Silicon |
| Windows | x86_64 |

## Python Version Support

- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13

## Performance

orthos is built on [faer](https://github.com/sarah-ek/faer-rs), a high-performance linear algebra library in Rust. It provides competitive performance especially for small to medium matrices.

## Links

- [Source Code](https://github.com/cosmos-orthos/orthos)
- [Issue Tracker](https://github.com/cosmos-orthos/orthos/issues)
- [Rust crate](https://crates.io/crates/orthos)

## License

Apache-2.0
