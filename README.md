# orthos

High-performance linear algebra library written in Rust with Python bindings.

## Installation

```bash
pip install orthos
```

## Usage

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

# Access elements
print(a[0, 0])      # 1.0
print(a.shape)      # (2, 2)

# NumPy interoperability
import numpy as np
np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
m = Matrix.from_numpy(np_array)
back_to_numpy = m.to_numpy()

# Vectors
v = Vector([1.0, 2.0, 3.0])
w = Vector([4.0, 5.0, 6.0])
dot_product = v.dot(w)
normalized = v.normalize()
```

## Features

- Matrix creation (zeros, ones, identity, from lists, from NumPy)
- Matrix arithmetic (add, subtract, multiply, scale)
- Matrix operations (transpose, inverse, determinant, trace)
- Vector operations (dot product, norm, normalize)
- Matrix-vector multiplication
- NumPy array interoperability

## Development

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate
uv pip install maturin pytest numpy

# Build and install in development mode
maturin develop

# Run tests
cargo test          # Rust tests
pytest tests/       # Python tests
```

## License

MIT
