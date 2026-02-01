"""
Basic matrix and vector operations with orthos.
"""

from orthos import Matrix, Vector

# Create matrices
print("=== Matrix Creation ===")
a = Matrix([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]])
print(f"Matrix A:\n{a}")
print(f"Shape: {a.shape}")

# Special matrices
zeros = Matrix.zeros(2, 3)
ones = Matrix.ones(2, 2)
identity = Matrix.identity(3)
print(f"\nZeros (2x3):\n{zeros}")
print(f"\nOnes (2x2):\n{ones}")
print(f"\nIdentity (3x3):\n{identity}")

# Matrix arithmetic
print("\n=== Matrix Arithmetic ===")
b = Matrix([[9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0]])

print(f"A + B:\n{(a + b)}")
print(f"\nA - B:\n{(a - b)}")
print(f"\nA * B (element-wise):\n{(a * b)}")
print(f"\nA @ B (matrix multiply):\n{(a @ b)}")
print(f"\nA scaled by 2:\n{a.scale(2.0)}")

# Matrix properties
print("\n=== Matrix Properties ===")
m = Matrix([[4.0, 7.0],
            [2.0, 6.0]])
print(f"Matrix M:\n{m}")
print(f"Determinant: {m.determinant()}")
print(f"Trace: {m.trace()}")
print(f"Transpose:\n{m.transpose()}")
print(f"Inverse:\n{m.inverse()}")

# Verify inverse
print(f"\nM @ M.inverse():\n{m @ m.inverse()}")

# Vector operations
print("\n=== Vector Operations ===")
v1 = Vector([1.0, 2.0, 3.0])
v2 = Vector([4.0, 5.0, 6.0])

print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"v1 + v2: {v1 + v2}")
print(f"v1 - v2: {v1 - v2}")
print(f"v1 dot v2: {v1.dot(v2)}")
print(f"v1 norm: {v1.norm()}")
print(f"v1 normalized: {v1.normalize()}")

# Matrix-vector multiplication
print("\n=== Matrix-Vector Multiplication ===")
mat = Matrix([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])
vec = Vector([1.0, 1.0])
result = mat.matvec(vec)
print(f"Matrix (3x2):\n{mat}")
print(f"Vector: {vec}")
print(f"Result: {result}")
