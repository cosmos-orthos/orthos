"""
NumPy interoperability with orthos.
"""

import numpy as np
from orthos import Matrix, Vector

print("=== NumPy to orthos ===")

# Create NumPy array
np_matrix = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
], dtype=np.float64)
print(f"NumPy array:\n{np_matrix}")
print(f"NumPy dtype: {np_matrix.dtype}")

# Convert to orthos Matrix
num_matrix = Matrix.from_numpy(np_matrix)
print(f"\northos Matrix:\n{num_matrix}")
print(f"Shape: {num_matrix.shape}")

print("\n=== orthos to NumPy ===")

# Create orthos Matrix
m = Matrix([[10.0, 20.0],
            [30.0, 40.0]])
print(f"orthos Matrix:\n{m}")

# Convert back to NumPy
np_result = m.to_numpy()
print(f"\nBack to NumPy:\n{np_result}")
print(f"Type: {type(np_result)}")
print(f"Dtype: {np_result.dtype}")

print("\n=== Mixed Operations ===")

# Use NumPy for data generation, orthos for computation
np_data = np.random.rand(3, 3) * 10
print(f"Random NumPy data:\n{np_data}")

# Convert and compute
num_m = Matrix.from_numpy(np_data)
transposed = num_m.transpose()
print(f"\nTransposed via orthos:\n{transposed.to_numpy()}")

print("\n=== Vector Interop ===")

np_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
num_vec = Vector.from_numpy(np_vec)
print(f"NumPy vector: {np_vec}")
print(f"orthos Vector: {num_vec}")
print(f"Norm (orthos): {num_vec.norm()}")
print(f"Norm (NumPy): {np.linalg.norm(np_vec)}")

# Convert back
back_to_np = num_vec.to_numpy()
print(f"Back to NumPy: {back_to_np}")

print("\n=== Performance Comparison ===")

import time

size = 100
np_a = np.random.rand(size, size)
np_b = np.random.rand(size, size)

# NumPy matmul
start = time.perf_counter()
for _ in range(100):
    _ = np_a @ np_b
np_time = time.perf_counter() - start

# orthos matmul
num_a = Matrix.from_numpy(np_a)
num_b = Matrix.from_numpy(np_b)

start = time.perf_counter()
for _ in range(100):
    _ = num_a @ num_b
num_time = time.perf_counter() - start

print(f"100 iterations of {size}x{size} matrix multiply:")
print(f"  NumPy:   {np_time:.4f}s")
print(f"  orthos: {num_time:.4f}s")
