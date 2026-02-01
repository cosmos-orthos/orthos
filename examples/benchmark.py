"""
Benchmark comparing orthos Matrix and NumPy performance.
"""

import time
import numpy as np
from orthos import Matrix

def benchmark(name, fn, iterations=100):
    """Run a benchmark and return average time."""
    # Warmup
    for _ in range(5):
        fn()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iterations) * 1000
    return avg_ms

def run_benchmarks():
    print("=" * 60)
    print("NUMERIX PERFORMANCE BENCHMARK")
    print("=" * 60)

    sizes = [50, 100, 200, 500]

    for size in sizes:
        print(f"\n--- Matrix size: {size}x{size} ---")

        # Create test data
        np_a = np.random.rand(size, size).astype(np.float64)
        np_b = np.random.rand(size, size).astype(np.float64)

        # Convert to orthos types
        mat_a = Matrix.from_numpy(np_a)
        mat_b = Matrix.from_numpy(np_b)

        iterations = 100 if size <= 200 else 20

        # Benchmark matrix multiplication
        print(f"\nMatrix Multiplication ({iterations} iterations):")

        numpy_time = benchmark("NumPy", lambda: np_a @ np_b, iterations)
        print(f"  NumPy:        {numpy_time:8.3f} ms")

        matrix_time = benchmark("Matrix", lambda: mat_a @ mat_b, iterations)
        print(f"  Matrix:       {matrix_time:8.3f} ms  ({matrix_time/numpy_time:.1f}x NumPy)")

        # Benchmark transpose
        print(f"\nTranspose ({iterations} iterations):")

        numpy_time = benchmark("NumPy", lambda: np_a.T.copy(), iterations)
        print(f"  NumPy:        {numpy_time:8.3f} ms")

        matrix_time = benchmark("Matrix", lambda: mat_a.transpose(), iterations)
        print(f"  Matrix:       {matrix_time:8.3f} ms  ({matrix_time/numpy_time:.1f}x NumPy)")

        # Benchmark element access (only for smaller matrices)
        if size <= 100:
            print(f"\nElement Access (10000 accesses):")

            def numpy_access():
                for i in range(100):
                    for j in range(100):
                        _ = np_a[i % size, j % size]

            def matrix_access():
                for i in range(100):
                    for j in range(100):
                        _ = mat_a[i % size, j % size]

            numpy_time = benchmark("NumPy", numpy_access, 10)
            print(f"  NumPy:        {numpy_time:8.3f} ms")

            matrix_time = benchmark("Matrix", matrix_access, 10)
            print(f"  Matrix:       {matrix_time:8.3f} ms  ({matrix_time/numpy_time:.1f}x NumPy)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key findings:
- NumPy uses highly optimized BLAS libraries (OpenBLAS/Accelerate)
- Matrix (faer) provides competitive performance for larger matrices
- faer is optimized for modern CPU architectures

Tips to improve performance:
1. Use batch operations when possible
2. Minimize Python <-> Rust conversions
3. For very large matrices, consider using NumPy interop
""")

if __name__ == "__main__":
    run_benchmarks()
