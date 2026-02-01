"""
Solving a linear system of equations using orthos.

Solve: Ax = b
Where:
  A = [[3, 1],
       [1, 2]]
  b = [9, 8]

Solution: x = A^(-1) * b
"""

from orthos import Matrix, Vector

print("=== Solving Linear System Ax = b ===\n")

# Define the system
A = Matrix([
    [3.0, 1.0],
    [1.0, 2.0]
])

b = Vector([9.0, 8.0])

print(f"Matrix A:\n{A}")
print(f"\nVector b: {b}")

# Solve using inverse
A_inv = A.inverse()
print(f"\nA inverse:\n{A_inv}")

# x = A^(-1) * b
x = A_inv.matvec(b)
print(f"\nSolution x = A^(-1) * b: {x}")

# Verify: A * x should equal b
print("\n=== Verification ===")
result = A.matvec(x)
print(f"A * x = {result}")
print(f"b     = {b}")

# Check error
error = (result - b).norm()
print(f"Error (norm of A*x - b): {error:.2e}")


print("\n=== 3x3 System ===\n")

# Larger system
A3 = Matrix([
    [2.0, 1.0, -1.0],
    [-3.0, -1.0, 2.0],
    [-2.0, 1.0, 2.0]
])

b3 = Vector([8.0, -11.0, -3.0])

print(f"Matrix A:\n{A3}")
print(f"\nVector b: {b3}")

# Check if system is solvable (non-zero determinant)
det = A3.determinant()
print(f"\nDeterminant of A: {det}")

if abs(det) > 1e-10:
    A3_inv = A3.inverse()
    x3 = A3_inv.matvec(b3)
    print(f"\nSolution x: {x3}")

    # Verify
    result3 = A3.matvec(x3)
    error3 = (result3 - b3).norm()
    print(f"Verification error: {error3:.2e}")
else:
    print("System is singular, no unique solution exists.")
