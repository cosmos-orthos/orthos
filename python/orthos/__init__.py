"""
orthos - High-performance linear algebra library written in Rust
"""

from orthos._orthos import Matrix, Vector, matmul, matvec

__version__ = "0.1.0"
__all__ = ["Matrix", "Vector", "matmul", "matvec"]
