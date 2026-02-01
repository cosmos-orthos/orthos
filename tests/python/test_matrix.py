import pytest
import numpy as np
from orthos import Matrix, Vector, matmul, matvec


class TestMatrixCreation:
    def test_create_from_list(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        assert m.shape == (2, 2)
        assert m[0, 0] == 1.0
        assert m[1, 1] == 4.0

    def test_create_zeros(self):
        m = Matrix.zeros(3, 4)
        assert m.shape == (3, 4)
        assert m[0, 0] == 0.0
        assert m[2, 3] == 0.0

    def test_create_ones(self):
        m = Matrix.ones(2, 3)
        assert m.shape == (2, 3)
        assert m[1, 2] == 1.0

    def test_create_identity(self):
        m = Matrix.identity(3)
        assert m.shape == (3, 3)
        assert m[0, 0] == 1.0
        assert m[1, 1] == 1.0
        assert m[0, 1] == 0.0

    def test_from_numpy(self):
        np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        m = Matrix.from_numpy(np_array)
        assert m.shape == (2, 2)
        assert m[0, 0] == 1.0


class TestMatrixOperations:
    def test_addition(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[5.0, 6.0], [7.0, 8.0]])
        c = a + b
        assert c[0, 0] == 6.0
        assert c[1, 1] == 12.0

    def test_subtraction(self):
        a = Matrix([[5.0, 6.0], [7.0, 8.0]])
        b = Matrix([[1.0, 2.0], [3.0, 4.0]])
        c = a - b
        assert c[0, 0] == 4.0
        assert c[1, 1] == 4.0

    def test_elementwise_multiplication(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[2.0, 2.0], [2.0, 2.0]])
        c = a * b
        assert c[0, 0] == 2.0
        assert c[1, 1] == 8.0

    def test_matrix_multiplication(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        assert c[0, 0] == 19.0  # 1*5 + 2*7
        assert c[0, 1] == 22.0  # 1*6 + 2*8
        assert c[1, 0] == 43.0  # 3*5 + 4*7
        assert c[1, 1] == 50.0  # 3*6 + 4*8

    def test_scale(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        scaled = m.scale(2.0)
        assert scaled[0, 0] == 2.0
        assert scaled[1, 1] == 8.0

    def test_transpose(self):
        m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = m.transpose()
        assert t.shape == (3, 2)
        assert t[0, 1] == 4.0
        assert t[2, 0] == 3.0

    def test_determinant(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        det = m.determinant()
        assert abs(det - (-2.0)) < 1e-10

    def test_inverse(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        inv = m.inverse()
        identity = m @ inv
        assert abs(identity[0, 0] - 1.0) < 1e-10
        assert abs(identity[0, 1]) < 1e-10
        assert abs(identity[1, 0]) < 1e-10
        assert abs(identity[1, 1] - 1.0) < 1e-10

    def test_trace(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        assert m.trace() == 5.0


class TestMatrixConversions:
    def test_to_list(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        lst = m.to_list()
        assert lst == [[1.0, 2.0], [3.0, 4.0]]

    def test_to_numpy(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        arr = m.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        np.testing.assert_array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))


class TestMatrixErrors:
    def test_dimension_mismatch_add(self):
        a = Matrix.zeros(2, 3)
        b = Matrix.zeros(3, 2)
        with pytest.raises(ValueError):
            a + b

    def test_dimension_mismatch_matmul(self):
        a = Matrix.zeros(2, 3)
        b = Matrix.zeros(4, 2)
        with pytest.raises(ValueError):
            a @ b

    def test_index_out_of_bounds(self):
        m = Matrix.zeros(2, 2)
        with pytest.raises(IndexError):
            _ = m[5, 5]

    def test_singular_matrix_inverse(self):
        m = Matrix([[1.0, 2.0], [2.0, 4.0]])
        with pytest.raises(ValueError):
            m.inverse()


class TestVector:
    def test_create_from_list(self):
        v = Vector([1.0, 2.0, 3.0])
        assert len(v) == 3
        assert v[0] == 1.0

    def test_create_zeros(self):
        v = Vector.zeros(5)
        assert len(v) == 5
        assert v[3] == 0.0

    def test_dot_product(self):
        a = Vector([1.0, 2.0, 3.0])
        b = Vector([4.0, 5.0, 6.0])
        result = a.dot(b)
        assert result == 32.0  # 1*4 + 2*5 + 3*6

    def test_norm(self):
        v = Vector([3.0, 4.0])
        assert v.norm() == 5.0

    def test_normalize(self):
        v = Vector([3.0, 4.0])
        n = v.normalize()
        assert abs(n.norm() - 1.0) < 1e-10
        assert abs(n[0] - 0.6) < 1e-10
        assert abs(n[1] - 0.8) < 1e-10

    def test_add(self):
        a = Vector([1.0, 2.0])
        b = Vector([3.0, 4.0])
        c = a + b
        assert c[0] == 4.0
        assert c[1] == 6.0


class TestMatvec:
    def test_matvec_function(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        v = Vector([1.0, 1.0])
        result = matvec(m, v)
        assert result[0] == 3.0
        assert result[1] == 7.0

    def test_matvec_method(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        v = Vector([1.0, 1.0])
        result = m.matvec(v)
        assert result[0] == 3.0
        assert result[1] == 7.0
