use crate::error::MatrixError;
use faer::prelude::*;
use faer::Mat;

/// High-performance matrix using faer backend.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub(crate) inner: Mat<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, MatrixError> {
        if data.len() != rows * cols {
            return Err(MatrixError::InvalidInput {
                message: format!(
                    "data length {} does not match dimensions {}x{}",
                    data.len(),
                    rows,
                    cols
                ),
            });
        }
        let mat = Mat::from_fn(rows, cols, |i, j| data[i * cols + j]);
        Ok(Self { inner: mat })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            inner: Mat::zeros(rows, cols),
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            inner: Mat::from_fn(rows, cols, |_, _| 1.0),
        }
    }

    pub fn identity(size: usize) -> Self {
        Self {
            inner: Mat::from_fn(size, size, |i, j| if i == j { 1.0 } else { 0.0 }),
        }
    }

    pub fn from_rows(data: Vec<Vec<f64>>) -> Result<Self, MatrixError> {
        if data.is_empty() {
            return Err(MatrixError::InvalidInput {
                message: "cannot create matrix from empty data".to_string(),
            });
        }

        let rows = data.len();
        let cols = data[0].len();

        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(MatrixError::InvalidInput {
                    message: format!("row {} has length {}, expected {}", i, row.len(), cols),
                });
            }
        }

        let flat: Vec<f64> = data.into_iter().flatten().collect();
        Self::new(rows, cols, flat)
    }

    pub fn rows(&self) -> usize {
        self.inner.nrows()
    }

    pub fn cols(&self) -> usize {
        self.inner.ncols()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    pub fn get(&self, row: usize, col: usize) -> Result<f64, MatrixError> {
        if row >= self.rows() || col >= self.cols() {
            return Err(MatrixError::IndexOutOfBounds {
                row,
                col,
                rows: self.rows(),
                cols: self.cols(),
            });
        }
        Ok(self.inner[(row, col)])
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), MatrixError> {
        if row >= self.rows() || col >= self.cols() {
            return Err(MatrixError::IndexOutOfBounds {
                row,
                col,
                rows: self.rows(),
                cols: self.cols(),
            });
        }
        self.inner[(row, col)] = value;
        Ok(())
    }

    pub fn transpose(&self) -> Self {
        Self {
            inner: self.inner.transpose().to_owned(),
        }
    }

    pub fn to_vec(&self) -> Vec<Vec<f64>> {
        (0..self.rows())
            .map(|i| (0..self.cols()).map(|j| self.inner[(i, j)]).collect())
            .collect()
    }

    pub fn matmul(&self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if self.cols() != other.rows() {
            return Err(MatrixError::DimensionMismatch {
                expected: format!("{} columns", self.cols()),
                actual: format!("{} rows", other.rows()),
            });
        }

        let result = &self.inner * &other.inner;
        Ok(Self { inner: result })
    }

    pub fn add(&self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if self.shape() != other.shape() {
            return Err(MatrixError::DimensionMismatch {
                expected: format!("{}x{}", self.rows(), self.cols()),
                actual: format!("{}x{}", other.rows(), other.cols()),
            });
        }
        Ok(Self {
            inner: &self.inner + &other.inner,
        })
    }

    pub fn sub(&self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if self.shape() != other.shape() {
            return Err(MatrixError::DimensionMismatch {
                expected: format!("{}x{}", self.rows(), self.cols()),
                actual: format!("{}x{}", other.rows(), other.cols()),
            });
        }
        Ok(Self {
            inner: &self.inner - &other.inner,
        })
    }

    pub fn scale(&self, scalar: f64) -> Matrix {
        Self {
            inner: Mat::from_fn(self.rows(), self.cols(), |i, j| self.inner[(i, j)] * scalar),
        }
    }

    pub fn mul_elementwise(&self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if self.shape() != other.shape() {
            return Err(MatrixError::DimensionMismatch {
                expected: format!("{}x{}", self.rows(), self.cols()),
                actual: format!("{}x{}", other.rows(), other.cols()),
            });
        }
        let result = Mat::from_fn(self.rows(), self.cols(), |i, j| {
            self.inner[(i, j)] * other.inner[(i, j)]
        });
        Ok(Self { inner: result })
    }

    pub fn determinant(&self) -> Result<f64, MatrixError> {
        if self.rows() != self.cols() {
            return Err(MatrixError::InvalidInput {
                message: "determinant is only defined for square matrices".to_string(),
            });
        }
        Ok(self.inner.determinant())
    }

    pub fn inverse(&self) -> Result<Self, MatrixError> {
        if self.rows() != self.cols() {
            return Err(MatrixError::InvalidInput {
                message: "cannot invert non-square matrix".to_string(),
            });
        }

        // Check if matrix is singular
        let det = self.inner.determinant();
        if det.abs() < 1e-10 {
            return Err(MatrixError::SingularMatrix);
        }

        let lu = self.inner.partial_piv_lu();
        let n = self.rows();
        let identity = Mat::<f64>::identity(n, n);
        let inverse = lu.solve(&identity);
        Ok(Self { inner: inverse })
    }

    pub fn trace(&self) -> f64 {
        let min_dim = self.rows().min(self.cols());
        (0..min_dim).map(|i| self.inner[(i, i)]).sum()
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                if (self.inner[(i, j)] - other.inner[(i, j)]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.get(0, 0).unwrap(), 1.0);
        assert_eq!(m.get(1, 2).unwrap(), 6.0);
    }

    #[test]
    fn test_matrix_from_rows() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        assert_eq!(m.shape(), (2, 2));
        assert_eq!(m.get(0, 1).unwrap(), 2.0);
        assert_eq!(m.get(1, 0).unwrap(), 3.0);
    }

    #[test]
    fn test_matrix_zeros() {
        let m = Matrix::zeros(3, 4);
        assert_eq!(m.shape(), (3, 4));
        assert_eq!(m.get(2, 3).unwrap(), 0.0);
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::identity(3);
        assert_eq!(m.get(0, 0).unwrap(), 1.0);
        assert_eq!(m.get(1, 1).unwrap(), 1.0);
        assert_eq!(m.get(0, 1).unwrap(), 0.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t.get(0, 1).unwrap(), 4.0);
    }

    #[test]
    fn test_matrix_matmul() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let b = Matrix::identity(2);
        let c = a.matmul(&b).unwrap();
        assert_eq!(a, c);
    }

    #[test]
    fn test_matrix_determinant() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let det = m.determinant().unwrap();
        assert!((det - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_inverse() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let inv = m.inverse().unwrap();
        let identity = m.matmul(&inv).unwrap();
        assert!((identity.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((identity.get(0, 1).unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(4, 2);
        assert!(a.matmul(&b).is_err());
    }
}
