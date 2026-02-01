use crate::error::MatrixError;
use crate::matrix::Matrix;
use crate::vector::Vector;
use faer::Mat;

pub fn matmul(a: &Matrix, b: &Matrix) -> Result<Matrix, MatrixError> {
    a.matmul(b)
}

pub fn add(a: &Matrix, b: &Matrix) -> Result<Matrix, MatrixError> {
    a.add(b)
}

pub fn subtract(a: &Matrix, b: &Matrix) -> Result<Matrix, MatrixError> {
    a.sub(b)
}

pub fn elementwise_multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, MatrixError> {
    a.mul_elementwise(b)
}

pub fn scale(matrix: &Matrix, scalar: f64) -> Matrix {
    matrix.scale(scalar)
}

pub fn matvec(matrix: &Matrix, vector: &Vector) -> Result<Vector, MatrixError> {
    if matrix.cols() != vector.len() {
        return Err(MatrixError::DimensionMismatch {
            expected: format!("{} columns", matrix.cols()),
            actual: format!("{} elements", vector.len()),
        });
    }

    // Convert vector to column matrix, multiply, convert back
    let v_mat = Mat::from_fn(vector.len(), 1, |i, _| vector.inner[i]);
    let result = &matrix.inner * &v_mat;

    let data: Vec<f64> = (0..result.nrows()).map(|i| result[(i, 0)]).collect();
    Ok(Vector::new(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_function() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let b = Matrix::from_rows(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.get(0, 0).unwrap(), 19.0); // 1*5 + 2*7
        assert_eq!(c.get(0, 1).unwrap(), 22.0); // 1*6 + 2*8
        assert_eq!(c.get(1, 0).unwrap(), 43.0); // 3*5 + 4*7
        assert_eq!(c.get(1, 1).unwrap(), 50.0); // 3*6 + 4*8
    }

    #[test]
    fn test_matvec_function() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let v = Vector::new(vec![1.0, 1.0]);
        let result = matvec(&m, &v).unwrap();
        assert_eq!(result.get(0).unwrap(), 3.0); // 1*1 + 2*1
        assert_eq!(result.get(1).unwrap(), 7.0); // 3*1 + 4*1
    }

    #[test]
    fn test_scale_function() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let scaled = scale(&m, 2.0);
        assert_eq!(scaled.get(0, 0).unwrap(), 2.0);
        assert_eq!(scaled.get(1, 1).unwrap(), 8.0);
    }
}
