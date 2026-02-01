use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MatrixError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("index out of bounds: ({row}, {col}) for matrix of size ({rows}, {cols})")]
    IndexOutOfBounds {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },

    #[error("matrix is singular and cannot be inverted")]
    SingularMatrix,

    #[error("invalid input: {message}")]
    InvalidInput { message: String },
}

impl From<MatrixError> for PyErr {
    fn from(err: MatrixError) -> PyErr {
        match err {
            MatrixError::IndexOutOfBounds { .. } => PyIndexError::new_err(err.to_string()),
            _ => PyValueError::new_err(err.to_string()),
        }
    }
}

#[derive(Debug, Error)]
pub enum VectorError {
    #[error("dimension mismatch: expected length {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("index out of bounds: {index} for vector of length {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("invalid input: {message}")]
    InvalidInput { message: String },
}

impl From<VectorError> for PyErr {
    fn from(err: VectorError) -> PyErr {
        match err {
            VectorError::IndexOutOfBounds { .. } => PyIndexError::new_err(err.to_string()),
            _ => PyValueError::new_err(err.to_string()),
        }
    }
}
