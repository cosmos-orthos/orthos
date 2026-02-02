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

#[cfg(feature = "python")]
impl From<MatrixError> for pyo3::PyErr {
    fn from(err: MatrixError) -> pyo3::PyErr {
        use pyo3::exceptions::{PyIndexError, PyValueError};
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

#[cfg(feature = "python")]
impl From<VectorError> for pyo3::PyErr {
    fn from(err: VectorError) -> pyo3::PyErr {
        use pyo3::exceptions::{PyIndexError, PyValueError};
        match err {
            VectorError::IndexOutOfBounds { .. } => PyIndexError::new_err(err.to_string()),
            _ => PyValueError::new_err(err.to_string()),
        }
    }
}
