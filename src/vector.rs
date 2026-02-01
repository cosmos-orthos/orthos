use crate::error::VectorError;

#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub(crate) inner: Vec<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        Self { inner: data }
    }

    pub fn zeros(size: usize) -> Self {
        Self {
            inner: vec![0.0; size],
        }
    }

    pub fn ones(size: usize) -> Self {
        Self {
            inner: vec![1.0; size],
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn get(&self, index: usize) -> Result<f64, VectorError> {
        if index >= self.len() {
            return Err(VectorError::IndexOutOfBounds {
                index,
                len: self.len(),
            });
        }
        Ok(self.inner[index])
    }

    pub fn set(&mut self, index: usize, value: f64) -> Result<(), VectorError> {
        if index >= self.len() {
            return Err(VectorError::IndexOutOfBounds {
                index,
                len: self.len(),
            });
        }
        self.inner[index] = value;
        Ok(())
    }

    pub fn dot(&self, other: &Vector) -> Result<f64, VectorError> {
        if self.len() != other.len() {
            return Err(VectorError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }
        Ok(self
            .inner
            .iter()
            .zip(other.inner.iter())
            .map(|(a, b)| a * b)
            .sum())
    }

    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    pub fn norm_squared(&self) -> f64 {
        self.inner.iter().map(|x| x * x).sum()
    }

    pub fn normalize(&self) -> Result<Self, VectorError> {
        let n = self.norm();
        if n == 0.0 {
            return Err(VectorError::InvalidInput {
                message: "cannot normalize zero vector".to_string(),
            });
        }
        Ok(Self {
            inner: self.inner.iter().map(|x| x / n).collect(),
        })
    }

    pub fn add(&self, other: &Vector) -> Result<Vector, VectorError> {
        if self.len() != other.len() {
            return Err(VectorError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }
        Ok(Self {
            inner: self
                .inner
                .iter()
                .zip(other.inner.iter())
                .map(|(a, b)| a + b)
                .collect(),
        })
    }

    pub fn sub(&self, other: &Vector) -> Result<Vector, VectorError> {
        if self.len() != other.len() {
            return Err(VectorError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }
        Ok(Self {
            inner: self
                .inner
                .iter()
                .zip(other.inner.iter())
                .map(|(a, b)| a - b)
                .collect(),
        })
    }

    pub fn scale(&self, scalar: f64) -> Vector {
        Self {
            inner: self.inner.iter().map(|x| x * scalar).collect(),
        }
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.inner.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(0).unwrap(), 1.0);
        assert_eq!(v.get(2).unwrap(), 3.0);
    }

    #[test]
    fn test_vector_zeros() {
        let v = Vector::zeros(5);
        assert_eq!(v.len(), 5);
        assert_eq!(v.get(3).unwrap(), 0.0);
    }

    #[test]
    fn test_vector_dot() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);
        let result = a.dot(&b).unwrap();
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_vector_norm() {
        let v = Vector::new(vec![3.0, 4.0]);
        assert_eq!(v.norm(), 5.0);
    }

    #[test]
    fn test_vector_normalize() {
        let v = Vector::new(vec![3.0, 4.0]);
        let n = v.normalize().unwrap();
        assert!((n.norm() - 1.0).abs() < 1e-10);
        assert!((n.get(0).unwrap() - 0.6).abs() < 1e-10);
        assert!((n.get(1).unwrap() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_vector_add() {
        let a = Vector::new(vec![1.0, 2.0]);
        let b = Vector::new(vec![3.0, 4.0]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.get(0).unwrap(), 4.0);
        assert_eq!(c.get(1).unwrap(), 6.0);
    }

    #[test]
    fn test_vector_dimension_mismatch() {
        let a = Vector::new(vec![1.0, 2.0]);
        let b = Vector::new(vec![1.0, 2.0, 3.0]);
        assert!(a.dot(&b).is_err());
    }
}
