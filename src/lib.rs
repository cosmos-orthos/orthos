pub mod error;
pub mod matrix;
pub mod ops;
pub mod vector;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;

use crate::matrix::Matrix;
use crate::vector::Vector;

#[pyclass(name = "Matrix")]
#[derive(Clone)]
pub struct PyMatrix {
    inner: Matrix,
}

#[pymethods]
impl PyMatrix {
    #[new]
    fn new(data: Vec<Vec<f64>>) -> PyResult<Self> {
        let inner = Matrix::from_rows(data)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            inner: Matrix::zeros(rows, cols),
        }
    }

    #[staticmethod]
    fn ones(rows: usize, cols: usize) -> Self {
        Self {
            inner: Matrix::ones(rows, cols),
        }
    }

    #[staticmethod]
    fn identity(size: usize) -> Self {
        Self {
            inner: Matrix::identity(size),
        }
    }

    #[staticmethod]
    fn from_numpy(array: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let shape = array.shape();
        let rows = shape[0];
        let cols = shape[1];
        let data: Vec<f64> = array.as_array().iter().copied().collect();
        let inner = Matrix::new(rows, cols, data)?;
        Ok(Self { inner })
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols()
    }

    fn __repr__(&self) -> String {
        format!(
            "Matrix(shape=({}, {}))",
            self.inner.rows(),
            self.inner.cols()
        )
    }

    fn __str__(&self) -> String {
        let data = self.inner.to_vec();
        let rows: Vec<String> = data
            .iter()
            .map(|row| {
                let elements: Vec<String> = row.iter().map(|x| format!("{:.6}", x)).collect();
                format!("[{}]", elements.join(", "))
            })
            .collect();
        format!("[{}]", rows.join(",\n "))
    }

    fn __getitem__(&self, idx: (usize, usize)) -> PyResult<f64> {
        Ok(self.inner.get(idx.0, idx.1)?)
    }

    fn __setitem__(&mut self, idx: (usize, usize), value: f64) -> PyResult<()> {
        self.inner.set(idx.0, idx.1, value)?;
        Ok(())
    }

    fn __add__(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        let inner = self.inner.add(&other.inner)?;
        Ok(PyMatrix { inner })
    }

    fn __sub__(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        let inner = self.inner.sub(&other.inner)?;
        Ok(PyMatrix { inner })
    }

    fn __mul__(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        let inner = self.inner.mul_elementwise(&other.inner)?;
        Ok(PyMatrix { inner })
    }

    fn __matmul__(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        let inner = self.inner.matmul(&other.inner)?;
        Ok(PyMatrix { inner })
    }

    fn __rmul__(&self, scalar: f64) -> PyMatrix {
        PyMatrix {
            inner: self.inner.scale(scalar),
        }
    }

    fn scale(&self, scalar: f64) -> PyMatrix {
        PyMatrix {
            inner: self.inner.scale(scalar),
        }
    }

    fn transpose(&self) -> PyMatrix {
        PyMatrix {
            inner: self.inner.transpose(),
        }
    }

    fn inverse(&self) -> PyResult<PyMatrix> {
        let inner = self.inner.inverse()?;
        Ok(PyMatrix { inner })
    }

    fn determinant(&self) -> PyResult<f64> {
        Ok(self.inner.determinant()?)
    }

    fn trace(&self) -> f64 {
        self.inner.trace()
    }

    fn to_list(&self) -> Vec<Vec<f64>> {
        self.inner.to_vec()
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let data = self.inner.to_vec();
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        let array =
            numpy::ndarray::Array2::from_shape_vec((self.inner.rows(), self.inner.cols()), flat)
                .unwrap();
        array.to_pyarray(py)
    }

    fn matvec(&self, vector: &PyVector) -> PyResult<PyVector> {
        let inner = ops::matvec(&self.inner, &vector.inner)?;
        Ok(PyVector { inner })
    }
}

#[pyclass(name = "Vector")]
#[derive(Clone)]
pub struct PyVector {
    inner: Vector,
}

#[pymethods]
impl PyVector {
    #[new]
    fn new(data: Vec<f64>) -> Self {
        Self {
            inner: Vector::new(data),
        }
    }

    #[staticmethod]
    fn zeros(size: usize) -> Self {
        Self {
            inner: Vector::zeros(size),
        }
    }

    #[staticmethod]
    fn ones(size: usize) -> Self {
        Self {
            inner: Vector::ones(size),
        }
    }

    #[staticmethod]
    fn from_numpy(array: PyReadonlyArray1<f64>) -> Self {
        let data: Vec<f64> = array.as_array().iter().copied().collect();
        Self {
            inner: Vector::new(data),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("Vector(len={})", self.inner.len())
    }

    fn __str__(&self) -> String {
        let elements: Vec<String> = self
            .inner
            .to_vec()
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect();
        format!("[{}]", elements.join(", "))
    }

    fn __getitem__(&self, idx: usize) -> PyResult<f64> {
        Ok(self.inner.get(idx)?)
    }

    fn __setitem__(&mut self, idx: usize, value: f64) -> PyResult<()> {
        self.inner.set(idx, value)?;
        Ok(())
    }

    fn __add__(&self, other: &PyVector) -> PyResult<PyVector> {
        let inner = self.inner.add(&other.inner)?;
        Ok(PyVector { inner })
    }

    fn __sub__(&self, other: &PyVector) -> PyResult<PyVector> {
        let inner = self.inner.sub(&other.inner)?;
        Ok(PyVector { inner })
    }

    fn __rmul__(&self, scalar: f64) -> PyVector {
        PyVector {
            inner: self.inner.scale(scalar),
        }
    }

    fn scale(&self, scalar: f64) -> PyVector {
        PyVector {
            inner: self.inner.scale(scalar),
        }
    }

    fn dot(&self, other: &PyVector) -> PyResult<f64> {
        Ok(self.inner.dot(&other.inner)?)
    }

    fn norm(&self) -> f64 {
        self.inner.norm()
    }

    fn norm_squared(&self) -> f64 {
        self.inner.norm_squared()
    }

    fn normalize(&self) -> PyResult<PyVector> {
        let inner = self.inner.normalize()?;
        Ok(PyVector { inner })
    }

    fn to_list(&self) -> Vec<f64> {
        self.inner.to_vec()
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.to_vec().to_pyarray(py)
    }
}

#[pyfunction]
fn matmul(a: &PyMatrix, b: &PyMatrix) -> PyResult<PyMatrix> {
    let inner = ops::matmul(&a.inner, &b.inner)?;
    Ok(PyMatrix { inner })
}

#[pyfunction]
fn matvec(matrix: &PyMatrix, vector: &PyVector) -> PyResult<PyVector> {
    let inner = ops::matvec(&matrix.inner, &vector.inner)?;
    Ok(PyVector { inner })
}

#[pymodule]
fn _orthos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMatrix>()?;
    m.add_class::<PyVector>()?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(matvec, m)?)?;
    Ok(())
}
