use crate::core::{Dimension, Factory};
use num_traits::Num;
use smallvec::SmallVec;
use std::mem::size_of;

#[derive(Debug, Clone)]
pub struct Tensor<T: Num + Sized + Copy> {
    data: Vec<T>,
    shape: SmallVec<[usize; 5]>,
}

#[allow(dead_code)]
type FloatTensor = Tensor<f32>;

#[allow(dead_code)]
type DoubleTensor = Tensor<f64>;

#[allow(dead_code)]
type IntTensor = Tensor<i32>;

#[allow(dead_code)]
type UIntTensor = Tensor<u32>;

#[allow(dead_code)]
type LongTensor = Tensor<i64>;

#[allow(dead_code)]
type ULongTensor = Tensor<u64>;

impl<T: Num + Sized + Copy> Factory<T> for Tensor<T> {
    /// Allocate a new multi-dimensional array with all elements filled with `value`.
    ///
    /// # Arguments
    ///
    /// * `value`: Value to set for each element of the multi-dimensional array
    /// * `shape`: The shape of the multi-dimensional array
    ///
    /// returns: Self
    ///
    /// # Examples
    ///
    /// ```
    /// use mdarray::native::cpu::Tensor;
    /// use mdarray::core::Factory;
    ///
    /// let tensor = Tensor::<f32>::fill(5f32, &[32, 128]);
    /// ```
    fn fill(value: T, shape: &[usize]) -> Self {
        let numel = shape.iter().product();
        Tensor {
            data: vec![value; numel],
            shape: SmallVec::from(shape),
        }
    }

    /// Allocate a new multi-dimensional array with all elements filled with zeroes.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the multi-dimensional array
    ///
    /// returns: Self
    ///
    /// # Examples
    ///
    /// ```
    /// use mdarray::native::cpu::Tensor;
    /// use mdarray::core::Factory;
    ///
    /// let tensor = Tensor::<f32>::zeros(&[32, 128]);
    /// ```
    fn zeros(shape: &[usize]) -> Self {
        Self::fill(T::zero(), shape)
    }

    /// Allocate a new multi-dimensional array with all elements filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the multi-dimensional array
    ///
    /// returns: Self
    ///
    /// # Examples
    ///
    /// ```
    /// use mdarray::native::cpu::Tensor;
    /// use mdarray::core::Factory;
    ///
    /// let tensor = Tensor::<f32>::ones(&[32, 128]);
    /// ```
    fn ones(shape: &[usize]) -> Self {
        Self::fill(T::one(), shape)
    }
}

impl<T: Num + Sized + Copy> Dimension for Tensor<T> {
    /// # Examples
    ///
    /// ```
    /// use mdarray::native::cpu::Tensor;
    /// use mdarray::core::{Factory, Dimension};
    ///
    /// let tensor = Tensor::<f32>::ones(&[2, 5]);
    /// println!("Tensor's axes definition is {}", tensor.shape());
    /// ```
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// # Examples
    ///
    /// ```
    /// use mdarray::native::cpu::Tensor;
    /// use mdarray::core::{Factory, Dimension};
    ///
    /// let tensor = Tensor::<f32>::ones(&[2, 5]);
    /// println!("Tensor requires {} bytes", tensor.size());
    /// ```
    fn size(&self) -> usize {
        self.numel() * size_of::<T>()
    }

    /// Return the flattened number of element contained in the tensor
    ///
    /// returns: usize total number of element in this tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use mdarray::native::cpu::Tensor;
    /// use mdarray::core::{Factory, Dimension};
    ///
    /// let tensor = Tensor::<f32>::ones(&[2, 5]);
    /// println!("Tensor has {} elements", tensor.numel());
    /// ```
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    mod allocator {
        use crate::core::Factory;
        use crate::native::cpu::tensor::{DoubleTensor, FloatTensor};

        #[test]
        pub fn test_ones() {
            let t = FloatTensor::ones(&[4, 16]);
            assert!((t.data.iter().sum::<f32>() - 4f32 * 16f32).abs() < f32::EPSILON);

            let t = DoubleTensor::ones(&[4, 16]);
            assert!((t.data.iter().sum::<f64>() - 4f64 * 16f64).abs() < f64::EPSILON);
        }

        #[test]
        pub fn test_zeros() {
            let t = FloatTensor::zeros(&[4, 16]);
            assert!(t.data.iter().sum::<f32>() < f32::EPSILON);

            let t = DoubleTensor::zeros(&[4, 16]);
            assert!(t.data.iter().sum::<f64>() < f64::EPSILON);
        }

        #[test]
        pub fn test_fill() {
            let t = FloatTensor::fill(5f32, &[4, 16]);
            assert!((t.data.iter().sum::<f32>() - 5f32 * 4f32 * 16f32).abs() < f32::EPSILON);

            let t = DoubleTensor::fill(5f64, &[4, 16]);
            assert!((t.data.iter().sum::<f64>() - 5f64 * 4f64 * 16f64).abs() < f64::EPSILON);
        }
    }

    mod dimension {
        use crate::core::{Dimension, Factory};
        use crate::native::cpu::tensor::{DoubleTensor, FloatTensor};
        use std::mem::size_of;

        #[test]
        pub fn test_shape() {
            let t = FloatTensor::fill(5f32, &[4, 16]);
            assert_eq!(t.shape(), [4_usize, 16_usize]);

            let t = DoubleTensor::fill(5f64, &[4, 16]);
            assert_eq!(t.shape(), [4_usize, 16_usize]);
        }

        #[test]
        pub fn test_size() {
            let t = FloatTensor::fill(5f32, &[4, 16]);
            assert_eq!(t.size(), (4 * 16) * size_of::<f32>());

            let t = DoubleTensor::fill(5f64, &[4, 16]);
            assert_eq!(t.size(), (4 * 16) * size_of::<f64>());
        }

        #[test]
        pub fn test_numel() {
            let t = FloatTensor::fill(5f32, &[4, 16]);
            assert_eq!(t.numel(), (4 * 16));

            let t = DoubleTensor::fill(5f64, &[4, 16]);
            assert_eq!(t.numel(), (4 * 16));
        }
    }
}
