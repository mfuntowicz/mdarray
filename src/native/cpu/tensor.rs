use crate::core::{Dimension, Factory};
use num_traits::Num;
use smallvec::SmallVec;

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
    /// let tensor = Tensor::<f32>::fill(5f32, &vec![32, 128]);
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
    /// let tensor = Tensor::<f32>::zeros(&vec![32, 128]);
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
    /// let tensor = Tensor::<f32>::ones(&vec![32, 128]);
    /// ```
    fn ones(shape: &[usize]) -> Self {
        Self::fill(T::one(), shape)
    }
}

impl<T: Num + Sized + Copy> Dimension for Tensor<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn size(&self) -> usize {
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
            let t = FloatTensor::ones(&vec![4, 16]);
            assert_eq!(t.data.iter().sum::<f32>(), 4f32 * 16f32);

            let t = DoubleTensor::ones(&vec![4, 16]);
            assert_eq!(t.data.iter().sum::<f64>(), 4f64 * 16f64);
        }

        #[test]
        pub fn test_zeros() {
            let t = FloatTensor::zeros(&vec![4, 16]);
            assert_eq!(t.data.iter().sum::<f32>(), 0.0);

            let t = DoubleTensor::zeros(&vec![4, 16]);
            assert_eq!(t.data.iter().sum::<f64>(), 0.0);
        }

        #[test]
        pub fn test_fill() {
            let t = FloatTensor::fill(5f32, &vec![4, 16]);
            assert_eq!(t.data.iter().sum::<f32>(), 5f32 * 4f32 * 16f32);

            let t = DoubleTensor::fill(5f64, &vec![4, 16]);
            assert_eq!(t.data.iter().sum::<f64>(), 5f64 * 4f64 * 16f64);
        }
    }

    mod dimension {
        use crate::core::{Dimension, Factory};
        use crate::native::cpu::tensor::{DoubleTensor, FloatTensor};

        #[test]
        pub fn test_shape() {
            let t = FloatTensor::fill(5f32, &vec![4, 16]);
            assert_eq!(t.shape(), [4_usize, 16_usize]);

            let t = DoubleTensor::fill(5f64, &vec![4, 16]);
            assert_eq!(t.shape(), [4_usize, 16_usize]);
        }

        #[test]
        pub fn test_size() {
            let t = FloatTensor::fill(5f32, &vec![4, 16]);
            assert_eq!(t.size(), (4 * 16) as usize);

            let t = DoubleTensor::fill(5f64, &vec![4, 16]);
            assert_eq!(t.shape(), [4_usize, 16_usize]);
        }
    }
}
