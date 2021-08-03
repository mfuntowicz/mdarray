use num_traits::{Num};
use smallvec::SmallVec;
use crate::core::{Factory, Dimension};

#[derive(Debug, Clone)]
pub struct Tensor<T: Num + Sized + Copy> {
    data: Vec<T>,
    shape: SmallVec<[usize; 5]>
}

type FloatTensor = Tensor<f32>;
type DoubleTensor = Tensor<f64>;


impl <T: Num + Sized + Copy> Factory<T> for Tensor<T> {

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
            shape: SmallVec::from(shape)
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


#[cfg(test)]
mod tests {

    mod allocator {
        use crate::native::cpu::tensor::{DoubleTensor, FloatTensor};
        use crate::core::Factory;

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
}
