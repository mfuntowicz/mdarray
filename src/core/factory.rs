use num_traits::Num;

pub trait Factory<T: Num + Copy> {
    /// Allocate a new multi-dimensional array with all elements filled with `value`.
    ///
    /// # Arguments
    ///
    /// * `value`: Value to set for each element of the multi-dimensional array
    /// * `shape`: The shape of the multi-dimensional array
    ///
    /// returns: Self
    fn fill(value: T, shape: &[usize]) -> Self;

    /// Allocate a new multi-dimensional array with all elements filled with zeroes.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the multi-dimensional array
    ///
    /// returns: Self
    fn zeros(shape: &[usize]) -> Self;

    /// Allocate a new multi-dimensional array with all elements filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the multi-dimensional array
    ///
    /// returns: Self
    fn ones(shape: &[usize]) -> Self;
}
