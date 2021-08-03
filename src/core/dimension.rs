pub trait Dimension {
    /// Return the shape (i.e. number of elements on each axis) of the tensor.
    ///
    /// returns: tuple with each items being the number of elements on this tensor axis
    fn shape(&self) -> &[usize];

    /// Return the size in memory this tensor uses
    ///
    /// returns: usize size in byte used to store the tensor in memory
    fn size(&self) -> usize;

    /// Return the flattened number of element contained in the tensor
    ///
    /// returns: usize total number of element in this tensor
    fn numel(&self) -> usize;
}
