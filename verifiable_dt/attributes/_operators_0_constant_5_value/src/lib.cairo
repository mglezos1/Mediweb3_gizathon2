mod chunk0;

use orion::operators::tensor::{I32Tensor, Tensor, TensorTrait};


fn get__operators_0_constant_5_value() -> Tensor<i32> {
    let mut shape = array![1];

    let mut data = array![];
     chunk0::compute(ref data);

    TensorTrait::new(shape.span(), data.span())
}