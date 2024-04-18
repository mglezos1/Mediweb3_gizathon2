mod chunk0;

use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node__operators_0_biases_2() -> Tensor<FP16x16> {
    let mut shape = array![8];

    let mut data = array![];
     chunk0::compute(ref data);

    TensorTrait::new(shape.span(), data.span())
}