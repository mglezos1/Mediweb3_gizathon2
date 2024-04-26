mod construct_treeensembleregressor;
use orion::operators::tensor::{Tensor, TensorTrait};
use orion::operators::tensor::{U32Tensor, I32Tensor, I8Tensor, FP8x23Tensor, FP16x16Tensor, FP32x32Tensor, BoolTensor};
use orion::numbers::{FP8x23, FP16x16, FP32x32};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
use orion::operators::nn::{NNTrait, FP16x16NN};
use orion::operators::ml;


use treeensembleregressor_nodes_falsenodeids::get_treeensembleregressor_nodes_falsenodeids;
use treeensembleregressor_nodes_featureids::get_treeensembleregressor_nodes_featureids;
use treeensembleregressor_nodes_missing_value_tracks_true::get_treeensembleregressor_nodes_missing_value_tracks_true;
use treeensembleregressor_nodes_modes::get_treeensembleregressor_nodes_modes;
use treeensembleregressor_nodes_nodeids::get_treeensembleregressor_nodes_nodeids;
use treeensembleregressor_nodes_treeids::get_treeensembleregressor_nodes_treeids;
use treeensembleregressor_nodes_truenodeids::get_treeensembleregressor_nodes_truenodeids;
use treeensembleregressor_nodes_values::get_treeensembleregressor_nodes_values;
use treeensembleregressor_target_ids::get_treeensembleregressor_target_ids;
use treeensembleregressor_target_nodeids::get_treeensembleregressor_target_nodeids;
use treeensembleregressor_target_treeids::get_treeensembleregressor_target_treeids;
use treeensembleregressor_target_weights::get_treeensembleregressor_target_weights;

fn main(node_float_input: Tensor<FP16x16>) -> MutMatrix::<FP16x16> {
let node_variable = ml::TreeEnsembleRegressorTrait::predict(construct_treeensembleregressor::construct(
            get_treeensembleregressor_nodes_falsenodeids(),
            get_treeensembleregressor_nodes_featureids(),
            get_treeensembleregressor_nodes_missing_value_tracks_true(),
            get_treeensembleregressor_nodes_modes(),
            get_treeensembleregressor_nodes_nodeids(),
            get_treeensembleregressor_nodes_treeids(),
            get_treeensembleregressor_nodes_truenodeids(),
            get_treeensembleregressor_nodes_values(),
            get_treeensembleregressor_nodes_treeids(),
            get_treeensembleregressor_target_ids(),
            get_treeensembleregressor_target_nodeids(),
            get_treeensembleregressor_target_treeids(),
            get_treeensembleregressor_target_weights(),
            Option::None,
            1,
            ml::AGGREGATE_FUNCTION::SUM,
            ml::tree_ensemble::tree_ensemble_regressor::POST_TRANSFORM::NONE,
        ),  node_float_input)
        ;

        node_variable
    }