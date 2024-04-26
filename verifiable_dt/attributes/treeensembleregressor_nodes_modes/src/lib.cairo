mod chunk0;

use orion::operators::ml::NODE_MODES;

fn get_treeensembleregressor_nodes_modes() -> Span<NODE_MODES> {

    let mut data = array![];
     chunk0::compute(ref data);

    data.span()
}