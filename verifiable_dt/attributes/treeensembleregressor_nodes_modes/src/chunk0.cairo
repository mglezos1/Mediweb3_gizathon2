use orion::operators::ml::NODE_MODES;

fn compute(ref a: Array<NODE_MODES>) {
a.append(NODE_MODES::BRANCH_LEQ);
a.append(NODE_MODES::BRANCH_LEQ);
a.append(NODE_MODES::BRANCH_LEQ);
a.append(NODE_MODES::BRANCH_LEQ);
a.append(NODE_MODES::LEAF);
a.append(NODE_MODES::BRANCH_LEQ);
a.append(NODE_MODES::LEAF);
a.append(NODE_MODES::LEAF);
a.append(NODE_MODES::LEAF);
a.append(NODE_MODES::BRANCH_LEQ);
a.append(NODE_MODES::LEAF);
a.append(NODE_MODES::LEAF);
a.append(NODE_MODES::BRANCH_LEQ);
a.append(NODE_MODES::BRANCH_LEQ);
a.append(NODE_MODES::LEAF);
a.append(NODE_MODES::LEAF);
a.append(NODE_MODES::BRANCH_LEQ);
a.append(NODE_MODES::LEAF);
a.append(NODE_MODES::BRANCH_LEQ);
a.append(NODE_MODES::LEAF);
a.append(NODE_MODES::BRANCH_LEQ);
a.append(NODE_MODES::LEAF);
a.append(NODE_MODES::LEAF);
}