
    use orion::operators::ml;
    use orion::numbers::FP16x16;
    

    fn construct(
        nodes_falsenodeids: Span<usize>,
        nodes_featureids: Span<usize>,
        nodes_missing_value_tracks_true: Span<usize>,
        nodes_modes: Span<ml::NODE_MODES>,
        nodes_nodeids: Span<usize>,
        nodes_treeids: Span<usize>,
        nodes_truenodeids: Span<usize>,
        nodes_values: Span<FP16x16>,
        tree_ids: Span<usize>,
        target_ids: Span<usize>,
        target_nodeids: Span<usize>,
        target_treeids: Span<usize>,
        target_weights: Span<FP16x16>,
        base_values: Option<Span<FP16x16>>,
        n_targets: usize,
        aggregate_function: ml::AGGREGATE_FUNCTION,
        post_transform: ml::POST_TRANSFORM,
    ) -> ml::TreeEnsembleRegressor<FP16x16>{
        
        
    let atts = ml::TreeEnsembleAttributes {
        nodes_falsenodeids,
        nodes_featureids,
        nodes_missing_value_tracks_true,
        nodes_modes,
        nodes_nodeids,
        nodes_treeids,
        nodes_truenodeids,
        nodes_values
    };
     

        let mut root_index: Felt252Dict<usize> = Default::default();
    root_index.insert(0, 0); 

        let mut node_index: Felt252Dict<usize> = Default::default();
    node_index.insert(0x49ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804, 0);
    node_index.insert(0x46c9aeb066cc2f41c7124af30514f9e607137fbac950524f5fdace5788f9d43, 1);
    node_index.insert(0x5bb663c0ca15dc40c77b0202ea500d6188765a1196a91b446cb44284ad5b93c, 2);
    node_index.insert(0x68ba2a188dd231112c1cb5aaa5d18be6d84f6c8683e5c3a6638dee83e727acc, 3);
    node_index.insert(0x103bf3cdf64606c7b426f50434142ed14004b68f7b92e9f5ccee1a89dcc2229, 4);
    node_index.insert(0x764c361ccd3f2cd1a621916f1e5338c303b2909bf0170e54c3c49740933196d, 5);
    node_index.insert(0x174af49dd7dedac9e78334bedba58d9cbb8b09d42ae3532773dfb11e71b9527, 6);
    node_index.insert(0x6453d426673d0a0a477bd603579a6aa41c8dc1d862b2138c074be609afeb08e, 7);
    node_index.insert(0x7bb0195a9ea78bcd86aea585e2eeea63195e14980e8b1872e19aeb2cb96fe56, 8);
    node_index.insert(0x2c3135c0bc04b8fe04410bf0dd6442823108f2d8addbd89336fb15c4be2abed, 9);
    node_index.insert(0x4c77a4b1a33ba92acb8b62453ca1a86979f893d45a05bf389a4c5885a5aa180, 10);
    node_index.insert(0x27f73e6c94fa8249ec9f2f4eec607acc97fa632c9e8fb6c49437e62390d9860, 11);
    node_index.insert(0x5bb5b0e499560ee76313b28a1eb3ad3264b7a07c148a4614157913fae33b163, 12);
    node_index.insert(0x147435e940ce9e81f77cc8c67afe590e21b00b660a14a119b75f43899c05c71, 13);
    node_index.insert(0x3a362ea2efbf5ee0a8bb23de90e0f03bd88908990f16dd992f56f7e8c8e1f68, 14);
    node_index.insert(0x636160564968cfec125192185ffa5199638742d95729688a39c27863d2a9062, 15);
    node_index.insert(0x584dc4d760cd2df0eb3cef390531890e0466ff83c1492e3e7bae83fb58cb6b0, 16);
    node_index.insert(0x46dfe94c320b1647e085c573328f7cb76c8e1700c123c8cd8e43a8424189bf2, 17);
    node_index.insert(0x4e60801adf03f936d9e1a2e1dddf40779c97835b52b2d910e764ac2f9fc5ec7, 18);
    node_index.insert(0x70b2b8d07ea7a99fab000c8daf00a0ef4d13cd5ef3150a3a5bb40d1839dd93b, 19);
    node_index.insert(0x2ab49caa95827e28817e6f8ec7d9d56635dbd2bfcaa7f8731ba764a86a1e59b, 20);
    node_index.insert(0x4228361665fa0bd45b3a31ad7571700fa7457f5431fdcd70c871a6e902c41f6, 21);
    node_index.insert(0x15dcf5f2843dba0e0c01f0d8e2cf3e567c87268cd036636ff3ea9f0b5572a8e, 22); 

        
    let mut ensemble = ml::TreeEnsemble {
        atts,
        tree_ids,
        root_index,
        node_index
    };
     

        
    ml::TreeEnsembleRegressor {
        ensemble,
        target_ids,
        target_nodeids,
        target_treeids,
        target_weights,
        base_values,
        n_targets,
        aggregate_function,
        post_transform
    }
     

        
    }
    