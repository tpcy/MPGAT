#ifndef __CONVOLUTION_LAYER_H__
#define __CONVOLUTION_LAYER_H__

#include "gat_define.h"

void compute_convolution_layer(
    int layer_index,
    FEATURE_MAP_VECTOR node_embedding_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR next_node_embedding_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR next_source_attention_scores[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    FEATURE_MAP_VECTOR next_target_attention_scores[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    FEATURE_MAP_VECTOR node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR next_node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_TYPE* inference_result,
    int node_count
);

#endif