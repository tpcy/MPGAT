#ifndef __GRAPH_NODE_EMBEDDING_H__
#define __GRAPH_NODE_EMBEDDING_H__

#include "gat_define.h"
#include "hls_stream.h"

void node_embedding_multi_pe(
    hls::stream<node_embedding_input_t> messages[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR node_embedding_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR next_node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    int layer_num,
    int node_count
);

#endif