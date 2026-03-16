#ifndef __GRAPH_MESSAGE_PASSING_H__
#define __GRAPH_MESSAGE_PASSING_H__

#include "gat_define.h"
#include "hls_stream.h"

void message_passing_pe(
    int pe_id,
    FEATURE_MAP_VECTOR node_embedding_buffer[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    hls::stream<msg_passing_output_t> messages[NODE_PARALLEL_FACTOR],
    hls::stream<FEATURE_MAP_VECTOR> attention_score_sums[NODE_PARALLEL_FACTOR],
    int node_count
);

#endif