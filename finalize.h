#ifndef __FINALIZE_H__
#define __FINALIZE_H__

#include "gat_define.h"
#include "hls_stream.h"

void finalize(
    hls::stream<node_embedding_input_t> messages[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_weights[TASK_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_bias[TASK_NUM],
    FEATURE_MAP_TYPE* inference_result,
    int node_count
);

#endif
