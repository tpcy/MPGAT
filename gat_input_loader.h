#ifndef __INPUT_LOADER_H__
#define __INPUT_LOADER_H__

#include "gat_define.h"
#include "hls_stream.h"

void load_weights(
    WEIGHT_TYPE target_attention_weights_in[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE source_attention_weights_in[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE linear_projection_weights_in[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE skip_projection_weights_in[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_weights_in[TASK_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_bias_in[TASK_NUM]
);

void load_graph(
    edge_struct_t* input_edge_list,
    int node_count,
    int edge_count
);

void load_input_node_embeddings(node_feature_struct_t* input_node_features, int node_count);

#endif