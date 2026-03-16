#ifndef __GAT_HARDWARE_DEF_H__
#define __GAT_HARDWARE_DEF_H__

// https://support.xilinx.com/s/question/0D52E00006iHkfp/vivado-20153-hls-bug-gmph?language=en_US
#include <gmp.h>
#define __gmp_const const

#include "gat_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ap_fixed.h>
#include <hls_vector.h>

// #region Model Parameters
constexpr int MAX_EDGE_COUNT = 8000;
constexpr int MAX_NODE_COUNT = 1000;
constexpr int NODE_FEATURE_DIM = 9;
constexpr int TOTAL_NODE_FEATURE_DIM = 173;
constexpr int EDGE_ATTRIBUTE_DIM = 3;
constexpr int EDGE_FEATURE_PER_LAYER = 13;
constexpr int EMBEDDING_DIM = 16;
constexpr int ATTENTION_HEAD_NUM = 4;
constexpr int NETWORK_LAYER_NUM = 2;
constexpr int TASK_NUM = 1;
#region Model Parameters

// Neighbor aggregation range: [NEIGHBOR_BASE - RANGE_OFFSET, NEIGHBOR_BASE + RANGE_OFFSET]
constexpr int NEIGHBOR_BASE = 15;
constexpr int RANGE_OFFSET = 5;

// #endregion

// #region Hardware Parameters
constexpr int GATHER_PARALLEL_FACTOR = 8; // how many dimensions of EMBEDDING_DIM should a message passing PE process each cycle?
constexpr int APPLY_PARALLEL_FACTOR = 1; // how many dimensions of EMBEDDING_DIM should the node embedding PE process each cycle?
constexpr int NODE_PARALLEL_FACTOR = 2; // how many nodes should the node embedding PE process simultaneously?
constexpr int EDGE_PARALLEL_FACTOR = 4; // how many message passing PEs are there?
constexpr int MLP_PARALLEL_FACTOR = 2;
// #endregion

// Graph analysis parameters
constexpr int GRAPH_ANALYSIS_NUM = 1;
constexpr int GRAPH_ANALYSIS_MIN_NODE_NUM = 19;
constexpr int GRAPH_ANALYSIS_AVG_NODE_NUM = 19;
constexpr int GRAPH_ANALYSIS_MAX_NODE_NUM = 19;
constexpr int GRAPH_ANALYSIS_MIN_EDGE_NUM = 40;
constexpr int GRAPH_ANALYSIS_AVG_EDGE_NUM = 40;
constexpr int GRAPH_ANALYSIS_MAX_EDGE_NUM = 40;
// #endregion

// #region Data Types
typedef ap_fixed<16, 6> FEATURE_MAP_TYPE;
typedef ap_fixed<16, 6> WEIGHT_TYPE;

typedef hls::vector<FEATURE_MAP_TYPE, ATTENTION_HEAD_NUM> FEATURE_MAP_VECTOR;
typedef hls::vector<WEIGHT_TYPE, ATTENTION_HEAD_NUM> WEIGHT_VECTOR;

typedef hls::vector<FEATURE_MAP_VECTOR, GATHER_PARALLEL_FACTOR> msg_passing_output_t;
typedef hls::vector<FEATURE_MAP_VECTOR, APPLY_PARALLEL_FACTOR> node_embedding_input_t;
typedef hls::vector<FEATURE_MAP_TYPE, MLP_PARALLEL_FACTOR> mlp_transfer_t;

typedef hls::vector<int, NODE_FEATURE_DIM> node_feature_struct_t;
typedef hls::vector<int, EDGE_ATTRIBUTE_DIM> edge_attribute_struct_t;

typedef struct {
    int source_node_id;
    int target_node_id;
} edge_struct_t;

typedef struct {
    int id;        // Neighbor node ID
    FEATURE_MAP_VECTOR attention_score;  // Attention score of neighbor node
} NeighborNode;
// #endregion

// #region Function Declarations
extern "C" {
void GraphAttentionNetwork_ProcessGraphs(
    int graph_count,
    int* node_count_per_graph,
    int* edge_count_per_graph,
    int* weight_reload_flag,
    FEATURE_MAP_TYPE output_data[][TASK_NUM],
    node_feature_struct_t* input_node_features,
    edge_struct_t* input_edge_list,
    WEIGHT_TYPE target_attention_weights[][NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE source_attention_weights[][NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE linear_projection_weights[][NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE skip_projection_weights[][NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_weights[][TASK_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_bias[][TASK_NUM]
);
}
// #endregion

// #region Global Variables
extern WEIGHT_TYPE target_attention_weights[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM];
extern WEIGHT_TYPE source_attention_weights[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM];
extern WEIGHT_TYPE linear_projection_weights[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM][ATTENTION_HEAD_NUM][EMBEDDING_DIM];
extern WEIGHT_TYPE skip_projection_weights[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM][ATTENTION_HEAD_NUM][EMBEDDING_DIM];
extern WEIGHT_TYPE graph_prediction_weights[TASK_NUM][EMBEDDING_DIM];
extern WEIGHT_TYPE graph_prediction_bias[TASK_NUM];

extern int node_degree_table[MAX_NODE_COUNT];
extern int parallel_degree_tables[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT];
extern int parallel_neighbor_tables[EDGE_PARALLEL_FACTOR][MAX_EDGE_COUNT];
extern int parallel_neighbor_tables2[EDGE_PARALLEL_FACTOR][MAX_EDGE_COUNT];
extern int edge_count_per_pe[EDGE_PARALLEL_FACTOR];

// BRAM for intermediate storage
extern FEATURE_MAP_VECTOR node_embedding_ping_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM];
extern FEATURE_MAP_VECTOR node_embedding_pong_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM];
extern FEATURE_MAP_VECTOR node_feature_skip_concat_bias_ping[MAX_NODE_COUNT][EMBEDDING_DIM];
extern FEATURE_MAP_VECTOR node_feature_skip_concat_bias_pong[MAX_NODE_COUNT][EMBEDDING_DIM];
extern FEATURE_MAP_VECTOR source_attention_scores_ping[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT];
extern FEATURE_MAP_VECTOR source_attention_scores_pong[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT];
extern FEATURE_MAP_VECTOR target_attention_scores_ping[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)];
extern FEATURE_MAP_VECTOR target_attention_scores_pong[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)];
// #endregion

#endif