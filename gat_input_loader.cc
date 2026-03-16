#include "gat_input_loader.h"

void load_weights(
    WEIGHT_TYPE target_attention_weights_in[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE source_attention_weights_in[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE linear_projection_weights_in[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE skip_projection_weights_in[NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_weights_in[TASK_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_bias_in[TASK_NUM]
)
{
#pragma HLS INLINE off

    load_scoring_fn_target: for (int layer_idx = 0; layer_idx < NETWORK_LAYER_NUM; layer_idx++)
    {
        load_scoring_fn_target_head: for (int head_idx = 0; head_idx < ATTENTION_HEAD_NUM; head_idx++)
        {
#pragma HLS PIPELINE II=ceildiv(EMBEDDING_DIM, 2)
            load_scoring_fn_target_dim: for (int dim = 0; dim < EMBEDDING_DIM; dim++)
            {
                target_attention_weights[layer_idx][head_idx][dim] = target_attention_weights_in[layer_idx][head_idx][dim];
            }
        }
    }

    load_scoring_fn_source: for (int layer_idx = 0; layer_idx < NETWORK_LAYER_NUM; layer_idx++)
    {
        load_scoring_fn_source_head: for (int head_idx = 0; head_idx < ATTENTION_HEAD_NUM; head_idx++)
        {
#pragma HLS PIPELINE II=ceildiv(EMBEDDING_DIM, 2)
            load_scoring_fn_source_dim: for (int dim = 0; dim < EMBEDDING_DIM; dim++)
            {
                source_attention_weights[layer_idx][head_idx][dim] = source_attention_weights_in[layer_idx][head_idx][dim];
            }
        }
    }

    load_linear_proj_weights: for (int layer_idx = 0; layer_idx < NETWORK_LAYER_NUM; layer_idx++)
    {
        load_linear_proj_weights_head_out: for (int head_out_idx = 0; head_out_idx < ATTENTION_HEAD_NUM; head_out_idx++)
        {
            load_linear_proj_weights_dim_out: for (int dim_out = 0; dim_out < EMBEDDING_DIM; dim_out++)
            {
                load_linear_proj_weights_head_in: for (int head_in_idx = 0; head_in_idx < ATTENTION_HEAD_NUM; head_in_idx++)
                {
#pragma HLS PIPELINE II=ceildiv(EMBEDDING_DIM, 2)
                    load_linear_proj_weights_dim_in: for (int dim_in = 0; dim_in < EMBEDDING_DIM; dim_in++)
                    {
                        linear_projection_weights[layer_idx][head_out_idx][dim_out][head_in_idx][dim_in] = linear_projection_weights_in[layer_idx][head_out_idx][dim_out][head_in_idx][dim_in];
                    }
                }
            }
        }
    }

    load_skip_proj_weights: for (int layer_idx = 0; layer_idx < NETWORK_LAYER_NUM; layer_idx++)
    {
        load_skip_proj_weights_head_out: for (int head_out_idx = 0; head_out_idx < ATTENTION_HEAD_NUM; head_out_idx++)
        {
            load_skip_proj_weights_dim_out: for (int dim_out = 0; dim_out < EMBEDDING_DIM; dim_out++)
            {
                load_skip_proj_weights_head_in: for (int head_in_idx = 0; head_in_idx < ATTENTION_HEAD_NUM; head_in_idx++)
                {
                    load_skip_proj_weights_dim_in: for (int dim_in = 0; dim_in < EMBEDDING_DIM; dim_in++)
                    {
                        skip_projection_weights[layer_idx][head_out_idx][dim_out][head_in_idx][dim_in] = skip_projection_weights_in[layer_idx][head_out_idx][dim_out][head_in_idx][dim_in];
                    }
                }
            }
        }
    }

    load_graph_pred_bias: for (int task_idx = 0; task_idx < TASK_NUM; task_idx++)
    {
        graph_prediction_bias[task_idx] = graph_prediction_bias_in[task_idx];
    }

    load_graph_pred_weights: for (int task_idx = 0; task_idx < TASK_NUM; task_idx++)
    {
        load_graph_pred_weights_dim: for (int dim_in = 0; dim_in < EMBEDDING_DIM; dim_in++)
        {
            graph_prediction_weights[task_idx][dim_in] = graph_prediction_weights_in[task_idx][dim_in];
        }
    }
}

void load_graph(
    edge_struct_t* input_edge_list,
    int node_count,
    int edge_count
)
{
#pragma HLS INLINE off

    int parallel_neighbor_table_offsets[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT];

#pragma HLS ARRAY_PARTITION variable=parallel_degree_tables complete dim=1
#pragma HLS ARRAY_PARTITION variable=parallel_neighbor_tables complete dim=1
#pragma HLS ARRAY_PARTITION variable=parallel_neighbor_tables2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=parallel_neighbor_table_offsets complete dim=1
#pragma HLS ARRAY_PARTITION variable=edge_count_per_pe complete dim=1
    for (int node_idx = 0; node_idx < node_count; node_idx++)
    {
#pragma HLS LOOP_TRIPCOUNT min=GRAPH_ANALYSIS_MIN_NODE_NUM max=GRAPH_ANALYSIS_MAX_NODE_NUM avg=GRAPH_ANALYSIS_AVG_NODE_NUM
        node_degree_table[node_idx] = 1;

        for (int pe_idx = 0; pe_idx < EDGE_PARALLEL_FACTOR; pe_idx++)
        {
#pragma HLS UNROLL
            parallel_degree_tables[pe_idx][node_idx] = (node_idx % EDGE_PARALLEL_FACTOR == pe_idx) ? 1 : 0;
        }
    }
    for (int edge_idx = 0; edge_idx < edge_count; edge_idx++)
    {
        // TODO: can we make this II=1?
#pragma HLS PIPELINE II=3
#pragma HLS LOOP_TRIPCOUNT min=GRAPH_ANALYSIS_MIN_EDGE_NUM max=GRAPH_ANALYSIS_MAX_EDGE_NUM avg=GRAPH_ANALYSIS_AVG_EDGE_NUM
        edge_struct_t edge = input_edge_list[edge_idx];
        int source_node_id = edge.source_node_id;
        int target_node_id = edge.target_node_id;
        int pe_id = source_node_id % EDGE_PARALLEL_FACTOR;
        node_degree_table[target_node_id]++;
        parallel_degree_tables[pe_id][target_node_id]++;
    }
    for (int pe_idx = 0; pe_idx < EDGE_PARALLEL_FACTOR; pe_idx++)
    {
#pragma HLS UNROLL
        edge_count_per_pe[pe_idx] = 0;
    }

    for (int node_idx = 0; node_idx < node_count; node_idx++)
    {
#pragma HLS LOOP_TRIPCOUNT min=GRAPH_ANALYSIS_MIN_NODE_NUM max=GRAPH_ANALYSIS_MAX_NODE_NUM avg=GRAPH_ANALYSIS_AVG_NODE_NUM
        for (int pe_idx = 0; pe_idx < EDGE_PARALLEL_FACTOR; pe_idx++)
        {
#pragma HLS UNROLL
            int accumulated_edge_count = edge_count_per_pe[pe_idx];
            int degree_pe = parallel_degree_tables[pe_idx][node_idx];
            parallel_neighbor_table_offsets[pe_idx][node_idx] = accumulated_edge_count;
            edge_count_per_pe[pe_idx] = accumulated_edge_count + degree_pe;

            if (node_idx % EDGE_PARALLEL_FACTOR == pe_idx)
            {
                // add self edge for every node
                parallel_neighbor_tables[pe_idx][accumulated_edge_count] = node_idx / EDGE_PARALLEL_FACTOR;
                parallel_neighbor_tables2[pe_idx][accumulated_edge_count] = node_idx / EDGE_PARALLEL_FACTOR;
                parallel_neighbor_table_offsets[pe_idx][node_idx] = accumulated_edge_count + 1;
            }
        }
    }
    for (int edge_idx = 0; edge_idx < edge_count; edge_idx++)
    {
        // TODO: can we make this II=1?
#pragma HLS PIPELINE II=4
#pragma HLS LOOP_TRIPCOUNT min=GRAPH_ANALYSIS_MIN_EDGE_NUM max=GRAPH_ANALYSIS_MAX_EDGE_NUM avg=GRAPH_ANALYSIS_AVG_EDGE_NUM
        edge_struct_t edge = input_edge_list[edge_idx];
        int source_node_id = edge.source_node_id;
        int target_node_id = edge.target_node_id;
        int pe_id = source_node_id % EDGE_PARALLEL_FACTOR;
        int edge_pe_offset = parallel_neighbor_table_offsets[pe_id][target_node_id];
        parallel_neighbor_tables[pe_id][edge_pe_offset] = source_node_id / EDGE_PARALLEL_FACTOR;
        parallel_neighbor_tables2[pe_id][edge_pe_offset] = source_node_id / EDGE_PARALLEL_FACTOR;
        parallel_neighbor_table_offsets[pe_id][target_node_id] = edge_pe_offset + 1;
    }
}

void load_input_node_embeddings(node_feature_struct_t* input_node_features, int node_count)
{
#pragma HLS INLINE off

    /*Embedding: compute input node embedding */
    for (int node_idx = 0; node_idx < node_count; node_idx++)
    {
#pragma HLS PIPELINE II=ceildiv(EMBEDDING_DIM, APPLY_PARALLEL_FACTOR)
#pragma HLS LOOP_TRIPCOUNT min=GRAPH_ANALYSIS_MIN_NODE_NUM max=GRAPH_ANALYSIS_MAX_NODE_NUM avg=GRAPH_ANALYSIS_AVG_NODE_NUM

        node_feature_struct_t node_feature_nd = input_node_features[node_idx];
        FEATURE_MAP_VECTOR node_feature_projection[EMBEDDING_DIM];
#pragma HLS ARRAY_PARTITION variable=node_feature_projection complete dim=0

        for (int dim = 0; dim < EMBEDDING_DIM; dim++)
        {
            node_feature_skip_concat_bias_ping[node_idx][dim] = FEATURE_MAP_TYPE(0);
            node_feature_projection[dim] = FEATURE_MAP_TYPE(0);
        }

        for (int feature_idx = 0; feature_idx < NODE_FEATURE_DIM; feature_idx++)
        {
            FEATURE_MAP_TYPE node_feature_value = node_feature_nd[feature_idx];
            node_feature_skip_concat_bias_ping[node_idx][feature_idx][0] = node_feature_value;

            for (int dim_out = 0; dim_out < EMBEDDING_DIM; dim_out++)
            {
                WEIGHT_VECTOR weights;
                for (int head_out_idx = 0; head_out_idx < ATTENTION_HEAD_NUM; head_out_idx++)
                {
                    weights[head_out_idx] = linear_projection_weights[0][head_out_idx][dim_out][0][feature_idx];
                }
                node_feature_projection[dim_out] += node_feature_value * weights;
            }
        }

        FEATURE_MAP_VECTOR source_attention_score_acc = FEATURE_MAP_TYPE(0);
        FEATURE_MAP_VECTOR target_attention_score_acc = FEATURE_MAP_TYPE(0);
        for (int dim = 0; dim < EMBEDDING_DIM; dim++)
        {
            WEIGHT_VECTOR source_attention_weights_vec;
            WEIGHT_VECTOR target_attention_weights_vec;
            for (int head_idx = 0; head_idx < ATTENTION_HEAD_NUM; head_idx++)
            {
                source_attention_weights_vec[head_idx] = source_attention_weights[0][head_idx][dim];
                target_attention_weights_vec[head_idx] = target_attention_weights[0][head_idx][dim];
            }

            FEATURE_MAP_VECTOR projection_result = node_feature_projection[dim];
            node_embedding_ping_buffer[node_idx % EDGE_PARALLEL_FACTOR][node_idx / EDGE_PARALLEL_FACTOR][dim] = projection_result;
            source_attention_score_acc += projection_result * source_attention_weights_vec;
            target_attention_score_acc += projection_result * target_attention_weights_vec;
        }
        for (int pe_idx = 0; pe_idx < EDGE_PARALLEL_FACTOR; pe_idx++)
        {
            source_attention_scores_ping[pe_idx][node_idx] = source_attention_score_acc;
        }
        target_attention_scores_ping[node_idx % EDGE_PARALLEL_FACTOR][node_idx / EDGE_PARALLEL_FACTOR] = target_attention_score_acc;
    }
}