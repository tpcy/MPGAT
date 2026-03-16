#include "finalize.h"
#include "linear.h"

using std::array;

// #region Internal Function Declarations
static void generate_node_embeddings(
    hls::stream<node_embedding_input_t> messages[NODE_PARALLEL_FACTOR],
    hls::stream<hls::vector<FEATURE_MAP_TYPE, APPLY_PARALLEL_FACTOR>> node_embeddings[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    int node_count
);
static void global_average_pooling(
    hls::stream<hls::vector<FEATURE_MAP_TYPE, APPLY_PARALLEL_FACTOR>> node_embeddings[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_TYPE graph_embedding[EMBEDDING_DIM],
    int node_count
);
// #endregion

void finalize(
    hls::stream<node_embedding_input_t> messages[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_weights[TASK_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_bias[TASK_NUM],
    FEATURE_MAP_TYPE* inference_result,
    int node_count
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<hls::vector<FEATURE_MAP_TYPE, APPLY_PARALLEL_FACTOR>> node_embeddings[NODE_PARALLEL_FACTOR];
#pragma HLS STREAM variable=node_embeddings depth=(4 * ceildiv(EMBEDDING_DIM, APPLY_PARALLEL_FACTOR))
    FEATURE_MAP_TYPE graph_embedding[EMBEDDING_DIM];

    generate_node_embeddings(messages, node_embeddings, node_feature_skip_concat_bias, node_count);
    global_average_pooling(node_embeddings, graph_embedding, node_count);
    linear<EMBEDDING_DIM, TASK_NUM, TASK_NUM, false>(
        graph_embedding,
        graph_prediction_weights,
        graph_prediction_bias,
        inference_result
    );
}

static void generate_node_embeddings(
    hls::stream<node_embedding_input_t> messages[NODE_PARALLEL_FACTOR],
    hls::stream<hls::vector<FEATURE_MAP_TYPE, APPLY_PARALLEL_FACTOR>> node_embeddings[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    int node_count
)
{
#pragma HLS INLINE off

    WEIGHT_TYPE current_skip_proj_weights[ATTENTION_HEAD_NUM][APPLY_PARALLEL_FACTOR][ATTENTION_HEAD_NUM][EMBEDDING_DIM];
#pragma HLS ARRAY_PARTITION variable=current_skip_proj_weights complete dim=0

    int iteration_count = ceildiv(node_count, NODE_PARALLEL_FACTOR);
    for (int iter_idx = 0, node_base_idx = 0; iter_idx < iteration_count; iter_idx++, node_base_idx += NODE_PARALLEL_FACTOR)
    {
#pragma HLS LOOP_TRIPCOUNT min=ceildiv(GRAPH_ANALYSIS_MIN_NODE_NUM, NODE_PARALLEL_FACTOR) max=ceildiv(GRAPH_ANALYSIS_MAX_NODE_NUM, NODE_PARALLEL_FACTOR) avg=ceildiv(GRAPH_ANALYSIS_AVG_NODE_NUM, NODE_PARALLEL_FACTOR)
        for (int dim_base = 0; dim_base < EMBEDDING_DIM; dim_base += APPLY_PARALLEL_FACTOR)
        {
#pragma HLS PIPELINE II=1
            for (int dim_offset = 0; dim_offset < APPLY_PARALLEL_FACTOR; dim_offset++)
            {
                int dim = dim_base + dim_offset;
                for (int head_out = 0; head_out < ATTENTION_HEAD_NUM; head_out++)
                {
                    for (int other_dim = 0; other_dim < EMBEDDING_DIM; other_dim++)
                    {
                        for (int head_in = 0; head_in < ATTENTION_HEAD_NUM; head_in++)
                        {
                            current_skip_proj_weights[head_out][dim_offset][head_in][other_dim] = skip_projection_weights[NETWORK_LAYER_NUM - 1][head_out][dim][head_in][other_dim];
                        }
                    }
                }
            }

            for (int node_offset = 0; node_offset < NODE_PARALLEL_FACTOR; node_offset++)
            {
                int node_idx = node_base_idx + node_offset;
                if (node_idx < node_count)
                {
                    node_embedding_input_t message;
                    hls::vector<FEATURE_MAP_TYPE, APPLY_PARALLEL_FACTOR> embedding;
                    messages[node_offset] >> message;

                    // prepare_out_nodes_features() & compute_not_concat()
                    for (int dim_out_offset = 0; dim_out_offset < APPLY_PARALLEL_FACTOR; dim_out_offset++)
                    {
                        FEATURE_MAP_TYPE output_node_feature = 0;
                        for (int head = 0; head < ATTENTION_HEAD_NUM; head++)
                        {
                            output_node_feature += message[dim_out_offset][head];
                        }
                        for (int dim_in = 0; dim_in < EMBEDDING_DIM; dim_in++)
                        {
                            FEATURE_MAP_VECTOR activation = node_feature_skip_concat_bias[node_idx][dim_in];
                            for (int head_out = 0; head_out < ATTENTION_HEAD_NUM; head_out++)
                            {
                                for (int head_in = 0; head_in < ATTENTION_HEAD_NUM; head_in++)
                                {
                                    WEIGHT_TYPE weight = current_skip_proj_weights[head_out][dim_out_offset][head_in][dim_in];
                                    output_node_feature += activation[head_in] * weight;
                                }
                            }
                        }
                        embedding[dim_out_offset] = output_node_feature / ATTENTION_HEAD_NUM;
                    }

                    node_embeddings[node_offset] << embedding;
                }
            }
        }
    }
}

static void global_average_pooling(
    hls::stream<hls::vector<FEATURE_MAP_TYPE, APPLY_PARALLEL_FACTOR>> node_embeddings[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_TYPE graph_embedding[EMBEDDING_DIM],
    int node_count
)
{
#pragma HLS INLINE off

    FEATURE_MAP_TYPE embedding_sums[EMBEDDING_DIM];
#pragma HLS ARRAY_PARTITION variable=embedding_sums cyclic factor=APPLY_PARALLEL_FACTOR dim=1

    int main_iteration_count = ceildiv(node_count, NODE_PARALLEL_FACTOR) - 1;
    int tail_node_count = ((node_count - 1) % NODE_PARALLEL_FACTOR) + 1;

    global_mean_pooling_main: for (int iter_idx = 0; iter_idx < main_iteration_count; iter_idx++)
    {
#pragma HLS LOOP_TRIPCOUNT min=(ceildiv(GRAPH_ANALYSIS_MIN_NODE_NUM, NODE_PARALLEL_FACTOR) - 1) max=(ceildiv(GRAPH_ANALYSIS_MAX_NODE_NUM, NODE_PARALLEL_FACTOR) - 1) avg=(ceildiv(GRAPH_ANALYSIS_AVG_NODE_NUM, NODE_PARALLEL_FACTOR) - 1)
        for (int dim_base = 0; dim_base < EMBEDDING_DIM; dim_base += APPLY_PARALLEL_FACTOR)
        {
#pragma HLS PIPELINE II=1

            hls::vector<FEATURE_MAP_TYPE, APPLY_PARALLEL_FACTOR> embedding_slice[NODE_PARALLEL_FACTOR];
#pragma HLS ARRAY_PARTITION variable=embedding_slice complete dim=1

            for (int node_offset = 0; node_offset < NODE_PARALLEL_FACTOR; node_offset++)
            {
#pragma HLS UNROLL
                node_embeddings[node_offset] >> embedding_slice[node_offset];
            }

            for (int dim_offset = 0; dim_offset < APPLY_PARALLEL_FACTOR; dim_offset++)
            {
#pragma HLS UNROLL
                int dim = dim_base + dim_offset;
                FEATURE_MAP_TYPE graph_embedding_elem = 0;

                for (int node_offset = 0; node_offset < NODE_PARALLEL_FACTOR; node_offset++)
                {
#pragma HLS UNROLL
                    graph_embedding_elem += embedding_slice[node_offset][dim_offset];
                }

                if (iter_idx != 0) graph_embedding_elem += embedding_sums[dim];
                embedding_sums[dim] = graph_embedding_elem;
            }
        }
    }

    global_mean_pooling_tail: for (int dim_base = 0; dim_base < EMBEDDING_DIM; dim_base += APPLY_PARALLEL_FACTOR)
    {
#pragma HLS PIPELINE II=1

        hls::vector<FEATURE_MAP_TYPE, APPLY_PARALLEL_FACTOR> embedding_slice[NODE_PARALLEL_FACTOR];
#pragma HLS ARRAY_PARTITION variable=embedding_slice complete dim=1

        for (int node_offset = 0; node_offset < NODE_PARALLEL_FACTOR; node_offset++)
        {
#pragma HLS UNROLL
            if (node_offset == tail_node_count) break;
            node_embeddings[node_offset] >> embedding_slice[node_offset];
        }

        for (int dim_offset = 0; dim_offset < APPLY_PARALLEL_FACTOR; dim_offset++)
        {
#pragma HLS UNROLL
            int dim = dim_base + dim_offset;
            FEATURE_MAP_TYPE graph_embedding_elem = 0;

            for (int node_offset = 0; node_offset < NODE_PARALLEL_FACTOR; node_offset++)
            {
#pragma HLS UNROLL
                if (node_offset == tail_node_count) break;
                graph_embedding_elem += embedding_slice[node_offset][dim_offset];
            }

            if (main_iteration_count != 0) graph_embedding_elem += embedding_sums[dim];
            graph_embedding[dim] = graph_embedding_elem / node_count;
        }
    }
}