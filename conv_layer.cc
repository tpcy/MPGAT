#include "conv_layer.h"
#include "node_embedding.h"
#include "message_passing.h"
#include "gat_input_loader.h"
#include "finalize.h"
#include "hls_stream.h"

// #region Internal Function Declarations
void validate_node_embedding(
    hls::stream<node_embedding_input_t> messages[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR node_embedding_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR next_node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    FEATURE_MAP_TYPE* inference_result,
    int layer_index,
    int node_count
);
void msg_passing_to_node_emb_adapter(
    hls::stream<msg_passing_output_t> msg_passing_output[EDGE_PARALLEL_FACTOR][NODE_PARALLEL_FACTOR],
    hls::stream<FEATURE_MAP_VECTOR> attention_score_sums[EDGE_PARALLEL_FACTOR][NODE_PARALLEL_FACTOR],
    hls::stream<node_embedding_input_t>& node_embedding_input,
    int node_offset,
    int node_count
);
// #endregion

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
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable=node_embedding_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=node_embedding_buffer cyclic factor=GATHER_PARALLEL_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=next_node_embedding_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=next_node_embedding_buffer cyclic factor=GATHER_PARALLEL_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=source_attention_scores complete dim=1
#pragma HLS ARRAY_PARTITION variable=next_source_attention_scores complete dim=1
#pragma HLS ARRAY_PARTITION variable=target_attention_scores complete dim=1
#pragma HLS ARRAY_PARTITION variable=next_target_attention_scores complete dim=1

    hls::stream<msg_passing_output_t> msg_passing_output[EDGE_PARALLEL_FACTOR][NODE_PARALLEL_FACTOR];
#pragma HLS STREAM variable=msg_passing_output depth=(20 * ceildiv(EMBEDDING_DIM, GATHER_PARALLEL_FACTOR))
    hls::stream<node_embedding_input_t> node_embedding_input[NODE_PARALLEL_FACTOR];
#pragma HLS STREAM variable=node_embedding_input depth=(4 * ceildiv(EMBEDDING_DIM, APPLY_PARALLEL_FACTOR))
    hls::stream<FEATURE_MAP_VECTOR> attention_score_sums[EDGE_PARALLEL_FACTOR][NODE_PARALLEL_FACTOR];
#pragma HLS STREAM variable=attention_score_sums depth=(20)
    for (int pe_id = 0; pe_id < EDGE_PARALLEL_FACTOR; pe_id++)
    {
#pragma HLS UNROLL
        message_passing_pe(
            pe_id,
            node_embedding_buffer[pe_id],
            source_attention_scores[pe_id],
            target_attention_scores[pe_id],
            msg_passing_output[pe_id],
            attention_score_sums[pe_id],
            node_count
        );
    }
    for (int node_offset = 0; node_offset < NODE_PARALLEL_FACTOR; node_offset++)
    {
#pragma HLS UNROLL
        msg_passing_to_node_emb_adapter(
            msg_passing_output,
            attention_score_sums,
            node_embedding_input[node_offset],
            node_offset,
            node_count
        );
    }
    validate_node_embedding(
        node_embedding_input,
        next_node_embedding_buffer,
        node_feature_skip_concat_bias,
        next_node_feature_skip_concat_bias,
        next_source_attention_scores,
        next_target_attention_scores,
        inference_result,
        layer_index,
        node_count
    );
}

void validate_node_embedding(
    hls::stream<node_embedding_input_t> messages[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR node_embedding_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR next_node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    FEATURE_MAP_TYPE* inference_result,
    int layer_index,
    int node_count
)
{
#pragma HLS INLINE off

    if (layer_index == NETWORK_LAYER_NUM - 1)
        finalize(
            messages,
            node_feature_skip_concat_bias,
            graph_prediction_weights,
            graph_prediction_bias,
            inference_result,
            node_count
        );
    else
        node_embedding_multi_pe(
            messages,
            node_embedding_buffer,
            node_feature_skip_concat_bias,
            next_node_feature_skip_concat_bias,
            source_attention_scores,
            target_attention_scores,
            layer_index,
            node_count
        );
}

void msg_passing_to_node_emb_adapter(
    hls::stream<msg_passing_output_t> msg_passing_output[EDGE_PARALLEL_FACTOR][NODE_PARALLEL_FACTOR],
    hls::stream<FEATURE_MAP_VECTOR> attention_score_sums[EDGE_PARALLEL_FACTOR][NODE_PARALLEL_FACTOR],
    hls::stream<node_embedding_input_t>& node_embedding_input,
    int node_offset,
    int node_count
)
{
#pragma HLS INLINE off

    FEATURE_MAP_VECTOR current_attention_score_sums;
    int iteration_count = ceildiv(node_count - node_offset, NODE_PARALLEL_FACTOR);

    for (int i = 0; i < iteration_count; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ceildiv(GRAPH_ANALYSIS_MIN_NODE_NUM, NODE_PARALLEL_FACTOR) max=ceildiv(GRAPH_ANALYSIS_MAX_NODE_NUM, NODE_PARALLEL_FACTOR) avg=ceildiv(GRAPH_ANALYSIS_AVG_NODE_NUM, NODE_PARALLEL_FACTOR)

        // assumes GATHER_PARALLEL_FACTOR is divisible by APPLY_PARALLEL_FACTOR
        for (int msg_dim_base = 0; msg_dim_base < EMBEDDING_DIM; msg_dim_base += GATHER_PARALLEL_FACTOR)
        {
#pragma HLS PIPELINE II=ceildiv(GATHER_PARALLEL_FACTOR, APPLY_PARALLEL_FACTOR)
#pragma HLS ALLOCATION operation instances=sdiv limit=(APPLY_PARALLEL_FACTOR * ATTENTION_HEAD_NUM)

            if (msg_dim_base == 0)
            {
#pragma HLS OCCURRENCE cycle=ceildiv(EMBEDDING_DIM, GATHER_PARALLEL_FACTOR)
                current_attention_score_sums = FEATURE_MAP_TYPE(0);
                for (int pe_id = 0; pe_id < EDGE_PARALLEL_FACTOR; pe_id++)
                {
                    FEATURE_MAP_VECTOR partial_sums;
                    attention_score_sums[pe_id][node_offset] >> partial_sums;
                    current_attention_score_sums += partial_sums;
                }
            }

            msg_passing_output_t message = FEATURE_MAP_VECTOR(0);
            for (int pe_id = 0; pe_id < EDGE_PARALLEL_FACTOR; pe_id++)
            {
                msg_passing_output_t partial_message;
                msg_passing_output[pe_id][node_offset] >> partial_message;
                message += partial_message;
            }
            
            if(current_attention_score_sums[0] == 0){
                message = FEATURE_MAP_VECTOR(0);
            }else{
                message /= current_attention_score_sums;
            }

            for (int msg_dim_offset = 0; msg_dim_offset < GATHER_PARALLEL_FACTOR; msg_dim_offset += APPLY_PARALLEL_FACTOR)
            {
                int node_emb_dim_base = msg_dim_base + msg_dim_offset;
                if (node_emb_dim_base < EMBEDDING_DIM)
                {
                    node_embedding_input_t message_split;
                    for (int node_emb_dim_offset = 0; node_emb_dim_offset < APPLY_PARALLEL_FACTOR; node_emb_dim_offset++)
                    {
                        int dim_offset = msg_dim_offset + node_emb_dim_offset;
                        message_split[node_emb_dim_offset] = message[dim_offset];
                    }
                    node_embedding_input << message_split;
                }
            }
        }
    }
}