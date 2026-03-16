#include "node_embedding.h"

// #region Internal Function Declarations
static void accumulate_node_features(
    hls::stream<node_embedding_input_t> messages[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR feature_accumulators[NODE_PARALLEL_FACTOR][EMBEDDING_DIM],
	FEATURE_MAP_VECTOR source_attention_score_accs[NODE_PARALLEL_FACTOR],
	FEATURE_MAP_VECTOR target_attention_score_accs[NODE_PARALLEL_FACTOR],
	FEATURE_MAP_VECTOR node_embedding_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
	FEATURE_MAP_VECTOR source_attention_scores[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT],
	FEATURE_MAP_VECTOR target_attention_scores[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    FEATURE_MAP_VECTOR node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR next_node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    int layer_num,
    int node_base_idx,
    int dim_base_idx,
    int node_count
);
static void output_node_features(
    FEATURE_MAP_VECTOR feature_accumulators[NODE_PARALLEL_FACTOR][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_score_accs[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR target_attention_score_accs[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR node_embedding_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    int layer_num,
    int node_base_idx,
    int dim_base_idx,
    int node_count
);
// #endregion

void node_embedding_multi_pe(
    hls::stream<node_embedding_input_t> messages[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR node_embedding_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR next_node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    int layer_num,
    int node_count
)
{

#pragma HLS INLINE off


    FEATURE_MAP_VECTOR feature_accumulators_ping[NODE_PARALLEL_FACTOR][EMBEDDING_DIM];
#pragma HLS ARRAY_PARTITION variable=feature_accumulators_ping complete dim=0
    FEATURE_MAP_VECTOR feature_accumulators_pong[NODE_PARALLEL_FACTOR][EMBEDDING_DIM];
#pragma HLS ARRAY_PARTITION variable=feature_accumulators_pong complete dim=0
// source_attention_score_accs和target_attention_score_accs：

    FEATURE_MAP_VECTOR source_attention_score_accs[NODE_PARALLEL_FACTOR];
#pragma HLS ARRAY_PARTITION variable=source_attention_score_accs complete dim=0
    FEATURE_MAP_VECTOR target_attention_score_accs[NODE_PARALLEL_FACTOR];
#pragma HLS ARRAY_PARTITION variable=target_attention_score_accs complete dim=0

    
    int iteration_count = ceildiv(node_count, NODE_PARALLEL_FACTOR) + 1;
    
    for (
        int iter_idx = 0, acc_node_base = 0, out_node_base = -NODE_PARALLEL_FACTOR;
        iter_idx < iteration_count;
        iter_idx++, acc_node_base += NODE_PARALLEL_FACTOR, out_node_base += NODE_PARALLEL_FACTOR
    )
    {

#pragma HLS LOOP_TRIPCOUNT min=(ceildiv(GRAPH_ANALYSIS_MIN_NODE_NUM, NODE_PARALLEL_FACTOR) + 1) max=(ceildiv(GRAPH_ANALYSIS_MAX_NODE_NUM, NODE_PARALLEL_FACTOR) + 1) avg=(ceildiv(GRAPH_ANALYSIS_AVG_NODE_NUM, NODE_PARALLEL_FACTOR) + 1)
        
        for (int dim_base_idx = 0; dim_base_idx < EMBEDDING_DIM; dim_base_idx += APPLY_PARALLEL_FACTOR)
        {

#pragma HLS PIPELINE II=1

#pragma HLS DEPENDENCE variable=node_embedding_buffer inter false
#pragma HLS DEPENDENCE variable=source_attention_scores inter false
#pragma HLS DEPENDENCE variable=target_attention_scores inter false

            
            if (iter_idx != iteration_count - 1)
            {
                accumulate_node_features(
                    messages,
                    feature_accumulators_ping,
					source_attention_score_accs,
					target_attention_score_accs,
					node_embedding_buffer,
					source_attention_scores,
					target_attention_scores,
                    node_feature_skip_concat_bias,
                    next_node_feature_skip_concat_bias,
                    layer_num,
                    acc_node_base,
                    dim_base_idx,
                    node_count
                );
            }else{  
                output_node_features(
                    feature_accumulators_ping,
                    source_attention_score_accs,
                    target_attention_score_accs,
                    node_embedding_buffer,
                    source_attention_scores,
                    target_attention_scores,
                    layer_num,
                    out_node_base,
                    dim_base_idx,
                    node_count
                );
            }
        }
    }
}



static void accumulate_node_features(
    hls::stream<node_embedding_input_t> messages[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR feature_accumulators[NODE_PARALLEL_FACTOR][EMBEDDING_DIM],
	FEATURE_MAP_VECTOR source_attention_score_accs[NODE_PARALLEL_FACTOR],
	FEATURE_MAP_VECTOR target_attention_score_accs[NODE_PARALLEL_FACTOR],
	FEATURE_MAP_VECTOR node_embedding_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
	FEATURE_MAP_VECTOR source_attention_scores[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT],
	FEATURE_MAP_VECTOR target_attention_scores[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    FEATURE_MAP_VECTOR node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR next_node_feature_skip_concat_bias[MAX_NODE_COUNT][EMBEDDING_DIM],
    int layer_num,
    int node_base_idx,
    int dim_base_idx,
    int node_count
)
{

#pragma HLS INLINE

#pragma HLS ARRAY_PARTITION variable=source_attention_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=source_attention_weights cyclic factor=APPLY_PARALLEL_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=target_attention_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=target_attention_weights cyclic factor=APPLY_PARALLEL_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=node_feature_skip_concat_bias cyclic factor=NODE_PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=node_feature_skip_concat_bias complete dim=2
#pragma HLS ARRAY_PARTITION variable=next_node_feature_skip_concat_bias cyclic factor=NODE_PARALLEL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=next_node_feature_skip_concat_bias complete dim=2
#pragma HLS ARRAY_PARTITION variable=skip_projection_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=skip_projection_weights cyclic factor=APPLY_PARALLEL_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=skip_projection_weights complete dim=4
#pragma HLS ARRAY_PARTITION variable=skip_projection_weights complete dim=5
#pragma HLS ARRAY_PARTITION variable=linear_projection_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=linear_projection_weights complete dim=3
#pragma HLS ARRAY_PARTITION variable=linear_projection_weights complete dim=4
#pragma HLS ARRAY_PARTITION variable=linear_projection_weights cyclic factor=APPLY_PARALLEL_FACTOR dim=5

// WEIGHT_VECTOR source_attention_weights[APPLY_PARALLEL_FACTOR];
// 和 WEIGHT_VECTOR target_attention_weights[APPLY_PARALLEL_FACTOR];
    WEIGHT_VECTOR source_attention_weights[APPLY_PARALLEL_FACTOR];
#pragma HLS ARRAY_PARTITION variable=source_attention_weights complete dim=0
    WEIGHT_VECTOR target_attention_weights[APPLY_PARALLEL_FACTOR];
#pragma HLS ARRAY_PARTITION variable=target_attention_weights complete dim=0


    for (int dim_offset = 0; dim_offset < APPLY_PARALLEL_FACTOR; dim_offset++)
    {
        int dim = dim_base_idx + dim_offset;
        for (int head_idx = 0; head_idx < ATTENTION_HEAD_NUM; head_idx++)
        {
            source_attention_weights[dim_offset][head_idx] = source_attention_weights[layer_num + 1][head_idx][dim];
            target_attention_weights[dim_offset][head_idx] = target_attention_weights[layer_num + 1][head_idx][dim];
        }
    }
 
    for (int node_offset = 0; node_offset < NODE_PARALLEL_FACTOR; node_offset++)
    {
        int node_idx = node_base_idx + node_offset;
        if (node_idx < node_count)
        {
        	
			FEATURE_MAP_VECTOR source_attention_score_acc = FEATURE_MAP_TYPE(0);
			FEATURE_MAP_VECTOR target_attention_score_acc = FEATURE_MAP_TYPE(0);
            node_embedding_input_t message;
            messages[node_offset] >> message;

            for (int dim_out_offset = 0; dim_out_offset < APPLY_PARALLEL_FACTOR; dim_out_offset++)
            {
                int dim_out = dim_base_idx + dim_out_offset;

                // prepare_out_nodes_features()
                FEATURE_MAP_VECTOR next_node_feature_skip_concat_bias_val = message[dim_out_offset];
                for (int dim_in = 0; dim_in < EMBEDDING_DIM; dim_in++)
                {
                    FEATURE_MAP_VECTOR activation = node_feature_skip_concat_bias[node_idx][dim_in];
                    for (int head_out_idx = 0; head_out_idx < ATTENTION_HEAD_NUM; head_out_idx++)
                    {
                        for (int head_in_idx = 0; head_in_idx < ATTENTION_HEAD_NUM; head_in_idx++)
                        {
                            WEIGHT_TYPE weight = skip_projection_weights[layer_num][head_out_idx][dim_out][head_in_idx][dim_in];
                            next_node_feature_skip_concat_bias_val[head_out_idx] += activation[head_in_idx] * weight;
                        }
                    }
                }

                // compute_activation()
                for (int head_out_idx = 0; head_out_idx < ATTENTION_HEAD_NUM; head_out_idx++)
                {
                    if (next_node_feature_skip_concat_bias_val[head_out_idx] <= 0)
                    {
                        next_node_feature_skip_concat_bias_val[head_out_idx] = hls::exp(next_node_feature_skip_concat_bias_val[head_out_idx]) - FEATURE_MAP_TYPE(1);
                    }
                }
                
                next_node_feature_skip_concat_bias[node_idx][dim_out] = next_node_feature_skip_concat_bias_val;

                // compute_nodes_features_proj()
                
                for (int proj_dim_out = 0; proj_dim_out < EMBEDDING_DIM; proj_dim_out++)
                {
                    FEATURE_MAP_VECTOR acc = (dim_out != 0) ? feature_accumulators[node_offset][proj_dim_out] : FEATURE_MAP_VECTOR(0);
                    for (int head_in_idx = 0; head_in_idx < ATTENTION_HEAD_NUM; head_in_idx++)
                    {
                        WEIGHT_VECTOR weight;
                        for (int head_out_idx = 0; head_out_idx < ATTENTION_HEAD_NUM; head_out_idx++)
                        {
                            weight[head_out_idx] = linear_projection_weights[layer_num + 1][head_out_idx][proj_dim_out][head_in_idx][dim_out];
                        }
                        acc += next_node_feature_skip_concat_bias_val[head_in_idx] * weight;
                    }
                    
                    feature_accumulators[node_offset][proj_dim_out] = acc;
                }
                
                FEATURE_MAP_VECTOR projection_result = feature_accumulators[node_offset][dim_out];
                node_embedding_buffer[node_idx % EDGE_PARALLEL_FACTOR][node_idx / EDGE_PARALLEL_FACTOR][dim_out] = projection_result;
                
                source_attention_score_acc += projection_result * source_attention_weights[dim_out_offset];
                target_attention_score_acc += projection_result * target_attention_weights[dim_out_offset];
            }
            
            if (dim_base_idx != 0)
            {
                source_attention_score_acc += source_attention_score_accs[node_offset];
                target_attention_score_acc += target_attention_score_accs[node_offset];
            }

            
            source_attention_score_accs[node_offset] = source_attention_score_acc;
            target_attention_score_accs[node_offset] = target_attention_score_acc;

            
            if (dim_base_idx == ((EMBEDDING_DIM - 1) / APPLY_PARALLEL_FACTOR) * APPLY_PARALLEL_FACTOR)
            {
                
                for (int pe_id = 0; pe_id < EDGE_PARALLEL_FACTOR; pe_id++)
                {
                    source_attention_scores[pe_id][node_idx] = source_attention_score_acc;
                }
                
                target_attention_scores[node_idx % EDGE_PARALLEL_FACTOR][node_idx / EDGE_PARALLEL_FACTOR] = target_attention_score_acc;
            }
        }
    }
}




static void output_node_features(
    FEATURE_MAP_VECTOR feature_accumulators[NODE_PARALLEL_FACTOR][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_score_accs[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR target_attention_score_accs[NODE_PARALLEL_FACTOR],
    FEATURE_MAP_VECTOR node_embedding_buffer[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[EDGE_PARALLEL_FACTOR][MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[EDGE_PARALLEL_FACTOR][ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    int layer_num,
    int node_base_idx,
    int dim_base_idx,
    int node_count
)
{

#pragma HLS INLINE

#pragma HLS ARRAY_PARTITION variable=source_attention_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=source_attention_weights cyclic factor=APPLY_PARALLEL_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=target_attention_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=target_attention_weights cyclic factor=APPLY_PARALLEL_FACTOR dim=3

// source_attention_weights和target_attention_weights：
    WEIGHT_VECTOR source_attention_weights[APPLY_PARALLEL_FACTOR];
#pragma HLS ARRAY_PARTITION variable=source_attention_weights complete dim=0
    WEIGHT_VECTOR target_attention_weights[APPLY_PARALLEL_FACTOR];
#pragma HLS ARRAY_PARTITION variable=target_attention_weights complete dim=0


    for (int dim_offset = 0; dim_offset < APPLY_PARALLEL_FACTOR; dim_offset++)
    {
        int dim = dim_base_idx + dim_offset;
        for (int head_idx = 0; head_idx < ATTENTION_HEAD_NUM; head_idx++)
        {
            source_attention_weights[dim_offset][head_idx] = source_attention_weights[layer_num + 1][head_idx][dim];
            target_attention_weights[dim_offset][head_idx] = target_attention_weights[layer_num + 1][head_idx][dim];
        }
    }

    for (int node_offset = 0; node_offset < NODE_PARALLEL_FACTOR; node_offset++)
    {
        int node_idx = node_base_idx + node_offset;
        if (node_idx < node_count)
        {
            FEATURE_MAP_VECTOR source_attention_score_acc = FEATURE_MAP_TYPE(0);
            FEATURE_MAP_VECTOR target_attention_score_acc = FEATURE_MAP_TYPE(0);
            for (int dim_offset = 0; dim_offset < APPLY_PARALLEL_FACTOR; dim_offset++)
            {
                int dim = dim_base_idx + dim_offset;
                FEATURE_MAP_VECTOR projection_result = feature_accumulators[node_offset][dim];
                node_embedding_buffer[node_idx % EDGE_PARALLEL_FACTOR][node_idx / EDGE_PARALLEL_FACTOR][dim] = projection_result;
                
                source_attention_score_acc += projection_result * source_attention_weights[dim_offset];
                target_attention_score_acc += projection_result * target_attention_weights[dim_offset];
            }

            if (dim_base_idx != 0)
            {
                source_attention_score_acc += source_attention_score_accs[node_offset];
                target_attention_score_acc += target_attention_score_accs[node_offset];
            }

            source_attention_score_accs[node_offset] = source_attention_score_acc;
            target_attention_score_accs[node_offset] = target_attention_score_acc;

            if (dim_base_idx == ((EMBEDDING_DIM - 1) / APPLY_PARALLEL_FACTOR) * APPLY_PARALLEL_FACTOR)
            {
                for (int pe_id = 0; pe_id < EDGE_PARALLEL_FACTOR; pe_id++)
                {
                    source_attention_scores[pe_id][node_idx] = source_attention_score_acc;
                }
                target_attention_scores[node_idx % EDGE_PARALLEL_FACTOR][node_idx / EDGE_PARALLEL_FACTOR] = target_attention_score_acc;
            }
        }
    }
}