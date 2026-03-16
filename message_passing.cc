#include "message_passing.h"
#include "hls_math.h"

typedef struct {
    int node_idx;
    int degree;
} node_t;

// #region Internal Function Declarations
static void read_node_degrees(
    int pe_id,
    hls::stream<int>& degrees,
    hls::stream<node_t>& nonzero_degree_nodes,
    hls::stream<node_t>& nonzero_degree_nodes2,
    int node_count
);
static void top_k_attention_accumulation(
    hls::stream<FEATURE_MAP_VECTOR>& threshold_score,
    int pe_id,
    hls::stream<node_t>& nonzero_degree_nodes,
    FEATURE_MAP_VECTOR node_embedding_buffer[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)]
);
static void gather_node_messages(
    hls::stream<FEATURE_MAP_VECTOR>& threshold_score,
    int pe_id,
    hls::stream<node_t>& nonzero_degree_nodes,
    FEATURE_MAP_VECTOR node_embedding_buffer[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    hls::stream<msg_passing_output_t>& messages,
    hls::stream<FEATURE_MAP_VECTOR>& attention_score_sums
);
static void expand_node_messages(
    hls::stream<msg_passing_output_t>& messages_per_nz_deg_node,
    hls::stream<FEATURE_MAP_VECTOR>& attention_score_sums_per_nz_deg_node,
    hls::stream<int>& degrees,
    hls::stream<msg_passing_output_t> messages_per_node[NODE_PARALLEL_FACTOR],
    hls::stream<FEATURE_MAP_VECTOR> attention_score_sums_per_node[NODE_PARALLEL_FACTOR],
    int node_count
);
// #endregion


void message_passing_pe(
    int pe_id,
    FEATURE_MAP_VECTOR node_embedding_buffer[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    hls::stream<msg_passing_output_t> messages[NODE_PARALLEL_FACTOR],
    hls::stream<FEATURE_MAP_VECTOR> attention_score_sums[NODE_PARALLEL_FACTOR],
    int node_count
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    
    hls::stream<int> degrees("degrees");
#pragma HLS STREAM variable=degrees depth=20
    hls::stream<node_t> nonzero_degree_nodes("nonzero_degree_nodes");
#pragma HLS STREAM variable=nonzero_degree_nodes depth=20
    hls::stream<node_t> nonzero_degree_nodes2("nonzero_degree_nodes2");
#pragma HLS STREAM variable=nonzero_degree_nodes2 depth=20
    hls::stream<msg_passing_output_t> messages_per_nz_deg_node("messages_per_nz_deg_node");
#pragma HLS STREAM variable=messages_per_nz_deg_node depth=(20 * ceildiv(EMBEDDING_DIM, GATHER_PARALLEL_FACTOR))
    hls::stream<FEATURE_MAP_VECTOR> attention_score_sums_per_nz_deg_node("attention_score_sums_per_nz_deg_node");
#pragma HLS STREAM variable=attention_score_sums_per_nz_deg_node depth=20
    hls::stream<FEATURE_MAP_VECTOR> threshold_score("threshold_score");
#pragma HLS STREAM variable=threshold_score depth=20
    read_node_degrees(pe_id, degrees, nonzero_degree_nodes, nonzero_degree_nodes2, node_count);
    top_k_attention_accumulation(threshold_score, pe_id, nonzero_degree_nodes, node_embedding_buffer, source_attention_scores, target_attention_scores);
    gather_node_messages(threshold_score, pe_id, nonzero_degree_nodes2, node_embedding_buffer, source_attention_scores, target_attention_scores, messages_per_nz_deg_node, attention_score_sums_per_nz_deg_node);
    expand_node_messages(messages_per_nz_deg_node, attention_score_sums_per_nz_deg_node, degrees, messages, attention_score_sums, node_count);
}


static void read_node_degrees(
    int pe_id,
    hls::stream<int>& degrees,
    hls::stream<node_t>& nonzero_degree_nodes,
    hls::stream<node_t>& nonzero_degree_nodes2,
    int node_count
)
{
#pragma HLS INLINE off

    for (int node_idx = 0; node_idx < node_count; node_idx++)
    {
#pragma HLS LOOP_TRIPCOUNT min=GRAPH_ANALYSIS_MIN_NODE_NUM max=GRAPH_ANALYSIS_MAX_NODE_NUM avg=GRAPH_ANALYSIS_AVG_NODE_NUM
        int degree = parallel_degree_tables[pe_id][node_idx];
        degrees << degree;
        if (degree != 0)
        {
            nonzero_degree_nodes << node_t{node_idx, degree};
            nonzero_degree_nodes2 << node_t{node_idx, degree};
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////
void mySwap(Neighbor arr[], int i, int j){
#pragma HLS INLINE
    Neighbor temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}


int partition(Neighbor arr[], int low, int high) {
#pragma HLS INLINE
    FEATURE_MAP_VECTOR pivot = arr[high].score;  
    int i = low - 1;
    for (int j = low; j < high; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=MAX_NODE_COUNT avg=ceildiv(MAX_NODE_COUNT, 4*A)
#pragma HLS PIPELINE
        if (arr[j].score > pivot) {  
            i++;
            mySwap(arr, i, j);
        }
    }
    mySwap(arr, i + 1, high);
    return i + 1;
}


int quickSelect(Neighbor arr[], int low, int high) {
    // 文章录用上传
}
static void top_k_attention_accumulation(
    hls::stream<FEATURE_MAP_VECTOR>& threshold_score,
    int pe_id,
    hls::stream<node_t>& nonzero_degree_nodes,
    FEATURE_MAP_VECTOR node_embedding_buffer[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)]
)
{
    // 文章录用上传
}



static void gather_node_messages(
    hls::stream<FEATURE_MAP_VECTOR>& threshold_score,
    int pe_id,
    hls::stream<node_t>& nonzero_degree_nodes,
    FEATURE_MAP_VECTOR node_embedding_buffer[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)][EMBEDDING_DIM],
    FEATURE_MAP_VECTOR source_attention_scores[MAX_NODE_COUNT],
    FEATURE_MAP_VECTOR target_attention_scores[ceildiv(MAX_NODE_COUNT, EDGE_PARALLEL_FACTOR)],
    hls::stream<msg_passing_output_t>& messages,
    hls::stream<FEATURE_MAP_VECTOR>& attention_score_sums
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    msg_passing_output_t msg_passing_outs[ceildiv(EMBEDDING_DIM, GATHER_PARALLEL_FACTOR)];
    FEATURE_MAP_VECTOR attention_score_sums_acc;
    int target_node_idx = 0;
    int edge_start = 0;
    int edge_end = 0;
    int edge_count = edge_count_per_pe[pe_id];

    
    FEATURE_MAP_VECTOR threshold;
    for (int edge_idx = 0; edge_idx < edge_count; edge_idx++)
    {
#pragma HLS LOOP_TRIPCOUNT min=0 max=GRAPH_ANALYSIS_MAX_EDGE_NUM avg=ceildiv(GRAPH_ANALYSIS_AVG_EDGE_NUM, EDGE_PARALLEL_FACTOR)

        
        int source_node_idx = parallel_neighbor_tables2[pe_id][edge_idx];

        
        for (int iter_idx = 0, dim_base = 0; iter_idx < ceildiv(EMBEDDING_DIM, GATHER_PARALLEL_FACTOR); iter_idx++, dim_base += GATHER_PARALLEL_FACTOR)
        {
#pragma HLS PIPELINE

            if (edge_idx >= edge_end)
            {
                node_t node;
                nonzero_degree_nodes >> node;
                target_node_idx = node.node_idx;
                edge_start = edge_idx;
                edge_end = edge_idx + node.degree;
                attention_score_sums_acc = FEATURE_MAP_TYPE(0);
                threshold_score >> threshold;
            }

            FEATURE_MAP_VECTOR attention_scores = source_attention_scores[target_node_idx] + target_attention_scores[source_node_idx];
            
            for (int head_idx = 0; head_idx < ATTENTION_HEAD_NUM; head_idx++)
            {
#pragma HLS UNROLL
                FEATURE_MAP_TYPE score = attention_scores[head_idx];
                if (score < 0) score = score * FEATURE_MAP_TYPE(0.2);
                attention_scores[head_idx] = hls::exp(score);
            }


            if (attention_scores < threshold){
                attention_scores = FEATURE_MAP_VECTOR(0);
            }
            if (iter_idx == 0) attention_score_sums_acc += attention_scores;

            msg_passing_output_t msg_passing_out = (edge_idx != edge_start) ? msg_passing_outs[iter_idx] : FEATURE_MAP_VECTOR(0);
            
            for (int dim_offset = 0; dim_offset < GATHER_PARALLEL_FACTOR; dim_offset++)
            {
#pragma HLS UNROLL
                int dim = dim_base + dim_offset;
                if (dim < EMBEDDING_DIM)
                {
                    msg_passing_out[dim_offset] += attention_scores * node_embedding_buffer[source_node_idx][dim];
                }
            }
            
            msg_passing_outs[iter_idx] = msg_passing_out;

            if (edge_idx + 1 == edge_end)
            {
                messages << msg_passing_out;
                if (iter_idx == 0) attention_score_sums << attention_score_sums_acc;
            }
        }
    }
}


static void expand_node_messages(
    hls::stream<msg_passing_output_t>& messages_per_nz_deg_node,
    hls::stream<FEATURE_MAP_VECTOR>& attention_score_sums_per_nz_deg_node,
    hls::stream<int>& degrees,
    hls::stream<msg_passing_output_t> messages_per_node[NODE_PARALLEL_FACTOR],
    hls::stream<FEATURE_MAP_VECTOR> attention_score_sums_per_node[NODE_PARALLEL_FACTOR],
    int node_count
)
{
#pragma HLS INLINE off

    int degree;
    
    for (int node_idx = 0; node_idx < node_count; node_idx++)
    {
#pragma HLS LOOP_TRIPCOUNT min=GRAPH_ANALYSIS_MIN_NODE_NUM max=GRAPH_ANALYSIS_MAX_NODE_NUM avg=GRAPH_ANALYSIS_AVG_NODE_NUM
        
        for (int iter_idx = 0; iter_idx < ceildiv(EMBEDDING_DIM, GATHER_PARALLEL_FACTOR); iter_idx++)
        {

#pragma HLS PIPELINE II=1

            
            if (iter_idx == 0)
            {
#pragma HLS OCCURRENCE cycle=ceildiv(EMBEDDING_DIM, GATHER_PARALLEL_FACTOR)
                
                degrees >> degree;

                FEATURE_MAP_VECTOR attention_score_sum;
                if (degree != 0)
                {
                    attention_score_sums_per_nz_deg_node >> attention_score_sum;
                }
                else
                {
                    attention_score_sum = FEATURE_MAP_TYPE(0);
                }
                
                attention_score_sums_per_node[node_idx % NODE_PARALLEL_FACTOR] << attention_score_sum;
            }

            msg_passing_output_t message;
            
            if (degree != 0)
            {
                messages_per_nz_deg_node >> message;
            }
            else
            {
                message = FEATURE_MAP_VECTOR(0);
            }
            
            messages_per_node[node_idx % NODE_PARALLEL_FACTOR] << message;
        }
    }
}