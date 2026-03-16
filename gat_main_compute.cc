#include "gat_define.h"
#include "gat_input_loader.h"
#include "conv_layer.h"
#include "finalize.h"
#include <chrono>

extern "C" {
void GraphAttentionNetwork_ProcessGraphs(
    int graph_count,
    int* node_count_per_graph,
    int* edge_count_per_graph,
    int* weight_reload_flag,
    FEATURE_MAP_TYPE output_data[][TASK_NUM],
    node_feature_struct_t* input_node_features,
    edge_struct_t* input_edge_list,
    WEIGHT_TYPE target_attention_weights_in[][NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE source_attention_weights_in[][NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE linear_projection_weights_in[][NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE skip_projection_weights_in[][NETWORK_LAYER_NUM][ATTENTION_HEAD_NUM][EMBEDDING_DIM][ATTENTION_HEAD_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_weights_in[][TASK_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE graph_prediction_bias_in[][TASK_NUM]
)
{
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE m_axi depth=(1) port=node_count_per_graph offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=edge_count_per_graph offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=weight_reload_flag offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=output_data offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(500) port=input_node_features offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(500) port=input_edge_list offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=target_attention_weights_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=source_attention_weights_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=linear_projection_weights_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=skip_projection_weights_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graph_prediction_weights_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graph_prediction_bias_in offset=slave bundle=mem

#pragma HLS AGGREGATE variable=node_embedding_ping_buffer
#pragma HLS AGGREGATE variable=node_embedding_pong_buffer
#pragma HLS AGGREGATE variable=node_feature_skip_concat_bias_ping
#pragma HLS AGGREGATE variable=node_feature_skip_concat_bias_pong
#pragma HLS AGGREGATE variable=source_attention_scores_ping
#pragma HLS AGGREGATE variable=source_attention_scores_pong
#pragma HLS AGGREGATE variable=target_attention_scores_ping
#pragma HLS AGGREGATE variable=target_attention_scores_pong

    for (int graph_idx = 0, weight_index = -1, node_offset = 0, edge_offset = 0; graph_idx < graph_count; graph_idx++)
    {
#pragma HLS LOOP_TRIPCOUNT min=GRAPH_ANALYSIS_NUM max=GRAPH_ANALYSIS_NUM avg=GRAPH_ANALYSIS_NUM
        int node_count = node_count_per_graph[graph_idx];
        int edge_count = edge_count_per_graph[graph_idx];
        bool reload_weights_for_graph = weight_reload_flag[graph_idx];

        if (reload_weights_for_graph)
        {
            weight_index++;
            load_weights(
                target_attention_weights_in[weight_index],
                source_attention_weights_in[weight_index],
                linear_projection_weights_in[weight_index],
                skip_projection_weights_in[weight_index],
                graph_prediction_weights_in[weight_index],
                graph_prediction_bias_in[weight_index]
            );
        }
        load_graph(
            &input_edge_list[edge_offset],
            node_count,
            edge_count
        );
        load_input_node_embeddings(input_node_features, node_count);
        auto startTime = std::chrono::high_resolution_clock::now();
        for (int layer_idx = 0; layer_idx < NETWORK_LAYER_NUM; layer_idx++)
        {
            if (layer_idx % 2 == 0)
                compute_convolution_layer(
                    layer_idx,
                    node_embedding_ping_buffer,
                    node_embedding_pong_buffer,
                    source_attention_scores_ping,
                    source_attention_scores_pong,
                    target_attention_scores_ping,
                    target_attention_scores_pong,
                    node_feature_skip_concat_bias_ping,
                    node_feature_skip_concat_bias_pong,
                    output_data[graph_idx],
                    node_count
                );
            else
                compute_convolution_layer(
                    layer_idx,
                    node_embedding_pong_buffer,
                    node_embedding_ping_buffer,
                    source_attention_scores_pong,
                    source_attention_scores_ping,
                    target_attention_scores_pong,
                    target_attention_scores_ping,
                    node_feature_skip_concat_bias_pong,
                    node_feature_skip_concat_bias_ping,
                    output_data[graph_idx],
                    node_count
                );
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        std::cout<<std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()<<"ms"<<std::endl;

        node_offset += node_count;
        edge_offset += edge_count;
    }
}
}