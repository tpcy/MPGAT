#ifndef __LINEAR_LAYER_H__
#define __LINEAR_LAYER_H__

#include "gat_define.h"
#include "hls_stream.h"

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU = true
>
void linear(
    FEATURE_MAP_TYPE input[DIM_IN],
    WEIGHT_TYPE weight[DIM_OUT][DIM_IN],
    WEIGHT_TYPE bias[DIM_OUT],
    FEATURE_MAP_TYPE output[DIM_OUT]
);

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU = true
>
void linear_output_stationary(
    FEATURE_MAP_TYPE input[DIM_IN],
    WEIGHT_TYPE weight[DIM_OUT][DIM_IN],
    WEIGHT_TYPE bias[DIM_OUT],
    hls::stream<std::array<FEATURE_MAP_TYPE, PARALLEL>>& output
);

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU = true
>
void linear_input_stationary(
    hls::stream<std::array<FEATURE_MAP_TYPE, PARALLEL>>& input,
    WEIGHT_TYPE weight[DIM_OUT][DIM_IN],
    WEIGHT_TYPE bias[DIM_OUT],
    FEATURE_MAP_TYPE output[DIM_OUT]
);

#endif