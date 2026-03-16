#include "linear.h"

using std::array;

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU
>
void linear(
    FEATURE_MAP_TYPE input[DIM_IN],
    WEIGHT_TYPE weight[DIM_OUT][DIM_IN],
    WEIGHT_TYPE bias[DIM_OUT],
    FEATURE_MAP_TYPE output[DIM_OUT]
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=PARALLEL dim=1

    for (int dim_out_base = 0; dim_out_base < DIM_OUT; dim_out_base += PARALLEL)
    {
#pragma HLS PIPELINE II=1
        for (int dim_out_offset = 0; dim_out_offset < PARALLEL; dim_out_offset++)
        {
#pragma HLS UNROLL
            int dim_out = dim_out_base + dim_out_offset;
            FEATURE_MAP_TYPE output_elem = 0;

            if (dim_out < DIM_OUT)
            {
                output_elem = bias[dim_out];
                for (int dim_in = 0; dim_in < DIM_IN; dim_in++)
                {
#pragma HLS UNROLL
                    output_elem += input[dim_in] * weight[dim_out][dim_in];
                }
            }

            if (RELU && hls::signbit(output_elem)) output_elem = 0;
            output[dim_out] = output_elem;
        }
    }
}

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU
>
void linear_output_stationary(
    FEATURE_MAP_TYPE input[DIM_IN],
    WEIGHT_TYPE weight[DIM_OUT][DIM_IN],
    WEIGHT_TYPE bias[DIM_OUT],
    hls::stream<array<FEATURE_MAP_TYPE, PARALLEL>>& output
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=PARALLEL dim=1

    for (int dim_out_base = 0; dim_out_base < DIM_OUT; dim_out_base += PARALLEL)
    {
#pragma HLS PIPELINE II=1
        array<FEATURE_MAP_TYPE, PARALLEL> output_slice;
        for (int dim_out_offset = 0; dim_out_offset < PARALLEL; dim_out_offset++)
        {
#pragma HLS UNROLL
            int dim_out = dim_out_base + dim_out_offset;
            FEATURE_MAP_TYPE output_elem = 0;

            if (dim_out < DIM_OUT)
            {
                output_elem = bias[dim_out];
                for (int dim_in = 0; dim_in < DIM_IN; dim_in++)
                {
#pragma HLS UNROLL
                    output_elem += input[dim_in] * weight[dim_out][dim_in];
                }
            }

            if (RELU && hls::signbit(output_elem)) output_elem = 0;
            output_slice[dim_out_offset] = output_elem;
        }
        output << output_slice;
    }
}

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU
>
void linear_input_stationary(
    hls::stream<array<FEATURE_MAP_TYPE, PARALLEL>>& input,
    WEIGHT_TYPE weight[DIM_OUT][DIM_IN],
    WEIGHT_TYPE bias[DIM_OUT],
    FEATURE_MAP_TYPE output[DIM_OUT]
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=output complete dim=1

    for (int dim_out = 0; dim_out < DIM_OUT; dim_out++)
    {
#pragma HLS UNROLL
        output[dim_out] = bias[dim_out];
    }

    for (int dim_in_base = 0; dim_in_base < DIM_IN; dim_in_base += PARALLEL)
    {
#pragma HLS PIPELINE II=1
        array<FEATURE_MAP_TYPE, PARALLEL> input_slice;
        input >> input_slice;
        for (int dim_out = 0; dim_out < DIM_OUT; dim_out++)
        {
#pragma HLS UNROLL
            FEATURE_MAP_TYPE additive_term = 0;
            for (int dim_in_offset = 0; dim_in_offset < PARALLEL; dim_in_offset++)
            {
#pragma HLS UNROLL
                int dim_in = dim_in_base + dim_in_offset;
                FEATURE_MAP_TYPE input_elem = input_slice[dim_in_offset];
                if (dim_in < DIM_IN)
                {
                    additive_term += input_elem * weight[dim_out][dim_in];
                }
            }
            output[dim_out] += additive_term;
        }
    }

    for (int dim_out = 0; dim_out < DIM_OUT; dim_out++)
    {
#pragma HLS UNROLL
        if (RELU && hls::signbit(output[dim_out])) output[dim_out] = 0;
    }
}

// #region Template Instantiations

// From finalize.cc
template void linear<EMBEDDING_DIM, TASK_NUM, TASK_NUM, false>(
    FEATURE_MAP_TYPE input[EMBEDDING_DIM],
    WEIGHT_TYPE weight[TASK_NUM][EMBEDDING_DIM],
    WEIGHT_TYPE bias[TASK_NUM],
    FEATURE_MAP_TYPE output[TASK_NUM]
);

// #endregion