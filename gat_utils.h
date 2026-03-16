#ifndef __GAT_UTILITIES_H__
#define __GAT_UTILITIES_H__

#include "hls_math.h"

/**
 * @brief 向上取整除法（ceil division）
 * @param dividend 被除数
 * @param divisor 除数
 * @return 向上取整后的除法结果
 */
template <typename T>
static constexpr T ceil_division(T dividend, T divisor)
{
#pragma HLS INLINE
    return (dividend + divisor - 1) / divisor;
}

/**
 * @brief 向上取整到除数的整数倍（round up to multiple）
 * @param dividend 目标值
 * @param divisor 基数（除数）
 * @return 大于等于目标值的最小基数整数倍
 */
template <typename T>
static constexpr T round_up_to_multiple(T dividend, T divisor)
{
#pragma HLS INLINE
    return ceil_division(dividend, divisor) * divisor;
}

/**
 * @brief 获取两个值中的最小值
 * @param a 第一个值
 * @param b 第二个值
 * @return 较小的那个值
 */
template <typename T>
static constexpr T min_value(T a, T b)
{
#pragma HLS INLINE
    return (a < b) ? a : b;
}

/**
 * @brief 获取两个值中的最大值
 * @param a 第一个值
 * @param b 第二个值
 * @return 较大的那个值
 */
template <typename T>
static constexpr T max_value(T a, T b)
{
#pragma HLS INLINE
    return (a > b) ? a : b;
}

/**
 * @brief 适用于ap_fixed类型的ReLU激活函数
 * @param x 输入值
 * @return ReLU激活后的结果（x>0返回x，否则返回0）
 */
template <typename T>
static constexpr T ap_fixed_relu(T x)
{
#pragma HLS INLINE
    return hls::signbit(x) ? T(0) : x;
}

/**
 * @brief 获取ap_fixed类型的最小精度值（epsilon）
 * @return ap_fixed类型的最小可表示正数
 */
template <typename T>
static constexpr T ap_fixed_epsilon()
{
#pragma HLS INLINE
    return T(1.0 / (1 << (T::width - T::iwidth)));
}

#endif