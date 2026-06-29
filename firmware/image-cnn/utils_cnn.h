// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __UTILS_H__
#define __UTILS_H__

#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

arm_cmsis_nn_status protected_convolution(
    const int8_t*  matrix,          // [H×W×Cin]
    int             matrix_height,
    int             matrix_width,
    int             input_channel,
    const int8_t*  kernel,          // [Kh×Kw×Cin×Cout]
    int             kernel_height,
    int             kernel_width,
    int             output_channel,
    int             pad_height,
    int             pad_width,
    int             stride_height,
    int             stride_width,
    int             dilation_height,
    int             dilation_width,
    const int32_t*  bias,           // [output_channel] or NULL
    int32_t         input_offset,
    const int32_t*  quant_multiplier, // [output_channel]
    const int32_t*  quant_shift,      // [output_channel]
    int32_t         output_offset,
    int8_t          clamp_min,
    int8_t          clamp_max,
    int32_t*        output          // [H_out×W_out×Cout]
);

arm_cmsis_nn_status protected_maxpool(
    const int8_t* input,
    int input_height, 
    int input_width, 
    int input_channel,
    int kernel_height, 
    int kernel_width,
    int stride_height, 
    int stride_width,
    int pad_height,
    int pad_width,
    int32_t activation_min,
    int32_t activation_max,
    int32_t* output  
);

#endif