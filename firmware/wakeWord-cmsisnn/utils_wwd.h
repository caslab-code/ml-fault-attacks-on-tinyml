// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __UTILS_H__
#define __UTILS_H__

#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

int8_t** allocate_matrix_wwd(int rows, int cols);
void free_matrix_wwd(int8_t** matrix, int rows);
int32_t** allocate_matrix32_wwd(int rows, int cols);
void free_matrix32_wwd(int32_t** matrix, int rows);
void random_matrix_wwd(int8_t** matrix, int rows, int cols);
void fill_matrix_wwd(int8_t** matrix, int rows, int cols, const int8_t* const input_arr);
void fill_matrix_transpose_wwd(int8_t** matrix, int rows, int cols, const int8_t* const input_arr);
void print_matrix_wwd(int8_t** matrix, int rows, int cols);
void subtract_matrix_wwd(int8_t** A, int8_t** B, int32_t** C, int rows, int cols);
void subtract_matrix_with_offset_wwd(int8_t** A, int8_t** B, int32_t** C, int rows, int cols, int32_t offset_A);
void matrix_multiplication_wwd(int8_t** A, int n, int m, int8_t** B, int p, int32_t** result);
void matrix_multiplication32_wwd_1(int32_t** A, int n, int m, int8_t** B, int p, int32_t** result);
void matrix_multiplication32_wwd_2(int8_t** A, int n, int m, int32_t** B, int p, int32_t** result);
void matrix_multiplication32_wwd_3(int32_t** A, int n, int m, int32_t** B, int p, int32_t**result);
int compare_matrices_wwd(int8_t** A, int8_t** B, int rows, int cols);
int32_t calculate_median_wwd(int32_t* values, int size);
void majority_vote_wwd(int32_t*** results, int num_results, int rows, int cols, int32_t* final_result);
void self_correct_matrix_multiplication_wwd(int8_t* A, int n, int m, int8_t* B, int p, int8_t beta, int32_t* const result, int8_t alpha_scale, int8_t beta_scale, int iter, const int32_t* bias, int32_t input_offset, int32_t weight_offset, int32_t quant_multiplier, int32_t quant_shift, int32_t output_offset, int8_t clamp_min, int8_t clamp_max);
void protectedVecmat(const int8_t* const vector, const int8_t* const matrix, unsigned ncols_A, unsigned ncols_B, int8_t alpha_scale, int8_t beta_scale, int32_t* const ret, int iter, const int32_t* bias, int32_t input_offset, int32_t weight_offset, int32_t quant_multiplier, int32_t quant_shift, int32_t output_offset, int8_t clamp_min, int8_t clamp_max);
arm_cmsis_nn_status protected_fully_connected(const int8_t* const vector, const int8_t* const matrix, unsigned ncols_A, unsigned ncols_B, int8_t alpha_scale, int8_t beta_scale, int32_t* const ret, int iter, const int32_t* bias, int32_t input_offset, int32_t weight_offset, int32_t quant_multiplier, int32_t quant_shift, int32_t output_offset, int8_t clamp_min, int8_t clamp_max);
#endif