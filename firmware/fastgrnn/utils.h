// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __UTILS_H__
#define __UTILS_H__

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

float min(float a, float b);
float max(float a, float b);

float relu(float x);
float sigmoid(float x);
float tanhyperbolic(float x);
float quantTanh(float x);
float quantSigmoid(float x);

void v_relu(const float* const vec, unsigned len, float* const ret);
void v_sigmoid(const float* const vec, unsigned len, float* const ret);
void v_tanh(const float* const vec, unsigned len, float* const ret);
void v_quantSigmoid(const float* const vec, unsigned len, float* const ret);
void v_quantTanh(const float* const vec, unsigned len, float* const ret);

/* Scaled matrix-vector multiplication:  ret = alpha * ret + beta * mat * vec
   alpha and beta are scalars
   ret is of size nrows, vec is of size ncols
   mat is of size nrows * ncols, stored in row major */
void matVec(const float* const mat, const float* const vec,
  unsigned nrows, unsigned ncols,
  float alpha, float beta,
  float* const ret);

// scaled vector addition: ret = scalar1 * vec1 + scalar2 * vector2
void v_add(float scalar1, const float* const vec1,
  float scalar2, const float* const vec2,
  unsigned len, float* const ret);

// point-wise vector multiplication ret = vec2 * vec1 
void v_mult(const float* const vec1, const float* const vec2,
  unsigned len, float* const ret);

// point-wise vector division ret = vec2 / vec1 
void v_div(const float* const vec1, const float* const vec2,
  unsigned len, float* const ret);

// Return squared Euclidean distance between vec1 and vec2
float l2squared(const float* const vec1,
  const float* const vec2, unsigned dim);

// Return index with max value, if tied, return first tied index.
unsigned argmax(const float* const vec, unsigned len);

// ret[i] = exp(input[i]) / \sum_i exp(input[i])
void softmax(const float* const input, unsigned len, float* const ret);

void protectedmatVec(const float* const matrix, const float* const vector, unsigned nrows, unsigned ncols, float alpha_scale, float beta_scale, float* const ret, int iter);
float** allocate_matrix(int rows, int cols);
void free_matrix(float** matrix, int rows);
void random_matrix(float** matrix, int rows, int cols);
void fill_matrix(float** matrix, int rows, int cols, const float* const input_arr);
void subtract_matrix(float** A, float** B, float** C, int rows, int cols);
void matrix_multiplication(float** A, int n, int m, float** B, int p, float** C);
float calculate_median(float* values, int size);
void majority_vote(float*** results, int num_results, int rows, int cols, float* final_result);
void self_correct_matrix_multiplication(float** A, int n, int m, float** B, int p, float beta, float* const result, float alpha_scale, float beta_scale, int iter);
int compare_matrices(float** A, float** B, int rows, int cols);
#endif
