// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "utils.h"

#define EPSILON 1e-6 // Tolerance for floating-point comparison

float min(float a, float b) {
  return (a < b) ? a : b;
}

float max(float a, float b) {
  return (a > b) ? a : b;
}

float relu(float x) {
  if (x < 0.0) return 0.0;
  else return x;
}

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-1.0f * x));
}

float tanhyperbolic(float x) {
  float ex = expf(x);
  float enx = expf(-1.0f * x);
  return (ex - enx) / (ex + enx);
}

float quantTanh(float x) {
  return max(min(x, 1.0f), -1.0f);
}

float quantSigmoid(float x) {
  return max(min((x + 1.0f) / 2.0f, 1.0f), 0.0f);
}

void v_relu(const float* const vec, unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = relu(vec[i]);
}

void v_sigmoid(const float* const vec, unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = sigmoid(vec[i]);
}

void v_tanh(const float* const vec, unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = tanhyperbolic(vec[i]);
}

void v_quantSigmoid(const float* const vec, unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = sigmoid(vec[i]);
}

void v_quantTanh(const float* const vec, unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = tanh(vec[i]);
}

void matVec(const float* const mat, const float* const vec,
  unsigned nrows, unsigned ncols,
  float alpha, float beta,
  float* const ret) {

  for (unsigned row = 0; row < nrows; row++) {
    float sum = 0.0f;
    float* mat_offset = (float*)mat + row * ncols;
    for (unsigned col = 0; col < ncols; col++) {
      sum += *mat_offset++ * vec[col];
    }
    ret[row] = alpha * ret[row] + beta * sum;
  }
}

void v_add(float scalar1, const float* const vec1,
  float scalar2, const float* const vec2,
  unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++)
    ret[i] = scalar1 * vec1[i] + scalar2 * vec2[i];
}

void v_mult(const float* const vec1, const float* const vec2,
  unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++)
    ret[i] = vec1[i] * vec2[i];
}

void v_div(const float* const vec1, const float* const vec2,
  unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++)
    ret[i] = vec2[i] / vec1[i];
}

float l2squared(const float* const vec1,
  const float* const vec2, unsigned dim) {
  float sum = 0.0f;
  for (unsigned i = 0; i < dim; i++)
    sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
  return sum;
}

unsigned argmax(const float* const vec, unsigned len) {
  unsigned maxId = 0;
  float maxScore = FLT_MIN;
  for (unsigned i = 0; i < len; i++) {
    if (vec[i] > maxScore) {
      maxScore = vec[i];
      maxId = i;
    }
  }
  return maxId;
}

void softmax(const float* const input, unsigned len, float* const ret) {
  float m = input[argmax(input, len)];
  float sum = 0.0f;
  for (unsigned i = 0; i < len; i++)
    sum += expf(input[i] - m);

  float offset = m + logf(sum);
  for (unsigned i = 0; i < len; i++)
    ret[i] = expf(input[i] - offset);
}

// Function to allocate memory for a matrix
float** allocate_matrix(int rows, int cols) {
  float** matrix = (float**)malloc(rows * sizeof(float*));
  for (int i = 0; i < rows; i++) {
      matrix[i] = (float*)malloc(cols * sizeof(float));
  }
  return matrix;
}

// Function to free memory for a matrix
void free_matrix(float** matrix, int rows) {
  for (int i = 0; i < rows; i++) {
      free(matrix[i]);
  }
  free(matrix);
}

// Function to initialize a matrix with random float values
void random_matrix(float** matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          matrix[i][j] = ((float)rand() / RAND_MAX) * 10.0; // Random values [0, 10)
      }
  }
}

// Function to initialize a matrix with float values from array
void fill_matrix(float** matrix, int rows, int cols, const float* const input_arr){
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          matrix[i][j] = input_arr[i * cols + j];
      }
  }
}


// Function to subtract matrices: C = A - B
void subtract_matrix(float** A, float** B, float** C, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          C[i][j] = A[i][j] - B[i][j];
      }
  }
}

// Reliable matrix multiplication: C = A * B
void matrix_multiplication(float** A, int n, int m, float** B, int p, float** C) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      C[i][j] = 0.0;
      for (int k = 0; k < m; k++) {
         C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// Function to compare two matrices with a tolerance for floating-point errors
int compare_matrices(float** A, float** B, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (fabs(A[i][j] - B[i][j]) > EPSILON) {
        return 0; // Matrices do not match
      }
    }
  }
  return 1; // Matrices match
}

// Function to calculate the median of an array
float calculate_median(float* values, int size) {
  for (int i = 0; i < size - 1; i++) {
    for (int j = 0; j < size - i - 1; j++) {
      if (values[j] > values[j + 1]) {
        float temp = values[j];
        values[j] = values[j + 1];
        values[j + 1] = temp;
      }
    }
  }
  return (size % 2 == 0) ? (values[size / 2 - 1] + values[size / 2]) / 2.0 : values[size / 2];
}

// Majority voting algorithm using median
void majority_vote(float*** results, int num_results, int rows, int cols, float* final_result) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float* values = (float*)malloc(num_results * sizeof(float));
      for (int k = 0; k < num_results; k++) {
        values[k] = results[k][i][j];
      }
      final_result[i] = calculate_median(values, num_results);
      free(values);
    }
  }
}

// Self-correcting matrix multiplication
void self_correct_matrix_multiplication(float** A, int n, int m, float** B, int p, float beta, float* const result, float alpha_scale, float beta_scale, int iter) {

  int num_repeats = 5;
  float*** answers = (float***)malloc(num_repeats * sizeof(float**));
  for (int i = 0; i < num_repeats; i++) answers[i] = allocate_matrix(n, p);

  float** A1 = allocate_matrix(n, m);
  float** A2 = allocate_matrix(n, m);
  float** B1 = allocate_matrix(m, p);
  float** B2 = allocate_matrix(m, p);
  float** temp1 = allocate_matrix(n, p);
  float** temp2 = allocate_matrix(n, p);
  float** temp3 = allocate_matrix(n, p);
  float** temp4 = allocate_matrix(n, p);

  for (int r = 0; r < num_repeats; r++) {
    
    random_matrix(A1, n, m);
    random_matrix(B1, m, p);
    subtract_matrix(A, A1, A2, n, m);
    subtract_matrix(B, B1, B2, m, p);

    matrix_multiplication(A1, n, m, B1, p, temp1);
    matrix_multiplication(A2, n, m, B1, p, temp2);
    matrix_multiplication(A1, n, m, B2, p, temp3);
    matrix_multiplication(A2, n, m, B2, p, temp4);
    
    for (int i = 0; i < n; i++){
      for (int j = 0; j < p; j++){
        answers[r][i][j] = temp1[i][j] + temp2[i][j] + temp3[i][j] + temp4[i][j];
        if (iter == 0 && alpha_scale == 1.0 && beta_scale == 1.0) {
          answers[r][i][j]  =  alpha_scale * result[i * p + j];
        } else if (iter > 0 && alpha_scale == 1.0 && beta_scale == 1.0){
          answers[r][i][j]  = alpha_scale * result[i * p + j] + beta_scale * answers[r][i][j];
        } 
      }
    }
  }

  majority_vote(answers, num_repeats, n, p, result);

  for (int i = 0; i < num_repeats; i++) free_matrix(answers[i], n);
  free(answers);
  free_matrix(A1, n);
  free_matrix(A2, n);
  free_matrix(B1, m);
  free_matrix(B2, m);
  free_matrix(temp1, n);
  free_matrix(temp2, n);
  free_matrix(temp3, n);
  free_matrix(temp4, n);
}

void protectedmatVec(const float* const matrix, const float* const vector, unsigned nrows, unsigned ncols, float alpha_scale, float beta_scale, float* const ret, int iter) {
  srand(0);
  int vector_out_col_length= 1;
  float beta = 0.01;

  float** A = allocate_matrix(nrows, ncols);
  float** B = allocate_matrix(ncols, vector_out_col_length);

  fill_matrix(A, nrows, ncols, matrix);
  fill_matrix(B, ncols, vector_out_col_length, vector);
  
  self_correct_matrix_multiplication(A, nrows, ncols, B, vector_out_col_length, beta, ret, alpha_scale, beta_scale, iter);

  free_matrix(A, nrows);
  free_matrix(B, ncols);
}
