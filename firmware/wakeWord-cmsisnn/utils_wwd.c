#include <math.h>
#include <stdlib.h>
#include "utils_wwd.h"

#define EPSILON 1e-6 // Tolerance for int8_ting-point comparison

// Function to allocate memory for a matrix
int8_t** allocate_matrix_wwd(int rows, int cols) {
  int8_t** matrix = (int8_t**)malloc(rows * sizeof(int8_t*));
  for (int i = 0; i < rows; i++) {
      matrix[i] = (int8_t*)malloc(cols * sizeof(int8_t));
  }
  return matrix;
}

// Function to free memory for a matrix
void free_matrix_wwd(int8_t** matrix, int rows) {
  for (int i = 0; i < rows; i++) {
      free(matrix[i]);
  }
  free(matrix);
}

int32_t** allocate_matrix32_wwd(int rows, int cols) {
  int32_t** matrix = (int32_t**)malloc(rows * sizeof(int32_t*));
  for (int i = 0; i < rows; i++) {
      matrix[i] = (int32_t*)malloc(cols * sizeof(int32_t));
  }
  return matrix;
}

void free_matrix32_wwd(int32_t** matrix, int rows) {
  for (int i = 0; i < rows; i++) {
      free(matrix[i]);
  }
  free(matrix);
}

// Function to initialize a matrix with random int8_t values
void random_matrix_wwd(int8_t** matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          matrix[i][j] = ((int8_t)rand() / RAND_MAX) * 10.0; // Random values [0, 10)
      }
  }
}

// Function to initialize a matrix with int8_t values from array
void fill_matrix_wwd(int8_t** matrix, int rows, int cols, const int8_t* const input_arr){
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          matrix[i][j] = input_arr[i * cols + j];
      }
  }
}

void fill_matrix_transpose_wwd(int8_t** matrix, int rows, int cols, const int8_t* const input_arr){
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          matrix[i][j] = input_arr[j * rows + i]; // transpose: swap i and j
      }
  }
}

// Function to print a matrix
void print_matrix_wwd(int8_t** matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          printf("%d ", matrix[i][j]);
      }
      printf("\n");
  }
}

// Function to subtract matrices: C = A - B
void subtract_matrix_wwd(int8_t** A, int8_t** B, int32_t** C, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          C[i][j] = A[i][j] - B[i][j];
      }
  }
}

// Function to subtract matrices: C = A - B
void subtract_matrix_with_offset_wwd(int8_t** A, int8_t** B, int32_t** C, int rows, int cols, int32_t offset_A) {
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          C[i][j] = (int32_t)A[i][j] + offset_A - (int32_t)B[i][j];
      }
  }
}

// Reliable matrix multiplication: C = A * B
void matrix_multiplication_wwd(int8_t** A, int n, int m, int8_t** B, int p, int32_t** result) {
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
          int32_t sum = 0;
          for (int k = 0; k < m; k++) {
              int32_t lhs = (int32_t)A[i][k];
              int32_t rhs = (int32_t)B[k][j];
              sum += lhs * rhs;
          }
          result[i][j] = sum;
      }
  }
}

// Reliable matrix multiplication: C = A * B
void matrix_multiplication32_wwd_1(int32_t** A, int n, int m, int8_t** B, int p, int32_t** result) {
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
          int32_t sum = 0;
          for (int k = 0; k < m; k++) {
              int32_t lhs = A[i][k];  
              int32_t rhs = (int32_t)B[k][j];
              sum += lhs * rhs;
          }
          result[i][j] = sum;
      }
  }
}

// Reliable matrix multiplication: C = A * B
void matrix_multiplication32_wwd_2(int8_t** A, int n, int m, int32_t** B, int p, int32_t** result) {
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
          int32_t sum = 0;
          for (int k = 0; k < m; k++) {
              int32_t lhs = (int32_t)A[i][k];
              int32_t rhs = B[k][j]; 
              sum += lhs * rhs;
          }
          result[i][j] = sum;
      }
  }
}

// Reliable matrix multiplication: C = A * B
void matrix_multiplication32_wwd_3(int32_t** A, int n, int m, int32_t** B, int p, int32_t** result) {
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
          int32_t sum = 0;
          for (int k = 0; k < m; k++) {
              int32_t lhs = A[i][k]; 
              int32_t rhs = B[k][j];  
              sum += lhs * rhs;
          }
          result[i][j] = sum;
      }
  }
}

// Function to compare two matrices with a tolerance for floating point errors
int compare_matrices_wwd(int8_t** A, int8_t** B, int rows, int cols) {
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
int32_t calculate_median_wwd(int32_t* values, int size) {
  for (int i = 0; i < size - 1; i++) {
    for (int j = 0; j < size - i - 1; j++) {
      if (values[j] > values[j + 1]) {
        int32_t temp = values[j];
        values[j] = values[j + 1];
        values[j + 1] = temp;
      }
    }
  }
  return (size % 2 == 0) ? (values[size / 2 - 1] + values[size / 2]) / 2.0 : values[size / 2];
}

// Majority voting algorithm using median
void majority_vote_wwd(int32_t*** results, int num_results, int rows, int cols, int32_t* final_result) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int32_t* values = (int32_t*)malloc(num_results * sizeof(int32_t));
      for (int k = 0; k < num_results; k++) {
        values[k] = results[k][i][j];
      }
      final_result[i * cols + j] = calculate_median_wwd(values, num_results);
      free(values);
    }
  }
}

// Self-correcting matrix multiplication
void self_correct_matrix_multiplication_wwd(int8_t* A, int n, int m, int8_t* B, int p, int8_t beta, int32_t* const result, int8_t alpha_scale, int8_t beta_scale, int iter, const int32_t* bias, int32_t input_offset, int32_t weight_offset, int32_t quant_multiplier, int32_t quant_shift, int32_t output_offset, int8_t clamp_min, int8_t clamp_max) {
  
  ////////////////////////////////////////////////////////////////////////////////
  // Current implementation 
  ////////////////////////////////////////////////////////////////////////////////
  int num_repeats = 5;
  int32_t temp_vals[num_repeats];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      // compute each repeat
      for (int r = 0; r < num_repeats; r++) {
        int32_t acc = 0;
        // fresh random shares per (i,k,r) and (k,j,r)
        for (int k = 0; k < m; k++) {
            int8_t A1 = (int8_t)(rand() & 0xFF);   
            int8_t B1 = (int8_t)(rand() & 0xFF);  
            int8_t a  = (int32_t)A[i*m + k];
            int8_t b  = (int32_t)B[j*m + k];
            int32_t A2 = (int32_t)a + input_offset - (int32_t)A1;
            int32_t B2 = (int32_t)b + weight_offset - (int32_t)B1;
            acc += (int32_t)A1 * (int32_t)B1;
            acc += A2 * (int32_t)B1;
            acc += (int32_t)A1 * B2;
            acc += A2 * B2;
        }
        temp_vals[r] = acc;
      }
      // median vote
      int32_t out = calculate_median_wwd(temp_vals, num_repeats);
      // bias
      if (bias) out += bias[j];
      // requantize
      out = arm_nn_requantize(out, quant_multiplier, quant_shift);
      // offset
      out += output_offset;
      // clamp
      if (out > clamp_max) out = clamp_max;
      if (out < clamp_min) out = clamp_min;
      // store
      result[i * p + j] = out;
    }
  }
}

void protectedVecmat(const int8_t* const vector, const int8_t* const matrix, unsigned ncols_A, unsigned ncols_B, int8_t alpha_scale, int8_t beta_scale, int32_t* const ret, int iter, const int32_t* bias, int32_t input_offset, int32_t weight_offset, int32_t quant_multiplier, int32_t quant_shift, int32_t output_offset, int8_t clamp_min, int8_t clamp_max) {
  srand(0);

  int vector_input_row_length= 1;
  int8_t beta = 0.01; 
  
  self_correct_matrix_multiplication_wwd(vector, vector_input_row_length, ncols_A, matrix, ncols_B, beta, ret, alpha_scale, beta_scale, iter, bias, input_offset, weight_offset, quant_multiplier, quant_shift, output_offset, clamp_min, clamp_max);
}

arm_cmsis_nn_status protected_fully_connected(const int8_t* const vector, const int8_t* const matrix, unsigned ncols_A, unsigned ncols_B, int8_t alpha_scale, int8_t beta_scale, int32_t* const ret, int iter, const int32_t* bias, int32_t input_offset, int32_t weight_offset, int32_t quant_multiplier, int32_t quant_shift, int32_t output_offset, int8_t clamp_min, int8_t clamp_max) {

  protectedVecmat(vector, matrix, ncols_A, ncols_B, alpha_scale, beta_scale, ret, iter, bias, input_offset, weight_offset, quant_multiplier, quant_shift, output_offset, clamp_min, clamp_max);

  return (ARM_CMSIS_NN_SUCCESS);
}
