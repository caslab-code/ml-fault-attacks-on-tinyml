#include <math.h>
#include <stdlib.h>
#include "utils_cnn.h"
#include "utils_wwd.h"     // for protectedVecmat()

arm_cmsis_nn_status protected_convolution(
  const int8_t*  matrix,          // [H×W×Cin]
  int             matrix_height,
  int             matrix_width,
  int             input_channel,
  const int8_t*  kernel,          // [Cout×Kh×Kw×Cin]
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
) {
  // compute output dims
  int eff_kh = (kernel_height - 1) * dilation_height + 1;
  int eff_kw = (kernel_width  - 1) * dilation_width  + 1;
  int H_out  = (matrix_height + 2 * pad_height - eff_kh) / stride_height + 1;
  int W_out  = (matrix_width  + 2 * pad_width  - eff_kw) / stride_width  + 1;

  const int M = kernel_height * kernel_width * input_channel;
  int8_t *patch = malloc(M * sizeof(int8_t));
  int8_t *weight_col = malloc(M * sizeof(int8_t));
  int32_t ret_scalar[1];

  if (!patch || !weight_col) {
    free(patch);
    free(weight_col);
    return ARM_CMSIS_NN_ARG_ERROR;
  }

  // slide the window
  for (int oh = 0; oh < H_out; oh++) {
    for (int ow = 0; ow < W_out; ow++) {
      // 1) build the M-vector of int8_t inputs (im2col for one patch)
      int idx = 0;
      for (int kh = 0; kh < kernel_height; kh++) {
        for (int kw = 0; kw < kernel_width; kw++) {
          int in_h = oh * stride_height + kh * dilation_height - pad_height;
          int in_w = ow * stride_width  + kw * dilation_width  - pad_width;

          for (int ic = 0; ic < input_channel; ic++) {
            if (in_h >= 0 && in_h < matrix_height &&
                in_w >= 0 && in_w < matrix_width) {
              patch[idx++] = matrix[(in_h * matrix_width + in_w) * input_channel + ic];
            } else {
              patch[idx++] = (int8_t)(-input_offset); // padding value
            }
          }
        }
      }

      // 2) for each output channel, extract weight column and compute dot product
      for (int oc = 0; oc < output_channel; oc++) {
        for (int m = 0; m < M; m++) {
          int kh = (m / (kernel_width * input_channel)) % kernel_height;
          int kw = (m / input_channel) % kernel_width;
          int ic = m % input_channel;

          // CMSIS-NN kernel layout: [C_OUT][KH][KW][C_IN]
          weight_col[m] = kernel[((oc * kernel_height + kh) * kernel_width + kw) * input_channel + ic];
        }

        protectedVecmat(
          patch,                 // vector A (1 × M)
          weight_col,            // matrix B (M × 1)
          M,                     // ncols_A
          1,                     // ncols_B = 1
          1,                     // alpha_scale
          1,                     // beta_scale
          ret_scalar,            // output
          0,                     // iter (used for randomness if needed)
          bias ? &bias[oc] : NULL,
          input_offset,
          0,                     // weight offset
          quant_multiplier[oc],
          quant_shift[oc],
          output_offset,
          clamp_min,
          clamp_max
        );

        output[(oh * W_out + ow) * output_channel + oc] = (int8_t)ret_scalar[0];
      }
    }
  }

  free(patch);
  free(weight_col);
  return ARM_CMSIS_NN_SUCCESS;
}

// MaxPool with random additive shares protection
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
) {
  int num_repeats = 5;

  int eff_H_out = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  int eff_W_out = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

  // Allocate results for each repeat: [num_repeats][eff_H_out][eff_W_out * input_channel]
  int32_t*** results = (int32_t***)malloc(num_repeats * sizeof(int32_t**));
  for (int r = 0; r < num_repeats; r++) {
    results[r] = (int32_t**)malloc(eff_H_out * sizeof(int32_t*));
    for (int oh = 0; oh < eff_H_out; oh++) {
      results[r][oh] = (int32_t*)malloc(eff_W_out * input_channel * sizeof(int32_t));
    }
  }

  // Repeat maxpool with small random noise added to each value
  for (int r = 0; r < num_repeats; r++) {
    for (int oh = 0; oh < eff_H_out; oh++) {
      for (int ow = 0; ow < eff_W_out; ow++) {
        for (int c = 0; c < input_channel; c++) {
          int32_t max_val = INT8_MIN; // start from smallest int8_t
          for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
              int in_h = oh * stride_height + kh - pad_height;
              int in_w = ow * stride_width + kw - pad_width;

              int32_t val;
              if (in_h < 0 || in_h >= input_height || in_w < 0 || in_w >= input_width) {
                val = INT8_MIN;  // treat out-of-bound as minimal value
              } else {
                val = input[(in_h * input_width + in_w) * input_channel + c];
                int8_t share1 = (int8_t)(rand() & 0xFF);
                int32_t share2 = (int32_t)val - (int32_t)share1;
                val = (int32_t)share1 + share2; // reconstruct
              }

             

              if (val > max_val) max_val = val;
            }
          }
          results[r][oh][ow * input_channel + c] = max_val;
        }
      }
    }
  }

  // Majority vote via median across repeats
  majority_vote_wwd(results, num_repeats, eff_H_out, eff_W_out * input_channel, output);

  // Apply activation clamping and write to int8 output
  for (int i = 0; i < eff_H_out * eff_W_out * input_channel; i++) {
    int32_t clamped = output[i];
    if (clamped < activation_min) clamped = activation_min;
    if (clamped > activation_max) clamped = activation_max;
    output[i] = (int8_t)clamped;
  }

  // Free results
  for (int r = 0; r < num_repeats; r++) {
    for (int oh = 0; oh < eff_H_out; oh++) {
      free(results[r][oh]);
    }
    free(results[r]);
  }
  free(results);

  return ARM_CMSIS_NN_SUCCESS;
}
