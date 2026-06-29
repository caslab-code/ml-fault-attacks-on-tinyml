#include <stdint.h>
#include <math.h>
// #include "stm32f4xx_gpio.h"


// Input dimensions
#define INPUT_BATCHES_CNN 1
#define INPUT_H_CNN 8
#define INPUT_W_CNN 8
#define INPUT_CH_CNN 1

// Convolution layer dimensions
#define CNN_OUT_CH 8
#define CNN_KERNEL_H 3
#define CNN_KERNEL_W 3
#define CNN_OUT_H 6
#define CNN_OUT_W 6

//Max Pooling Layer Dimensions
#define MAX_POOL_OUT_H 3
#define MAX_POOL_OUT_W 3

#define INPUT_DIM_CNN INPUT_BATCHES_CNN*INPUT_H_CNN*INPUT_W_CNN*INPUT_CH_CNN
#define CONV_FILTER_DIMS_CNN 72
#define CONV_OUT_CH_CNN 8
#define HIDDEN_DIM_1_CNN INPUT_BATCHES_CNN*CNN_OUT_H*CNN_OUT_W*CNN_OUT_CH
#define HIDDEN_DIM_2_CNN INPUT_BATCHES_CNN*MAX_POOL_OUT_H*MAX_POOL_OUT_W*CNN_OUT_CH
#define OUTPUT_DIM_CNN 10

// Define the quantization parameters
#define INPUT_OFFSET_CNN 128
#define OUTPUT_OFFSET_CNN -60
#define INPUT_SCALE_CNN 0.00392157
#define INPUT_ZERO_POINT_CNN -128
#define OUTPUT_SCALE_CNN 0.11720886
#define OUTPUT_ZERO_POINT_CNN 60

// Define the extracted multipliers and shifts for each layer
#define FCCNN_MULTIPLIER 1170927744
#define FCCNN_SHIFT -8

void quantize_input_cnn(const float *input_data_cnn, int8_t *quantized_data_cnn, size_t size);
void dequantize_output_cnn(const int8_t *quantized_data_cnn, float *output_data_cnn, size_t size);
void run_imgclass_inference(uint8_t layer_fault_no, const int8_t *input_data, int8_t *output_data,
                             int8_t* convolution_output, int8_t* maxpool_output, int8_t* fc_output);
int8_t predicted_number(float *dequantized_output, int size);
void image_select(float *image_to_infer, uint8_t image_index);
void run_imgclass_inference_protected(uint8_t layer_fault_no, const int8_t *input_data, int32_t *output_data,
                                        int8_t* convolution_output, int8_t* maxpool_output, int8_t* fc_output);
