//#include <stdio.h>
#include <stdint.h>
#include <math.h>
// #include "stm32f4xx_gpio.h"

#define WWD_INPUT_DIM 40
#define WWD_HIDDEN_DIM_1 256
#define WWD_HIDDEN_DIM_2 256
#define WWD_OUTPUT_DIM 2

// Define the quantization parameters
#define WWD_INPUT_OFFSET -78
#define WWD_OUTPUT_OFFSET 128
#define WWD_INPUT_SCALE 2.9606902599334717
#define WWD_INPUT_ZERO_POINT 78
#define WWD_OUTPUT_SCALE 0.00390625
#define WWD_OUTPUT_ZERO_POINT -128

// Define the extracted multipliers and shifts for each layer
#define WWD_FC1_MULTIPLIER 1843280384
#define WWD_FC1_SHIFT -3

#define WWD_FC2_MULTIPLIER 1631615232
#define WWD_FC2_SHIFT -9

#define WWD_FC3_MULTIPLIER 1941244288
#define WWD_FC3_SHIFT -10

// I/O quantization: converts float to int8 and int8 to float
void quantize_wwd_input(const float* input_data, int8_t* quantized_data, size_t size);
void dequantize_wwd_output(const int8_t* quantized_data, float* output_data, size_t size);


// Neural Network Forward Pass
void run_wwd(uint8_t layer_fault_no, const int8_t* input_data, int8_t* output_data,
            int8_t* fc1_output, int8_t* fc2_output, int8_t* fc3_output, int8_t* softmax_output);
void run_wwd_protected(uint8_t layer_fault_no, const int8_t* input_data, int32_t* output_data,
            int8_t* fc1_output, int8_t* fc2_output, int8_t* fc3_output, int8_t* softmax_output);
int8_t isWordDetected(float *out_pred);
void mfcc_select(float *mfcc_to_infer, uint8_t mfcc_index);


