#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "weights_cnn.h"
#include "image_cnn.h"
#include "arm_nnfunctions.h"
#include "cnn_images.h"
#include "utils_cnn.h"


// Quantization parameters for each layer
int32_t conv_multiplier_cnn[CNN_OUT_CH] = {2079830784, 1937730560, 1889943040, 1970120960, 1132961024, 1886510720, 1789831296, 2009985152};
int32_t conv_shift_cnn[CNN_OUT_CH] = {-9, -9, -9, -9, -8, -9, -9, -9};


void quantize_input_cnn(const float *input_data_cnn, int8_t *quantized_data_cnn, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        quantized_data_cnn[i] = (int8_t)round(input_data_cnn[i] / 0.00392157) - 128;
    }
    // Print quantized input data
}

void dequantize_output_cnn(const int8_t *quantized_data_cnn, float *output_data_cnn, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output_data_cnn[i] = (quantized_data_cnn[i] - 60) * 0.11720886;
    }
}

void run_imgclass_inference(uint8_t layer_fault_no, const int8_t *input_data, int8_t *output_data,
                             int8_t* convolution_output, int8_t* maxpool_output, int8_t* fc_output) 
{

    // Convolution Layer 1
    cmsis_nn_context cnn_ctx, fc_ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_per_channel_quant_params conv_quant_params;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims, max_pool_filter_dims, filter_dims, bias_dims, output_dims;

    int8_t hidden_1[HIDDEN_DIM_1_CNN] = {0};
    int8_t hidden_2[HIDDEN_DIM_2_CNN] = {0};

    conv_params.input_offset = 128;
    conv_params.output_offset = -128; //Zero Point of Ouput Tensor after Convolution
    conv_params.stride.w = 1;
    conv_params.stride.h = 1;
    conv_params.padding.w = 0;
    conv_params.padding.h = 0;
    conv_params.dilation.w = 1;
    conv_params.dilation.h = 1;
    conv_params.activation.max = 127;
    conv_params.activation.min = -128;

    conv_quant_params.multiplier = (int32_t *)conv_multiplier_cnn;
    conv_quant_params.shift = (int32_t *)conv_shift_cnn;

    input_dims.n = INPUT_BATCHES_CNN;
    input_dims.h = INPUT_H_CNN;
    input_dims.w = INPUT_W_CNN;
    input_dims.c = INPUT_CH_CNN;

    filter_dims.n = CONV_OUT_CH_CNN;
    filter_dims.h = CNN_KERNEL_H;
    filter_dims.w = CNN_KERNEL_W;
    filter_dims.c = 1;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = 8;

    output_dims.n = 1;
    output_dims.h = CNN_OUT_H;
    output_dims.w = CNN_OUT_W;
    output_dims.c = CNN_OUT_CH;


    // Get the required buffer size
    int32_t buffer_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    cnn_ctx.buf = malloc(buffer_size);
    cnn_ctx.size = 0;

    arm_cmsis_nn_status status;

   if(layer_fault_no == 0x01){
       trigger_high();
   }
    status = arm_convolve_s8(&cnn_ctx, &conv_params, &conv_quant_params,
                             &input_dims, input_data,
                             &filter_dims, sequential_conv2d_Conv2D,
                             &bias_dims, sequential_conv2d_BiasAdd_ReadVariableOp,
                             &output_dims, hidden_1);
    if (layer_fault_no == 0x01){
        trigger_low();
    }

    if (cnn_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(cnn_ctx.buf, 0, buffer_size);
        free(cnn_ctx.buf);
    }
    if (status != ARM_CMSIS_NN_SUCCESS) {
    }

    for(int i = 0; i<HIDDEN_DIM_1_CNN; i++){
        convolution_output[i] = hidden_1[i];
    }


    // MaxPooling2D
    pool_params.padding.h = 0;
    pool_params.padding.w = 0;
    pool_params.stride.h = 2;
    pool_params.stride.w = 2;
    pool_params.activation.max = 127;
    pool_params.activation.min = -128;

    cmsis_nn_dims pool_input_dims;
    pool_input_dims.n = 1;
    pool_input_dims.h = 6;
    pool_input_dims.w = 6;
    pool_input_dims.c = 8;
    cmsis_nn_dims pool_output_dims;
    pool_output_dims.h = 3;
    pool_output_dims.w = 3;
    pool_output_dims.c = 8;

    max_pool_filter_dims.h = 2;
    max_pool_filter_dims.w = 2;

    if(layer_fault_no == 0x02){
        trigger_high();
    }
    status = arm_max_pool_s8(NULL, &pool_params,
                             &pool_input_dims, hidden_1,
                             &max_pool_filter_dims,
                             &pool_output_dims, hidden_2);
    if (layer_fault_no == 0x02)
    {
        trigger_low();
    }

    if (status != ARM_CMSIS_NN_SUCCESS) {
    }
    for(int i = 0; i<HIDDEN_DIM_2_CNN; i++){
        maxpool_output[i] = hidden_2[i];
    }

    // Fully Connected Layer 1
    quant_params.multiplier = FCCNN_MULTIPLIER;
    quant_params.shift = FCCNN_SHIFT;

    fc_params.input_offset = 128;
    fc_params.filter_offset = 0;
    fc_params.output_offset = 60;
    fc_params.activation.max = 127;
    fc_params.activation.min = -128;

    cmsis_nn_dims fc1_input_dims;
    fc1_input_dims.n = 1;
    fc1_input_dims.h = 1;
    fc1_input_dims.w = 72;
    fc1_input_dims.c = 1;
    cmsis_nn_dims fc1_filter_dims;
    fc1_filter_dims.n = 72;
    fc1_filter_dims.c = 10;
    cmsis_nn_dims fc1_bias_dims;
    fc1_bias_dims.c = 10;
    cmsis_nn_dims fc1_output_dims;
    fc1_output_dims.n = 1;
    fc1_output_dims.c = 10;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&fc1_filter_dims);
    fc_ctx.buf = malloc(buf_size);
    fc_ctx.size = buf_size;


    if(layer_fault_no == 0x03){
        trigger_high();
    }
    status = arm_fully_connected_s8(&fc_ctx, &fc_params, &quant_params,
                                    &fc1_input_dims, hidden_2,
                                    &fc1_filter_dims, sequential_dense_MatMul,
                                    &fc1_bias_dims, sequential_dense_BiasAdd_ReadVariableOp,
                                    &fc1_output_dims, output_data);
    if (layer_fault_no == 0x03)
    {
        trigger_low();
    }

    if (status != ARM_CMSIS_NN_SUCCESS) {
        // printf("Error in fully connected layer 1: %d\n", status);
    }
    for(int i = 0; i<OUTPUT_DIM_CNN; i++){
        fc_output[i] = output_data[i];
    }

    if (fc_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(fc_ctx.buf, 0, buf_size);
        free(fc_ctx.buf);
    }

}


int8_t predicted_number(float *de_output, int size){
	int8_t max_ind = 0;
	float max_num = -FLT_MAX;
	for(int i = 0; i<size; i++){
		if(de_output[i]>max_num){
			max_ind = i;
			max_num = de_output[i];
		}
	}
	return max_ind;
}

void image_select(float *image_to_infer, uint8_t image_index){
	if(image_index == 0x00){
		for(int i = 0; i<64; i++){
			image_to_infer[i] = image0[i];
		}
	}
	else if (image_index == 0x01){
		for(int i = 0; i<64; i++){
			image_to_infer[i] = image1[i];
		}
	}
	else if (image_index == 0x02){
		for(int i = 0; i<64; i++){
			image_to_infer[i] = image2[i];
		}
	}
	else if (image_index == 0x03){
		for(int i = 0; i<64; i++){
			image_to_infer[i] = image3[i];
		}
	}
	else if (image_index == 0x04){
		for(int i = 0; i<64; i++){
			image_to_infer[i] = image4[i];
		}
	}
	else if (image_index == 0x05){
		for(int i = 0; i<64; i++){
			image_to_infer[i] = image5[i];
		}
	}
	else if (image_index == 0x06){
		for(int i = 0; i<64; i++){
			image_to_infer[i] = image6[i];
		}
	}
	else if (image_index == 0x07){
		for(int i = 0; i<64; i++){
			image_to_infer[i] = image7[i];
		}
	}
	else if (image_index == 0x08){
		for(int i = 0; i<64; i++){
			image_to_infer[i] = image8[i];
		}
	}
	else if (image_index == 0x09){
		for(int i = 0; i<64; i++){
			image_to_infer[i] = image9[i];
		}
	}
	else {
		for(int i = 0; i<64; i++){
			image_to_infer[i] = 0;
		}
	}
}


void run_imgclass_inference_protected(uint8_t layer_fault_no, const int8_t *input_data, int32_t *output_data,
                                        int8_t* convolution_output, int8_t* maxpool_output, int8_t* fc_output) 
{

    // Convolution Layer 1
    cmsis_nn_context cnn_ctx, fc_ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_per_channel_quant_params conv_quant_params;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims, max_pool_filter_dims, filter_dims, bias_dims, output_dims;

    int32_t hidden_1[HIDDEN_DIM_1_CNN] = {0};
    int8_t  hidden_2[HIDDEN_DIM_2_CNN] = {0};

    conv_params.input_offset = 128;
    conv_params.output_offset = -128; //Zero Point of Ouput Tensor after Convolution
    conv_params.stride.w = 1;
    conv_params.stride.h = 1;
    conv_params.padding.w = 0;
    conv_params.padding.h = 0;
    conv_params.dilation.w = 1;
    conv_params.dilation.h = 1;
    conv_params.activation.max = 127;
    conv_params.activation.min = -128;

    conv_quant_params.multiplier = (int32_t *)conv_multiplier_cnn;
    conv_quant_params.shift = (int32_t *)conv_shift_cnn;

    input_dims.n = INPUT_BATCHES_CNN;
    input_dims.h = INPUT_H_CNN;
    input_dims.w = INPUT_W_CNN;
    input_dims.c = INPUT_CH_CNN;

    filter_dims.n = CONV_OUT_CH_CNN;
    filter_dims.h = CNN_KERNEL_H;
    filter_dims.w = CNN_KERNEL_W;
    filter_dims.c = 1;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = 8;

    output_dims.n = 1;
    output_dims.h = CNN_OUT_H;
    output_dims.w = CNN_OUT_W;
    output_dims.c = CNN_OUT_CH;

    arm_cmsis_nn_status status;
   if(layer_fault_no == 0x01){
       trigger_high();
   }
    status = protected_convolution(input_data, INPUT_H_CNN, INPUT_W_CNN, INPUT_CH_CNN, sequential_conv2d_Conv2D, CNN_KERNEL_H, CNN_KERNEL_W, CONV_OUT_CH_CNN,
                                    conv_params.padding.h, conv_params.padding.w, conv_params.stride.h, conv_params.stride.w, conv_params.dilation.h, conv_params.dilation.w,
                                    sequential_conv2d_BiasAdd_ReadVariableOp, conv_params.input_offset, conv_quant_params.multiplier, 
                                    conv_quant_params.shift, conv_params.output_offset, conv_params.activation.min, conv_params.activation.max, hidden_1);
    if (layer_fault_no == 0x01)
    {
        trigger_low();
    }

    if (status != ARM_CMSIS_NN_SUCCESS) {
    }
    
    int8_t hidden_1_new[HIDDEN_DIM_1_CNN] = {0};
    for (int i = 0; i < HIDDEN_DIM_1_CNN; i++){
        hidden_1_new[i] = hidden_1[i];
        convolution_output = hidden_1[i];
    }

    // MaxPooling2D
    pool_params.padding.h = 0;
    pool_params.padding.w = 0;
    pool_params.stride.h = 2;
    pool_params.stride.w = 2;
    pool_params.activation.max = 127;
    pool_params.activation.min = -128;

    cmsis_nn_dims pool_input_dims;
    pool_input_dims.n = 1;
    pool_input_dims.h = 6;
    pool_input_dims.w = 6;
    pool_input_dims.c = 8;
    cmsis_nn_dims pool_output_dims;
    pool_output_dims.h = 3;
    pool_output_dims.w = 3;
    pool_output_dims.c = 8;

    max_pool_filter_dims.h = 2;
    max_pool_filter_dims.w = 2;

    if(layer_fault_no == 0x02){
        trigger_high();
    }
    
    status = arm_max_pool_s8(NULL, &pool_params,
                            &pool_input_dims, hidden_1_new,
                            &max_pool_filter_dims,
                            &pool_output_dims, hidden_2);
    if (layer_fault_no == 0x02)
    {
        trigger_low();
    }

    if (status != ARM_CMSIS_NN_SUCCESS) {
    }

    int8_t hidden_2_new[HIDDEN_DIM_2_CNN] = {0};
    for (int i = 0; i < HIDDEN_DIM_2_CNN; i++){
        hidden_2_new[i] = hidden_2[i];
        maxpool_output[i]= hidden_2[i];
    }

    // Fully Connected Layer 1
    quant_params.multiplier = FCCNN_MULTIPLIER;
    quant_params.shift = FCCNN_SHIFT;

    fc_params.input_offset = 128;
    fc_params.filter_offset = 0;
    fc_params.output_offset = 60;
    fc_params.activation.max = 127;
    fc_params.activation.min = -128;

    cmsis_nn_dims fc1_input_dims;
    fc1_input_dims.n = 1;
    fc1_input_dims.h = 1;
    fc1_input_dims.w = 72;
    fc1_input_dims.c = 1;
    cmsis_nn_dims fc1_filter_dims;
    fc1_filter_dims.n = 72;
    fc1_filter_dims.c = 10;
    cmsis_nn_dims fc1_bias_dims;
    fc1_bias_dims.c = 10;
    cmsis_nn_dims fc1_output_dims;
    fc1_output_dims.n = 1;
    fc1_output_dims.c = 10;

    if(layer_fault_no == 0x03){
        trigger_high();
    }
    
    status = protected_fully_connected(hidden_2_new, sequential_dense_MatMul, fc1_input_dims.w, fc1_output_dims.c , 0, 1, output_data, 1,
                                        sequential_dense_BiasAdd_ReadVariableOp, fc_params.input_offset,fc_params.filter_offset,quant_params.multiplier, 
                                        quant_params.shift,fc_params.output_offset,fc_params.activation.min, fc_params.activation.max);
    if (layer_fault_no == 0x03)
    {
        trigger_low();
    }

    if (status != ARM_CMSIS_NN_SUCCESS) {
        // printf("Error in fully connected layer 1: %d\n", status);
    }
    
    for (int i = 0; i < OUTPUT_DIM_CNN; i++){
        fc_output[i]= output_data[i];
    }
}