#include "wwd_cmsis.h"
#include "weights.h"
#include "utils_wwd.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include <wwd_mfcc.h>
#include <stdlib.h>

void quantize_wwd_input(const float *input_data, int8_t *quantized_data, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        quantized_data[i] = (int8_t)round(input_data[i] / WWD_INPUT_SCALE) + WWD_INPUT_ZERO_POINT;
    }
}

void dequantize_wwd_output(const int8_t *quantized_data, float *output_data, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        output_data[i] = (quantized_data[i] - WWD_OUTPUT_ZERO_POINT) * WWD_OUTPUT_SCALE;
    }
}

void run_wwd(uint8_t layer_fault_no, const int8_t *input_data, int8_t *output_data, 
            int8_t* fc1_output, int8_t* fc2_output, int8_t* fc3_output, int8_t* softmax_output)
{
    // Fully Connected Layer 1
    cmsis_nn_context fc1_ctx, fc2_ctx, fc3_ctx;
    cmsis_nn_fc_params fc1_params, fc2_params, fc3_params;
    cmsis_nn_per_tensor_quant_params fc1_quant_params, fc2_quant_params, fc3_quant_params;
    cmsis_nn_dims fc1_input_dims, fc2_input_dims, fc3_input_dims, fc1_output_dims, fc2_output_dims, fc3_output_dims;
    cmsis_nn_dims fc1_filter_dims, fc1_bias_dims, fc2_filter_dims, fc2_bias_dims, fc3_filter_dims, fc3_bias_dims;
    int8_t hidden_1[WWD_HIDDEN_DIM_1] = {0};
    int8_t hidden_2[WWD_HIDDEN_DIM_2] = {0};

    // Fullt Connected Layer 1 Start
    fc1_params.input_offset = -78;
    fc1_params.filter_offset = 0; 
    fc1_params.output_offset = -128;
    fc1_params.activation.max = 127;
    fc1_params.activation.min = -128;

    fc1_quant_params.multiplier = WWD_FC1_MULTIPLIER;
    fc1_quant_params.shift = WWD_FC1_SHIFT;           

    fc1_input_dims.n = 1;
    fc1_input_dims.h = 1;
    fc1_input_dims.w = 1;
    fc1_input_dims.c = WWD_INPUT_DIM;

    fc1_filter_dims.n = WWD_INPUT_DIM;
    fc1_filter_dims.h = 1;
    fc1_filter_dims.w = 1;
    fc1_filter_dims.c = WWD_HIDDEN_DIM_1;

    fc1_bias_dims.n = 1;
    fc1_bias_dims.h = 1;
    fc1_bias_dims.w = 1;
    fc1_bias_dims.c = WWD_HIDDEN_DIM_1;

    fc1_output_dims.n = 1;
    fc1_output_dims.h = 1;
    fc1_output_dims.w = 1;
    fc1_output_dims.c = WWD_HIDDEN_DIM_1;

    arm_cmsis_nn_status status;
    const int32_t buf1_size = arm_fully_connected_s8_get_buffer_size(&fc1_filter_dims);
    fc1_ctx.buf = malloc(buf1_size);
    fc1_ctx.size = buf1_size;
    
    if (layer_fault_no == 0x01)
    {
        trigger_high();
    }
    status = arm_fully_connected_s8(&fc1_ctx, &fc1_params, &fc1_quant_params,
                                    &fc1_input_dims, input_data,
                                    &fc1_filter_dims, sequential_dense_Weights,
                                    &fc1_bias_dims, sequential_dense_Biases,
                                    &fc1_output_dims, hidden_1);
    if (layer_fault_no == 0x01)
    {
        trigger_low();
    }

    if (fc1_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(fc1_ctx.buf, 0, buf1_size);
        free(fc1_ctx.buf);
    }

    for(int num=0; num<WWD_HIDDEN_DIM_1; num++){
        fc1_output[num] = hidden_1[num];
    }
    
    // Fully Connected Layer 2
    fc2_params.input_offset = 128;
    fc2_params.filter_offset = 0; 
    fc2_params.output_offset = -128;
    fc2_params.activation.max = 127;
    fc2_params.activation.min = -128;

    fc2_quant_params.multiplier = WWD_FC2_MULTIPLIER;
    fc2_quant_params.shift = WWD_FC2_SHIFT;

    fc2_input_dims.n = 1;
    fc2_input_dims.h = 1;
    fc2_input_dims.w = 1;
    fc2_input_dims.c = WWD_HIDDEN_DIM_1;

    fc2_filter_dims.n = WWD_HIDDEN_DIM_1;
    fc2_filter_dims.h = 1;
    fc2_filter_dims.w = 1;
    fc2_filter_dims.c = WWD_HIDDEN_DIM_2;

    fc2_bias_dims.n = 1;
    fc2_bias_dims.h = 1;
    fc2_bias_dims.w = 1;
    fc2_bias_dims.c = WWD_HIDDEN_DIM_2;

    fc2_output_dims.n = 1;
    fc2_output_dims.h = 1;
    fc2_output_dims.w = 1;
    fc2_output_dims.c = WWD_HIDDEN_DIM_2;

    const int32_t buf2_size = arm_fully_connected_s8_get_buffer_size(&fc2_filter_dims);
    fc2_ctx.buf = malloc(buf2_size);
    fc2_ctx.size = buf2_size;

    if (layer_fault_no == 0x02)
    {
        trigger_high();
    }
    status = arm_fully_connected_s8(&fc2_ctx, &fc2_params, &fc2_quant_params,
                                    &fc2_input_dims, hidden_1,
                                    &fc2_filter_dims, sequential_dense_1_Weights,
                                    &fc2_bias_dims, sequential_dense_1_Biases,
                                    &fc2_output_dims, hidden_2);
    if (layer_fault_no == 0x02)
    {
        trigger_low();
    }

    if (fc2_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(fc2_ctx.buf, 0, buf2_size);
        free(fc2_ctx.buf);
    }

    for(int num=0; num<WWD_HIDDEN_DIM_2; num++){
        fc2_output[num] = hidden_2[num];
    }
    
    // Fully Connected Layer 3
    fc3_params.input_offset = 128;
    fc3_params.filter_offset = 0;
    fc3_params.output_offset = 59;
    fc3_params.activation.max = 127;
    fc3_params.activation.min = -128;

    fc3_quant_params.multiplier = WWD_FC3_MULTIPLIER;
    fc3_quant_params.shift = WWD_FC3_SHIFT;

    fc3_input_dims.n = 1;
    fc3_input_dims.h = 1;
    fc3_input_dims.w = 1;
    fc3_input_dims.c = WWD_HIDDEN_DIM_2;

    fc3_filter_dims.n = WWD_HIDDEN_DIM_2;
    fc3_filter_dims.h = 1;
    fc3_filter_dims.w = 1;
    fc3_filter_dims.c = WWD_OUTPUT_DIM;

    fc3_bias_dims.n = 1;
    fc3_bias_dims.h = 1;
    fc3_bias_dims.w = 1;
    fc3_bias_dims.c = WWD_OUTPUT_DIM;

    fc3_output_dims.n = 1;
    fc3_output_dims.h = 1;
    fc3_output_dims.w = 1;
    fc3_output_dims.c = WWD_OUTPUT_DIM;

    const int32_t buf3_size = arm_fully_connected_s8_get_buffer_size(&fc3_filter_dims);
    fc3_ctx.buf = malloc(buf3_size);
    fc3_ctx.size = buf3_size;

    if (layer_fault_no == 0x03)
    {
        trigger_high();
    }
    status = arm_fully_connected_s8(&fc3_ctx, &fc3_params, &fc3_quant_params,
                                    &fc3_input_dims, hidden_2,
                                    &fc3_filter_dims, sequential_dense_2_Weights,
                                    &fc3_bias_dims, sequential_dense_2_Biases,
                                    &fc3_output_dims, output_data);
    if (layer_fault_no == 0x03)
    {
        trigger_low();
    }

    if (fc3_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(fc3_ctx.buf, 0, buf3_size);
        free(fc3_ctx.buf);
    }

    for(int num=0; num<WWD_OUTPUT_DIM; num++){
        fc3_output[num] = output_data[num];
    }

    // Softmax Activation
    int sft_shift = 23;
    int stf_mult = 1870077696;
    int sft_diff_min = -255;

    if (layer_fault_no == 0x04)
    {
        trigger_high();
    }
    arm_softmax_s8(output_data, 1, WWD_OUTPUT_DIM, stf_mult, sft_shift, sft_diff_min, output_data);
    if (layer_fault_no == 0x04)
    {
        trigger_low();
    }
    
    for(int num=0; num<WWD_OUTPUT_DIM; num++){
        softmax_output[num] = output_data[num];
    }
}

int8_t isWordDetected(float *out_pred)
{
    int8_t yes = 0;
    unsigned char det[1];
    if (out_pred[1] > 0.8)
    {
        yes = 1;
    }
    else
    {
        yes = 0;
    }
    return yes;
}

void mfcc_select(float *mfcc_to_infer, uint8_t mfcc_index)
{
    if (mfcc_index == 0x00)
    {
        for (int i = 0; i < 40; i++)
        {
            mfcc_to_infer[i] = mfcc_0[i]; // Background Noise
        }
    }
    else if (mfcc_index == 0x01)
    {
        for (int i = 0; i < 40; i++)
        {
            mfcc_to_infer[i] = mfcc_1[i]; // Audio Data
        }
    }
    else if (mfcc_index == 0x02)
    {
    	for (int i = 0; i < 40; i++)
    	{
    		mfcc_to_infer[i] = mfcc_2[i];
    	}
    }
    else
    {
        for (int i = 0; i < 64; i++)
        {
            mfcc_to_infer[i] = 0;
        }
    }
}

void run_wwd_protected(uint8_t layer_fault_no, const int8_t *input_data, int32_t *output_data,
                        int8_t* fc1_output, int8_t* fc2_output, int8_t* fc3_output, int8_t* softmax_output)
{
    // Fully Connected Layer 1
    cmsis_nn_fc_params fc1_params, fc2_params, fc3_params;
    cmsis_nn_per_tensor_quant_params fc1_quant_params, fc2_quant_params, fc3_quant_params;
    cmsis_nn_dims fc1_input_dims, fc2_input_dims, fc3_input_dims, fc1_output_dims, fc2_output_dims, fc3_output_dims;
    cmsis_nn_dims fc1_filter_dims, fc1_bias_dims, fc2_filter_dims, fc2_bias_dims, fc3_filter_dims, fc3_bias_dims;
    int32_t hidden_1[WWD_HIDDEN_DIM_1] = {0};
    int32_t hidden_2[WWD_HIDDEN_DIM_2] = {0};

    // Fully Connected Layer 1 Start
    fc1_params.input_offset = -78;
    fc1_params.filter_offset = 0; 
    fc1_params.output_offset = -128;
    fc1_params.activation.max = 127;
    fc1_params.activation.min = -128;

    fc1_quant_params.multiplier = WWD_FC1_MULTIPLIER;
    fc1_quant_params.shift = WWD_FC1_SHIFT;           

    fc1_input_dims.n = 1;
    fc1_input_dims.h = 1;
    fc1_input_dims.w = 1;
    fc1_input_dims.c = WWD_INPUT_DIM;

    fc1_filter_dims.n = WWD_INPUT_DIM;
    fc1_filter_dims.h = 1;
    fc1_filter_dims.w = 1;
    fc1_filter_dims.c = WWD_HIDDEN_DIM_1;

    fc1_bias_dims.n = 1;
    fc1_bias_dims.h = 1;
    fc1_bias_dims.w = 1;
    fc1_bias_dims.c = WWD_HIDDEN_DIM_1;

    fc1_output_dims.n = 1;
    fc1_output_dims.h = 1;
    fc1_output_dims.w = 1;
    fc1_output_dims.c = WWD_HIDDEN_DIM_1;

    arm_cmsis_nn_status status;
    if (layer_fault_no == 0x01)
    {
        trigger_high();
    }

    status = protected_fully_connected(input_data,sequential_dense_Weights, WWD_INPUT_DIM, WWD_HIDDEN_DIM_1, 0, 1, hidden_1, 1, 
                                        sequential_dense_Biases, fc1_params.input_offset,fc1_params.filter_offset,fc1_quant_params.multiplier, 
                                        fc1_quant_params.shift,fc1_params.output_offset,fc1_params.activation.min, fc1_params.activation.max);

    if (layer_fault_no == 0x01)
    {
        trigger_low();
    }
    
    int8_t hidden_1_new[WWD_HIDDEN_DIM_1] = {0};
    for (int i = 0; i < WWD_HIDDEN_DIM_1; i++){
        hidden_1_new[i] = hidden_1[i];
        fc1_output[i] = hidden_1[i];   
    }

    // Fully Connected Layer 2
    fc2_params.input_offset = 128;
    fc2_params.filter_offset = 0; 
    fc2_params.output_offset = -128;
    fc2_params.activation.max = 127;
    fc2_params.activation.min = -128;

    fc2_quant_params.multiplier = WWD_FC2_MULTIPLIER;
    fc2_quant_params.shift = WWD_FC2_SHIFT;

    fc2_input_dims.n = 1;
    fc2_input_dims.h = 1;
    fc2_input_dims.w = 1;
    fc2_input_dims.c = WWD_HIDDEN_DIM_1;

    fc2_filter_dims.n = WWD_HIDDEN_DIM_1;
    fc2_filter_dims.h = 1;
    fc2_filter_dims.w = 1;
    fc2_filter_dims.c = WWD_HIDDEN_DIM_2;

    fc2_bias_dims.n = 1;
    fc2_bias_dims.h = 1;
    fc2_bias_dims.w = 1;
    fc2_bias_dims.c = WWD_HIDDEN_DIM_2;

    fc2_output_dims.n = 1;
    fc2_output_dims.h = 1;
    fc2_output_dims.w = 1;
    fc2_output_dims.c = WWD_HIDDEN_DIM_2;
    
    if (layer_fault_no == 0x02)
    {
        trigger_high();
    }
    status = protected_fully_connected(hidden_1_new,sequential_dense_1_Weights, WWD_HIDDEN_DIM_1, WWD_HIDDEN_DIM_2, 0, 1, hidden_2, 1,
                                        sequential_dense_1_Biases, fc2_params.input_offset,fc2_params.filter_offset,fc2_quant_params.multiplier, 
                                        fc2_quant_params.shift,fc2_params.output_offset,fc2_params.activation.min, fc2_params.activation.max);
    if (layer_fault_no == 0x02)
    {
        trigger_low();
    }

    int8_t hidden_2_new[WWD_HIDDEN_DIM_2] = {0};
    for (int i = 0; i < WWD_HIDDEN_DIM_2; i++){
        hidden_2_new[i] = hidden_2[i];
        fc2_output[i] = hidden_2[i];
    }

    // Fully Connected Layer 3
    fc3_params.input_offset = 128;
    fc3_params.filter_offset = 0;
    fc3_params.output_offset = 59;
    fc3_params.activation.max = 127;
    fc3_params.activation.min = -128;

    fc3_quant_params.multiplier = WWD_FC3_MULTIPLIER;
    fc3_quant_params.shift = WWD_FC3_SHIFT;

    fc3_input_dims.n = 1;
    fc3_input_dims.h = 1;
    fc3_input_dims.w = 1;
    fc3_input_dims.c = WWD_HIDDEN_DIM_2;

    fc3_filter_dims.n = WWD_HIDDEN_DIM_2;
    fc3_filter_dims.h = 1;
    fc3_filter_dims.w = 1;
    fc3_filter_dims.c = WWD_OUTPUT_DIM;

    fc3_bias_dims.n = 1;
    fc3_bias_dims.h = 1;
    fc3_bias_dims.w = 1;
    fc3_bias_dims.c = WWD_OUTPUT_DIM;

    fc3_output_dims.n = 1;
    fc3_output_dims.h = 1;
    fc3_output_dims.w = 1;
    fc3_output_dims.c = WWD_OUTPUT_DIM;

    if (layer_fault_no == 0x03)
    {
        trigger_high();
    }
    status = protected_fully_connected(hidden_2_new, sequential_dense_2_Weights, WWD_HIDDEN_DIM_2, WWD_OUTPUT_DIM, 0, 1, output_data, 1,
                                        sequential_dense_2_Biases, fc3_params.input_offset,fc3_params.filter_offset,fc3_quant_params.multiplier, 
                                        fc3_quant_params.shift,fc3_params.output_offset,fc3_params.activation.min, fc3_params.activation.max);
    if (layer_fault_no == 0x03)
    {
        trigger_low();
    }

    int8_t output_data_new[WWD_OUTPUT_DIM] = {0};
    for(int i=0; i<WWD_OUTPUT_DIM; i++){
        output_data_new[i] = output_data[i];
        fc3_output[i] = output_data[i];
    }

    // Softmax Activation
    int sft_shift = 23;
    int stf_mult = 1870077696;
    int sft_diff_min = -255;

    if (layer_fault_no == 0x04)
    {
        trigger_high();
    }

    arm_softmax_s8(output_data_new, 1, WWD_OUTPUT_DIM, stf_mult, sft_shift, sft_diff_min, output_data_new);

    if (layer_fault_no == 0x04)
    {
        trigger_low();
    }
    
    for(int i=0; i<WWD_OUTPUT_DIM; i++){
        output_data[i] = output_data_new[i];
        softmax_output[i] = output_data_new[i];
    }  
}
