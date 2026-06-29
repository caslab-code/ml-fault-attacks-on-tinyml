#include "hal.h"
#include "simpleserial.h"
#include <string.h>

#include "fast_grnn.h"
#include "floatPrint.h"
#include "logistic_regression.h"
#include "wwd_cmsis.h"
#include "image_cnn.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////
// FAST GRNN
#define IMAGE_NUM_FLOATS 256
#define IMAGE_NUM_BYTES  (IMAGE_NUM_FLOATS * sizeof(float))

// static buffer & offset to accumulate chunks:
static uint8_t  image_buffer_fast_grnn[IMAGE_NUM_BYTES] = {}; //256 float values = 256*4bytes = 1024bytes
static uint16_t image_buffer_fast_grnn_offset = 0;

static uint8_t select_on_board_image_fast_grnn = 0;
static int8_t predict_num_fast_grnn = 0;
static uint8_t image_index_fast_grnn = 0;

static float image_to_infer_fast_grnn[256]; // uart is 16 x 16
static float image_to_infer_fast_grnn_board[256]; // board images are 16 x 16

float hiddenState[HIDDEN_DIMS] = { 0.0 };
float classScores[NUM_CLASSES] = {0.0};
uint8_t arg_classScores[NUM_CLASSES] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
float preComp[HIDDEN_DIMS] = { 0.0 };
float tempLRW[USPS_WRANK] = { 0.0 };
float tempLRU[USPS_URANK] = { 0.0 };
float normFeatures[INPUT_DIMS] = { 0.0 };
uint16_t layer_fault_no = 0x00;

// Fast Grnn Init
FastGRNN_LR_Params USPS_params = {
    .mean   = USPS_mean,
    .stdDev = USPS_stdDev,
    .W1     = USPS_W1,
    .W2     = USPS_W2,
    .wRank  = USPS_WRANK,
    .U1     = USPS_U1,
    .U2     = USPS_U2,
    .uRank  = USPS_URANK,
    .Bg     = USPS_Bg,
    .Bh     = USPS_Bh,
    .sigmoid_zeta = USPS_sigmoid_zeta,
    .sigmoid_nu   = USPS_sigmoid_nu
};

FastGRNN_LR_Buffers buffers = {
    .preComp = preComp,
    .tempLRW = tempLRW,
    .tempLRU = tempLRU,
    .normFeatures = normFeatures
};
////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
// Logistic Regression
#define LOG_REG_DATA_NUM_DOUBLE 11
#define LOG_REG_DATA_NUM_BYTES  (LOG_REG_DATA_NUM_DOUBLE * sizeof(double))

// static buffer & offset to accumulate chunks:
static uint8_t  input_data_buffer_log_reg[LOG_REG_DATA_NUM_BYTES] = {}; //11 double values = 11*8bytes = 88bytes; double is 8 bytes
static uint16_t input_data_buffer_log_reg_offset = 0;

static uint8_t log_reg_class_pred = 0xff;  // to differentiate from predicted 0 or 1
static double input_data_log_reg[11];

// Intermediate captured during unprotected inference (defined in logistic_regression.c)
extern double g_lr_products[LOG_REG_DATA_NUM_DOUBLE];  // per-feature terms X[j]*w[j]
////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////
// WakeWord
#define WAKE_WORD_NUM_FLOATS 40
#define WAKE_WORD_NUM_BYTES  (WAKE_WORD_NUM_FLOATS * sizeof(float))
#define MAX_BYTES_PER_PKT  128   

// static buffer & offset to accumulate chunks:
static uint8_t  mfcc_buffer[WAKE_WORD_NUM_BYTES] = {}; //40 float values = 40*4bytes = 160bytes
static uint16_t mfcc_buffer_offset = 0;

static uint8_t select_on_board_mfcc = 0;
static uint8_t mfcc_index = 0;
static float out_pred_wwd[WWD_OUTPUT_DIM];
static uint8_t out_pred_wwd_final = 0xff; // to differentiate from predicted 0 or 1

// Initialize input data either board input  or uart
static float mfcc_to_infer[40]; 

// Initialize input data 
static int8_t input_data_wwd[WWD_INPUT_DIM];
static int8_t output_data_wwd[WWD_OUTPUT_DIM];
static int32_t output_data_wwd_protected[WWD_OUTPUT_DIM];

// Wakeword Intermediate Values
static int8_t fc1_output[WWD_HIDDEN_DIM_1];
static int8_t fc2_output[WWD_HIDDEN_DIM_2]; 
static int8_t fc3_output[WWD_OUTPUT_DIM]; 
static int8_t softmax_output[WWD_OUTPUT_DIM]; 
////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
// Image CNN
#define IMAGE_CNN_NUM_FLOATS 64
#define IMAGE_CNN_NUM_BYTES  (IMAGE_CNN_NUM_FLOATS * sizeof(float)) 
#define MAX_BYTES_PER_PKT_CNN  144   

// static buffer & offset to accumulate chunks:
static uint8_t  image_buffer[IMAGE_CNN_NUM_BYTES] = {};  //64 float values = 64*4bytes = 256bytes
static uint16_t image_buffer_offset = 0;

static uint8_t select_on_board_image = 0;
static uint8_t image_index = 0;
static float out_pred_cnn[OUTPUT_DIM_CNN];
static int8_t predict_num = 0xff;

// Initialize input data either board input  or uart
static float image_to_infer[64];

// Initialize input data 
static int8_t input_data_cnn[INPUT_DIM_CNN];
static int8_t output_data_cnn[OUTPUT_DIM_CNN];
static int32_t output_data_cnn_protected[OUTPUT_DIM_CNN];

// Image CNN Intermediate Values
static int8_t convolution_output[HIDDEN_DIM_1_CNN];
static int8_t maxpool_output[HIDDEN_DIM_2_CNN]; 
static int8_t fc_output[OUTPUT_DIM_CNN]; 
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
/*
 * Converts byte array received from UART to float array
 *
 * Inputs:
 * buffer - Input byte array from UART
 * float_array - Array to store converted floats
 * size - Number of floats to convert
 *
 * Takes 4 bytes at a time and converts to single float value
 * */
void ConvertToFloat(uint8_t* buffer, float* float_array, int size) {
	for (int i = 0; i < size; i++) {
		memcpy(&float_array[i], &buffer[i*4], sizeof(float));
	}
}

void ConvertToFloat_fast_grnn(uint8_t* buffer, float* float_array) {
	for (int i = 0; i < 256; i++) {
		memcpy(&float_array[i], &buffer[i*4], sizeof(float));
	}
}

void ConvertToDouble(uint8_t* buffer, double* double_array, int size) {
	for (int i = 0; i < size; i++) {
		memcpy(&double_array[i], &buffer[i*8], sizeof(double));
	}
}

uint8_t fast_grnn(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf){
    // Receive number to inject fault and option to select on board image or not
    if(scmd & 0x01){
        layer_fault_no = ((uint16_t)buf[0] << 8) | buf[1];
        //Option to select on-board images
        select_on_board_image_fast_grnn = buf[2];
        image_index_fast_grnn = buf[3]; // setting image index if used
    }

    // Send Image and save it in an image buffer
    if(scmd & 0x02){
        // copy this chunk into our buffer
        memcpy(image_buffer_fast_grnn + image_buffer_fast_grnn_offset, buf, len);
        image_buffer_fast_grnn_offset += len;
        
        // if not full yet, wait for more
        if (image_buffer_fast_grnn_offset < IMAGE_NUM_BYTES) {
            return 0x00;
        }
    }

    if(scmd & 0x04){
        if (select_on_board_image_fast_grnn == 0x00){
            memset(hiddenState, 0, sizeof(float) * HIDDEN_DIMS);
            unsigned n = 0; 
            ConvertToFloat_fast_grnn(image_buffer_fast_grnn, image_to_infer_fast_grnn);
            top_fast_grnn(layer_fault_no, hiddenState, HIDDEN_DIMS,
            image_to_infer_fast_grnn + n * INPUT_DIMS * TIME_STEPS, INPUT_DIMS, TIME_STEPS,
            &USPS_params, &buffers, 0, 1);
        }
        else{
            memset(hiddenState, 0, sizeof(float) * HIDDEN_DIMS);
            unsigned n = 0; // 
            image_select_fast_grnn(image_to_infer_fast_grnn_board, image_index_fast_grnn);
            top_fast_grnn(layer_fault_no, hiddenState, HIDDEN_DIMS,
            image_to_infer_fast_grnn_board + n * INPUT_DIMS * TIME_STEPS, INPUT_DIMS, TIME_STEPS,
            &USPS_params, &buffers, 0, 1);
        }

        if(layer_fault_no == 1040){
            trigger_high();
            FC(FC_weights, FCbias, hiddenState, HIDDEN_DIMS, classScores, NUM_CLASSES);
            trigger_low();
        }
        else{
            FC(FC_weights, FCbias, hiddenState, HIDDEN_DIMS, classScores, NUM_CLASSES);
        }
        predict_num_fast_grnn = argmax(classScores, NUM_CLASSES);
        simpleserial_put('p', sizeof(predict_num_fast_grnn), &predict_num_fast_grnn);
        // reset for the next image!
        image_buffer_fast_grnn_offset = 0;
    }
    if(scmd & 0x10){
        simpleserial_put('m', HIDDEN_DIMS * sizeof(float), (uint8_t*)buffers.preComp);
    }
    if(scmd & 0x20){
        simpleserial_put('h', HIDDEN_DIMS * sizeof(float), (uint8_t*)hiddenState);
    }
    if(scmd & 0x40){
        simpleserial_put('c', NUM_CLASSES * sizeof(float), (uint8_t*)classScores);
    }
    return 0x00;
}

uint8_t fast_grnn_protected(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf){
    // Receive number to inject fault and option to select on board image or not
    if(scmd & 0x01){
        layer_fault_no = ((uint16_t)buf[0] << 8) | buf[1];
        //Option to select on-board images
        select_on_board_image_fast_grnn = buf[2];
        image_index_fast_grnn = buf[3]; // setting image index if used
    }

    // Send Image and save it in an image buffer
    if(scmd & 0x02){
        // copy this chunk into our buffer
        memcpy(image_buffer_fast_grnn + image_buffer_fast_grnn_offset, buf, len);
        image_buffer_fast_grnn_offset += len;
        
        // if not full yet, wait for more
        if (image_buffer_fast_grnn_offset < IMAGE_NUM_BYTES) {
            return 0x00;
        }
    }

    if(scmd & 0x04){
        if (select_on_board_image_fast_grnn == 0x00){
            memset(hiddenState, 0, sizeof(float) * HIDDEN_DIMS);
            unsigned n = 0; 
            ConvertToFloat_fast_grnn(image_buffer_fast_grnn, image_to_infer_fast_grnn);
            top_fast_grnn_protected(layer_fault_no, hiddenState, HIDDEN_DIMS,
            image_to_infer_fast_grnn + n * INPUT_DIMS * TIME_STEPS, INPUT_DIMS, TIME_STEPS,
            &USPS_params, &buffers, 0, 1);
        }
        else{
            memset(hiddenState, 0, sizeof(float) * HIDDEN_DIMS);
            unsigned n = 0; // 
            image_select_fast_grnn(image_to_infer_fast_grnn_board, image_index_fast_grnn);
            top_fast_grnn_protected(layer_fault_no, hiddenState, HIDDEN_DIMS,
            image_to_infer_fast_grnn_board + n * INPUT_DIMS * TIME_STEPS, INPUT_DIMS, TIME_STEPS,
            &USPS_params, &buffers, 0, 1);
        }

        if(layer_fault_no == 1040){
            trigger_high();
            FC_protected(FC_weights, FCbias, hiddenState, HIDDEN_DIMS, classScores, NUM_CLASSES);
            trigger_low();
        }
        else{
            FC_protected(FC_weights, FCbias, hiddenState, HIDDEN_DIMS, classScores, NUM_CLASSES);
        }
        predict_num_fast_grnn = argmax(classScores, NUM_CLASSES);
        simpleserial_put('p', sizeof(predict_num_fast_grnn), &predict_num_fast_grnn);
        // reset for the next image!
        image_buffer_fast_grnn_offset = 0;
    }
    if(scmd & 0x10){
        simpleserial_put('m', HIDDEN_DIMS * sizeof(float), (uint8_t*)buffers.preComp);
    }
    if(scmd & 0x20){
        simpleserial_put('h', HIDDEN_DIMS * sizeof(float), (uint8_t*)hiddenState);
    }
    if(scmd & 0x40){
        simpleserial_put('c', NUM_CLASSES * sizeof(float), (uint8_t*)classScores);
    }
    return 0x00;
}

uint8_t logistic_regression_cw(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf){
    // Receive number to inject fault and option to select on board image or not
    if(scmd & 0x01){
        layer_fault_no = ((uint16_t)buf[0] << 8) | buf[1];
    }

    // Send data and save it in an data buffer
    if(scmd & 0x02){
        // copy this chunk into our buffer
        memcpy(input_data_buffer_log_reg + input_data_buffer_log_reg_offset, buf, len);
        input_data_buffer_log_reg_offset += len;
        
        // if not full yet, wait for more
        if (input_data_buffer_log_reg_offset < LOG_REG_DATA_NUM_BYTES) {
            return 0x00;
        }
    }

    if(scmd & 0x04){
        ConvertToDouble(input_data_buffer_log_reg, input_data_log_reg, LOG_REG_DATA_NUM_DOUBLE);
        logistic_regression(layer_fault_no, input_data_log_reg, &log_reg_class_pred);
        simpleserial_put('p', sizeof(log_reg_class_pred), &log_reg_class_pred);
        // reset for the next input data!
        input_data_buffer_log_reg_offset = 0;
    }
    
    if(scmd & 0x10){
        // debug per-feature products  X[j]*w[j]  (11 doubles)
        simpleserial_put('m', LOG_REG_DATA_NUM_DOUBLE * sizeof(double), (uint8_t*)g_lr_products);
    }

    if(scmd & 0x20){
        // Lets the host verify the input actually landed before arming/glitching.
        simpleserial_put('n', LOG_REG_DATA_NUM_BYTES, (uint8_t*)input_data_buffer_log_reg);
    }
    return 0x00;
}

uint8_t logistic_regression_cw_protected(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf){
    // Receive number to inject fault and option to select on board image or not
    if(scmd & 0x01){
        layer_fault_no = ((uint16_t)buf[0] << 8) | buf[1];
    }

    // Send data and save it in an data buffer
    if(scmd & 0x02){
        // copy this chunk into our buffer
        memcpy(input_data_buffer_log_reg + input_data_buffer_log_reg_offset, buf, len);
        input_data_buffer_log_reg_offset += len;
        
        // if not full yet, wait for more
        if (input_data_buffer_log_reg_offset < LOG_REG_DATA_NUM_BYTES) {
            return 0x00;
        }
    }

    if(scmd & 0x04){
        ConvertToDouble(input_data_buffer_log_reg, input_data_log_reg, LOG_REG_DATA_NUM_DOUBLE);
        logistic_regression_protected_new(layer_fault_no, input_data_log_reg, &log_reg_class_pred);
        simpleserial_put('p', sizeof(log_reg_class_pred), &log_reg_class_pred);
        // reset for the next input data!
        input_data_buffer_log_reg_offset = 0;
    }

    if(scmd & 0x20){
        simpleserial_put('n', LOG_REG_DATA_NUM_BYTES, (uint8_t*)input_data_buffer_log_reg);
    }
    return 0x00;
}

uint8_t wake_word(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf){
    // Receive number to inject fault and option to select on input data or not
    if(scmd & 0x01){
        layer_fault_no = ((uint16_t)buf[0] << 8) | buf[1];
        //Option to select on-board input data
        select_on_board_mfcc = buf[2];
        mfcc_index = buf[3]; // setting input data index if used
    }

    // Send Input and save it in an input buffer
    if(scmd & 0x02){
        // copy this chunk into our buffer
        memcpy(mfcc_buffer + mfcc_buffer_offset, buf, len);
        mfcc_buffer_offset += len;
        
        // if not full yet, wait for more
        if (mfcc_buffer_offset < WAKE_WORD_NUM_BYTES) {
            return 0x00;
        }
    }

    if(scmd & 0x04){
        if (select_on_board_mfcc == 0x00){
            ConvertToFloat(mfcc_buffer, mfcc_to_infer, WWD_INPUT_DIM);
        }
        else{
            mfcc_select(mfcc_to_infer, mfcc_index);
        }
        quantize_wwd_input(mfcc_to_infer, input_data_wwd, WWD_INPUT_DIM);
    
        run_wwd(layer_fault_no, input_data_wwd, output_data_wwd, 
            fc1_output, fc2_output, fc3_output, softmax_output);
    
        dequantize_wwd_output(output_data_wwd, out_pred_wwd, WWD_OUTPUT_DIM);

        // Detection
        out_pred_wwd_final = isWordDetected(out_pred_wwd);

        simpleserial_put('p', sizeof(out_pred_wwd_final), &out_pred_wwd_final);
        // reset for the next input data!
        mfcc_buffer_offset = 0;

    }

    if (scmd & 0x08) {
        static size_t offset = 0;
        int8_t *raw = (int8_t*)fc1_output;

        // How many bytes remain?
        size_t remaining = WWD_HIDDEN_DIM_1 - offset;
        size_t chunk     = remaining > MAX_BYTES_PER_PKT
                          ? MAX_BYTES_PER_PKT
                          : remaining;

        simpleserial_put('a', chunk, raw + offset);
        offset += chunk;

        // Once we’ve handed out all 256 bytes, wrap around
        if (offset >= WWD_HIDDEN_DIM_1) {
            offset = 0;
        }
    }
    
    if (scmd & 0x10) {
        static size_t offset = 0;
        int8_t *raw = (int8_t*)fc2_output;

        // How many bytes remain?
        size_t remaining = WWD_HIDDEN_DIM_2 - offset;
        size_t chunk     = remaining > MAX_BYTES_PER_PKT
                          ? MAX_BYTES_PER_PKT
                          : remaining;

        simpleserial_put('b', chunk, raw + offset);
        offset += chunk;

        // Once we’ve handed out all 256 bytes, wrap around
        if (offset >= WWD_HIDDEN_DIM_2) {
            offset = 0;
        }
    }
    
    if(scmd & 0x20){
        simpleserial_put('c', WWD_OUTPUT_DIM * sizeof(int8_t), (int8_t*)fc3_output);
    }
    if(scmd & 0x40){
        simpleserial_put('d', WWD_OUTPUT_DIM * sizeof(int8_t), (int8_t*)softmax_output);
    }
    if(scmd & 0x80){
        simpleserial_put('e', WWD_OUTPUT_DIM * sizeof(float), (uint8_t*)out_pred_wwd);
    }
    return 0x00;
}

uint8_t wake_word_protected(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf){
    // Receive number to inject fault and option to select on input data or not
    if(scmd & 0x01){
        layer_fault_no = ((uint16_t)buf[0] << 8) | buf[1];
        //Option to select on-board input data
        select_on_board_mfcc = buf[2];
        mfcc_index = buf[3]; // setting input data index if used
    }

    // Send Input and save it in an input buffer
    if(scmd & 0x02){
        // copy this chunk into our buffer
        memcpy(mfcc_buffer + mfcc_buffer_offset, buf, len);
        mfcc_buffer_offset += len;
        
        // if not full yet, wait for more
        if (mfcc_buffer_offset < WAKE_WORD_NUM_BYTES) {
            return 0x00;
        }
    }

    if(scmd & 0x04){
        if (select_on_board_mfcc == 0x00){
            ConvertToFloat(mfcc_buffer, mfcc_to_infer, WWD_INPUT_DIM);
        }
        else{
            mfcc_select(mfcc_to_infer, mfcc_index);
        }
        quantize_wwd_input(mfcc_to_infer, input_data_wwd, WWD_INPUT_DIM);
    
        run_wwd_protected(layer_fault_no, input_data_wwd, output_data_wwd_protected, 
            fc1_output, fc2_output, fc3_output, softmax_output);

        int8_t output_data_wwd_protected_new[WWD_OUTPUT_DIM] = {0};
        for(int i=0; i<WWD_OUTPUT_DIM; i++){
            output_data_wwd_protected_new[i] = output_data_wwd_protected[i];
        }
        
        dequantize_wwd_output(output_data_wwd_protected_new, out_pred_wwd, WWD_OUTPUT_DIM);

        // Detection
        out_pred_wwd_final = isWordDetected(out_pred_wwd);

        simpleserial_put('p', sizeof(out_pred_wwd_final), &out_pred_wwd_final);
        // reset for the next input data!
        mfcc_buffer_offset = 0;

    }

    if (scmd & 0x08) {
        static size_t offset = 0;
        int8_t *raw = (int8_t*)fc1_output;

        // How many bytes remain?
        size_t remaining = WWD_HIDDEN_DIM_1 - offset;
        size_t chunk     = remaining > MAX_BYTES_PER_PKT
                          ? MAX_BYTES_PER_PKT
                          : remaining;

        simpleserial_put('a', chunk, raw + offset);
        offset += chunk;

        // Once we’ve handed out all 256 bytes, wrap around
        if (offset >= WWD_HIDDEN_DIM_1) {
            offset = 0;
        }
    }
    
    if (scmd & 0x10) {
        static size_t offset = 0;
        int8_t *raw = (int8_t*)fc2_output;

        // How many bytes remain?
        size_t remaining = WWD_HIDDEN_DIM_2 - offset;
        size_t chunk     = remaining > MAX_BYTES_PER_PKT
                          ? MAX_BYTES_PER_PKT
                          : remaining;

        simpleserial_put('b', chunk, raw + offset);
        offset += chunk;

        // Once we’ve handed out all 256 bytes, wrap around
        if (offset >= WWD_HIDDEN_DIM_2) {
            offset = 0;
        }
    }
    
    if(scmd & 0x20){
        simpleserial_put('c', WWD_OUTPUT_DIM * sizeof(int8_t), (int8_t*)fc3_output);
    }
    if(scmd & 0x40){
        simpleserial_put('d', WWD_OUTPUT_DIM * sizeof(int8_t), (int8_t*)softmax_output);
    }
    if(scmd & 0x80){
        simpleserial_put('e', WWD_OUTPUT_DIM * sizeof(float), (uint8_t*)out_pred_wwd);
    }
    return 0x00;
}

uint8_t tiny_cnn(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf){
    // Receive number to inject fault and option to select on input data or not
    if(scmd & 0x01){
        layer_fault_no = ((uint16_t)buf[0] << 8) | buf[1];
        //Option to select on-board input data
        select_on_board_image = buf[2];
        image_index = buf[3]; // setting input data index if used
        // A fresh 0x01 command always precedes a new image in the dataset path,
        // so reset the write offset to guarantee each image starts at byte 0.
        image_buffer_offset = 0;
    }

    // Send Input and save it in an input buffer
    if(scmd & 0x02){
        // Guard against overflow: if a resend (host verify/retry) or an extra
        // packet would run past the 256-byte buffer, wrap back to the start so a
        // full image still lands cleanly instead of corrupting adjacent memory.
        if (image_buffer_offset + len > IMAGE_CNN_NUM_BYTES) {
            image_buffer_offset = 0;
        }
        // copy this chunk into our buffer
        memcpy(image_buffer + image_buffer_offset, buf, len);
        image_buffer_offset += len;
        
        // if not full yet, wait for more
        if (image_buffer_offset < IMAGE_CNN_NUM_BYTES) {
            return 0x00;
        }
    }

    if(scmd & 0x04){
        if (select_on_board_image == 0x00){
            ConvertToFloat(image_buffer, image_to_infer, INPUT_DIM_CNN);
        }
        else{
            image_select(image_to_infer, image_index);
        }
        
		quantize_input_cnn(image_to_infer, input_data_cnn, INPUT_DIM_CNN);
    
		run_imgclass_inference(layer_fault_no, input_data_cnn, output_data_cnn,
            convolution_output, maxpool_output, fc_output);
    
		dequantize_output_cnn(output_data_cnn, out_pred_cnn, OUTPUT_DIM_CNN);

        // Prediction
        predict_num = predicted_number(out_pred_cnn, 10);

        simpleserial_put('p', sizeof(predict_num), &predict_num);
        // reset for the next input data!
        image_buffer_offset = 0;
    }

    if (scmd & 0x08) {
        static size_t offset = 0;
        int8_t *raw = (int8_t*)convolution_output;

        // How many bytes remain?
        size_t remaining = HIDDEN_DIM_1_CNN - offset;
        size_t chunk     = remaining > MAX_BYTES_PER_PKT_CNN
                          ? MAX_BYTES_PER_PKT_CNN
                          : remaining;

        simpleserial_put('a', chunk, raw + offset);
        offset += chunk;

        // Once we’ve handed out all 288 bytes, wrap around
        if (offset >= HIDDEN_DIM_1_CNN) {
            offset = 0;
        }
    }
    
    if(scmd & 0x10){
        simpleserial_put('b', HIDDEN_DIM_2_CNN * sizeof(int8_t), (int8_t*)maxpool_output);
    }
    if(scmd & 0x20){
        simpleserial_put('c', OUTPUT_DIM_CNN * sizeof(int8_t), (int8_t*)fc_output);
    }
    if(scmd & 0x40){
        simpleserial_put('d', OUTPUT_DIM_CNN * sizeof(float), (uint8_t*)out_pred_cnn);
    }
    return 0x00;
}

uint8_t tiny_cnn_protected(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf){
    // Receive number to inject fault and option to select on input data or not
    if(scmd & 0x01){
        layer_fault_no = ((uint16_t)buf[0] << 8) | buf[1];
        //Option to select on-board input data
        select_on_board_image = buf[2];
        image_index = buf[3]; // setting input data index if used
        // A fresh 0x01 command always precedes a new image in the dataset path,
        // so reset the write offset to guarantee each image starts at byte 0.
        image_buffer_offset = 0;
    }

    // Send Input and save it in an input buffer
    if(scmd & 0x02){
        // Guard against overflow: if a resend (host verify/retry) or an extra
        // packet would run past the 256-byte buffer, wrap back to the start so a
        // full image still lands cleanly instead of corrupting adjacent memory.
        if (image_buffer_offset + len > IMAGE_CNN_NUM_BYTES) {
            image_buffer_offset = 0;
        }
        // copy this chunk into our buffer
        memcpy(image_buffer + image_buffer_offset, buf, len);
        image_buffer_offset += len;
        
        // if not full yet, wait for more
        if (image_buffer_offset < IMAGE_CNN_NUM_BYTES) {
            return 0x00;
        }
    }

    if(scmd & 0x04){
        if (select_on_board_image == 0x00){
            ConvertToFloat(image_buffer, image_to_infer, INPUT_DIM_CNN);
        }
        else{
            image_select(image_to_infer, image_index);
        }
        
		quantize_input_cnn(image_to_infer, input_data_cnn, INPUT_DIM_CNN);
    
		run_imgclass_inference_protected(layer_fault_no, input_data_cnn, output_data_cnn_protected,
            convolution_output, maxpool_output, fc_output);

        int8_t output_data_cnn_protected_new[OUTPUT_DIM_CNN] = {0};
        for(int i=0; i<OUTPUT_DIM_CNN; i++){
            output_data_cnn_protected_new[i] = output_data_cnn_protected[i];
        }
        
		dequantize_output_cnn(output_data_cnn_protected_new, out_pred_cnn, OUTPUT_DIM_CNN);

        // Prediction
        predict_num = predicted_number(out_pred_cnn, 10);

        simpleserial_put('p', sizeof(predict_num), &predict_num);
        // reset for the next input data!
        image_buffer_offset = 0;
    }

    
    if (scmd & 0x08) {
        static size_t offset = 0;
        int8_t *raw = (int8_t*)convolution_output;

        // How many bytes remain?
        size_t remaining = HIDDEN_DIM_1_CNN - offset;
        size_t chunk     = remaining > MAX_BYTES_PER_PKT_CNN
                          ? MAX_BYTES_PER_PKT_CNN
                          : remaining;

        simpleserial_put('a', chunk, raw + offset);
        offset += chunk;

        // Once we’ve handed out all 288 bytes, wrap around
        if (offset >= HIDDEN_DIM_1_CNN) {
            offset = 0;
        }
    }
    
    if(scmd & 0x10){
        simpleserial_put('b', HIDDEN_DIM_2_CNN * sizeof(int8_t), (int8_t*)maxpool_output);
    }
    if(scmd & 0x20){
        simpleserial_put('c', OUTPUT_DIM_CNN * sizeof(int8_t), (int8_t*)fc_output);
    }
    if(scmd & 0x40){
        simpleserial_put('d', OUTPUT_DIM_CNN * sizeof(float), (uint8_t*)out_pred_cnn);
    }
    return 0x00;
}

int main(void)
{
    platform_init();
	init_uart();
	trigger_setup();

	simpleserial_init();

    simpleserial_addcmd(0x01, 252, fast_grnn);
    simpleserial_addcmd(0x02, 252, fast_grnn_protected);
    simpleserial_addcmd(0x03, 252, logistic_regression_cw);
    simpleserial_addcmd(0x04, 252, logistic_regression_cw_protected);
    simpleserial_addcmd(0x05, 252, wake_word);
    simpleserial_addcmd(0x06, 252, wake_word_protected);
    simpleserial_addcmd(0x07, 252, tiny_cnn);
    simpleserial_addcmd(0x08, 252, tiny_cnn_protected);
    
	while(1)
		simpleserial_get();
}
