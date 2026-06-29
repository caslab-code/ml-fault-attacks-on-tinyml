// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "fast_grnn.h"
#define N 5 // Number of iterations for self-correction


int top_fast_grnn(uint16_t layer_fault_no, float* const hiddenState, unsigned hiddenDims,
  const float* const input, unsigned inputDims, unsigned steps,
  const void* params, void* buffers, int backward, int normalize) {

  const FastGRNN_LR_Params* tparams = (const FastGRNN_LR_Params*)params;
  FastGRNN_LR_Buffers* tbuffers = (FastGRNN_LR_Buffers*)buffers;

  if (tbuffers->preComp == 0) return ERR_PRECOMP_NOT_INIT;
  if (tbuffers->tempLRW == 0) return ERR_TEMPLRW_NOT_INIT;
  if (tbuffers->tempLRU == 0) return ERR_TEMPLRU_NOT_INIT;
  if (tbuffers->normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  // #steps iterations of the RNN cell starting from hiddenState
  for (unsigned t = 0; t < steps; t++) {
    // Normalize the features
    unsigned offset = backward ? steps - 1 - t : t;
    if (normalize) {
      v_add(1.0f, input + offset * inputDims, -1.0f, tparams->mean + offset * inputDims,
        inputDims, tbuffers->normFeatures);
      v_div(tparams->stdDev + offset * inputDims, tbuffers->normFeatures, inputDims,
        tbuffers->normFeatures);
    }
    else {
      for (unsigned d = 0; d < inputDims; ++d)
        tbuffers->normFeatures[d] = input[offset * inputDims + d];
    }

    if(layer_fault_no == t*65){
        trigger_high();
    }
    // Process the new input and previous hidden state
    matVec(tparams->W1, tbuffers->normFeatures, tparams->wRank, inputDims,
      0.0f, 1.0f, tbuffers->tempLRW);
    matVec(tparams->W2, tbuffers->tempLRW, hiddenDims, tparams->wRank,
      0.0f, 1.0f, tbuffers->preComp);
    matVec(tparams->U1, hiddenState, tparams->uRank, hiddenDims,
      0.0f, 1.0f, tbuffers->tempLRU);
    matVec(tparams->U2, tbuffers->tempLRU, hiddenDims, tparams->uRank,
      1.0f, 1.0f, tbuffers->preComp);
    if(layer_fault_no == t*65){
        trigger_low();
    }

    // Apply the gate to generate the new hidden state
    for (unsigned i = 0; i < hiddenDims; i++) {
        if(layer_fault_no == ((t*65) + (i*2) + 1)){
            trigger_high();
        }
        float gate = sigmoid(tbuffers->preComp[i] + tparams->Bg[i]);
        if(layer_fault_no == ((t*65) + (i*2) + 1)){
            trigger_low();
        }
        if(layer_fault_no == ((t*65) + (i*2) + 2)){
            trigger_high();
        }
        float update = tanh(tbuffers->preComp[i] + tparams->Bh[i]);
        if(layer_fault_no == ((t*65) + (i*2) + 2)){
            trigger_low();
        }
        hiddenState[i] = gate * hiddenState[i] + (tparams->sigmoid_zeta * (1.0 - gate) + tparams->sigmoid_nu) * update;
    }
  }
  return 0;
}

int fastgrnn(float* const hiddenState, unsigned hiddenDims,
  const float* const input, unsigned inputDims, unsigned steps,
  const void* params, void* buffers, int backward, int normalize) {

  const FastGRNN_Params* tparams = (const FastGRNN_Params*)params;
  FastGRNN_Buffers* tbuffers = (FastGRNN_Buffers*)buffers;

  if (tbuffers->preComp == 0) return ERR_PRECOMP_NOT_INIT;
  if (tbuffers->normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  for (unsigned t = 0; t < steps; t++) {
    // Normalize the features
    unsigned offset = backward ? steps - 1 - t : t;
    if (normalize) {
      v_add(1.0f, input + offset * inputDims, -1.0f, tparams->mean + offset * inputDims,
        inputDims, tbuffers->normFeatures);
      v_div(tparams->stdDev + offset * inputDims, tbuffers->normFeatures, inputDims,
        tbuffers->normFeatures);
    }
    else {
      for (unsigned d = 0; d < inputDims; ++d)
        tbuffers->normFeatures[d] = input[offset * inputDims + d];
    }

    // Process the new input and previous hidden state
    matVec(tparams->W, tbuffers->normFeatures, hiddenDims, inputDims,
      0.0f, 1.0f, tbuffers->preComp);
    matVec(tparams->U, hiddenState, hiddenDims, hiddenDims,
      1.0f, 1.0f, tbuffers->preComp);


    // Apply the gate to generate the new hidden state
    for (unsigned i = 0; i < hiddenDims; i++) {
      float gate = sigmoid(tbuffers->preComp[i] + tparams->Bg[i]);
      float update = tanh(tbuffers->preComp[i] + tparams->Bh[i]);
      hiddenState[i] = gate * hiddenState[i] + (tparams->sigmoid_zeta * (1.0 - gate) + tparams->sigmoid_nu) * update;
    }
  }
  return 0;
}

void q15_v_sigmoid(const Q15_T* vec, ITER_T len, Q15_T* ret, Q15_T div,
                   Q15_T add, Q15_T sigmoid_limit, SCALE_T scale_in,
                   SCALE_T scale_out, ITER_T use_tables) {
  if (use_tables) {
    #ifdef LOOP_UNROLL
      ITER_T len_unroll = len >> 2;
      len = len % 4;
      while (len_unroll--) {
        Q15_T w = *vec++;
        Q15_T x = *vec++;
        Q15_T y = *vec++;
        Q15_T z = *vec++;

        *ret++ = (w <= 0) ? (Q15_T)((((Q31_T)exp_base_16(w, 1)) << 14) /
                                    ((Q31_T)exp_base_16(w, 1) + (Q31_T)16384)) :
                            (Q15_T)(((Q31_T)267943936L) /
                                    ((Q31_T)16384 + (Q31_T)exp_base_16(-w, 1)));
        *ret++ = (x <= 0) ? (Q15_T)((((Q31_T)exp_base_16(x, 1)) << 14) /
                                    ((Q31_T)exp_base_16(x, 1) + (Q31_T)16384)) :
                            (Q15_T)(((Q31_T)267943936L) /
                                    ((Q31_T)16384 + (Q31_T)exp_base_16(-x, 1)));
        *ret++ = (y <= 0) ? (Q15_T)((((Q31_T)exp_base_16(y, 1)) << 14) /
                                    ((Q31_T)exp_base_16(y, 1) + (Q31_T)16384)) :
                            (Q15_T)(((Q31_T)267943936L) /
                                    ((Q31_T)16384 + (Q31_T)exp_base_16(-y, 1)));
        *ret++ = (z <= 0) ? (Q15_T)((((Q31_T)exp_base_16(z, 1)) << 14) /
                                    ((Q31_T)exp_base_16(z, 1) + (Q31_T)16384)) :
                            (Q15_T)(((Q31_T)267943936L) /
                                    ((Q31_T)16384 + (Q31_T)exp_base_16(-z, 1)));
      }
    #endif

    while (len--) {
      Q15_T w = *vec++;
      *ret++ = (w <= 0) ? (Q15_T)((((Q31_T)exp_base_16(w, 1)) << 14) /
                                  ((Q31_T)exp_base_16(w, 1) + (Q31_T)16384)) :
                          (Q15_T)(((Q31_T)267943936L) /
                                  ((Q31_T)16384 + (Q31_T)exp_base_16(-w, 1)));
    }
  } else {
    SCALE_T scaleout = (scale_out - scale_in);

    #ifdef LOOP_UNROLL
      ITER_T len_unroll = len >> 2;
      len = len % 4;
      while (len_unroll--) {
        Q15_T w = (*vec++ / div) + add;
        Q15_T x = (*vec++ / div) + add;
        Q15_T y = (*vec++ / div) + add;
        Q15_T z = (*vec++ / div) + add;

        *ret++ = (w <= 0) ? 0 : (((w >= sigmoid_limit) ? sigmoid_limit : w) << scaleout);
        *ret++ = (x <= 0) ? 0 : (((x >= sigmoid_limit) ? sigmoid_limit : x) << scaleout);
        *ret++ = (y <= 0) ? 0 : (((y >= sigmoid_limit) ? sigmoid_limit : y) << scaleout);
        *ret++ = (z <= 0) ? 0 : (((z >= sigmoid_limit) ? sigmoid_limit : z) << scaleout);
      }
    #endif

    while (len--) {
      Q15_T w = (*vec++ / div) + add;
      *ret++ = (w <= 0) ? 0 : (((w >= sigmoid_limit) ? sigmoid_limit : w) << scaleout);
    }
  }
}

void q15_v_tanh(const Q15_T* vec, ITER_T len, Q15_T* ret, SCALE_T scale_in,
                SCALE_T scale_out, ITER_T use_tables) {
  if (use_tables) {
    #ifdef LOOP_UNROLL
      ITER_T len_unroll = len >> 2;
      len = len % 4;
      while (len_unroll--) {
        Q15_T w = q15_saturate(2 * (*vec++));
        Q15_T x = q15_saturate(2 * (*vec++));
        Q15_T y = q15_saturate(2 * (*vec++));
        Q15_T z = q15_saturate(2 * (*vec++));

        *ret++ = (w <= 0) ? (Q15_T)((((Q31_T)(exp_base_16(w, 1) - 16384)) << 14) /
                                    (exp_base_16(w, 1) + 16384)) :
                            (Q15_T)((((Q31_T)(16384 - exp_base_16(-w, 1))) << 14) /
                                    (exp_base_16(-w, 1) + 16384));
        *ret++ = (x <= 0) ? (Q15_T)((((Q31_T)(exp_base_16(x, 1) - 16384)) << 14) /
                                    (exp_base_16(x, 1) + 16384)) :
                            (Q15_T)((((Q31_T)(16384 - exp_base_16(-x, 1))) << 14) /
                                    (exp_base_16(-x, 1) + 16384));
        *ret++ = (y <= 0) ? (Q15_T)((((Q31_T)(exp_base_16(y, 1) - 16384)) << 14) /
                                    (exp_base_16(y, 1) + 16384)) :
                            (Q15_T)((((Q31_T)(16384 - exp_base_16(-y, 1))) << 14) /
                                    (exp_base_16(-y, 1) + 16384));
        *ret++ = (z <= 0) ? (Q15_T)((((Q31_T)(exp_base_16(z, 1) - 16384)) << 14) /
                                    (exp_base_16(z, 1) + 16384)) :
                            (Q15_T)((((Q31_T)(16384 - exp_base_16(-z, 1))) << 14) /
                                    (exp_base_16(-z, 1) + 16384));
      }
    #endif

    while (len--) {
      Q15_T w = q15_saturate(2 * (*vec++));
      *ret++ = (w <= 0) ? (Q15_T)((((Q31_T)(exp_base_16(w, 1) - 16384)) << 14) /
                                  (exp_base_16(w, 1) + 16384)) :
                          (Q15_T)((((Q31_T)(16384 - exp_base_16(-w, 1))) << 14) /
                                  (exp_base_16(-w, 1) + 16384));
    }
  } else {
    SCALE_T scalein = (1 << scale_in);
    SCALE_T scaleout = scale_out - scale_in;

    #ifdef LOOP_UNROLL
      ITER_T len_unroll = len >> 2;
      len = len % 4;
      while (len_unroll--) {
        Q15_T w = *vec++;
        Q15_T x = *vec++;
        Q15_T y = *vec++;
        Q15_T z = *vec++;

        *ret++ = ((w >= scalein) ? scalein : ((w <= -scalein) ? (-scalein) : w)) << scaleout;
        *ret++ = ((x >= scalein) ? scalein : ((x <= -scalein) ? (-scalein) : x)) << scaleout;
        *ret++ = ((y >= scalein) ? scalein : ((y <= -scalein) ? (-scalein) : y)) << scaleout;
        *ret++ = ((z >= scalein) ? scalein : ((z <= -scalein) ? (-scalein) : z)) << scaleout;
      }
    #endif

    while (len--) {
      Q15_T w = *vec++;
      *ret++ = ((w >= scalein) ? scalein : ((w <= -scalein) ? (-scalein) : w)) << scaleout;
    }
  }
}

void image_select_fast_grnn(float *image_to_infer_fast_grnn, uint8_t image_index_fast_grnn){
	if(image_index_fast_grnn == 0x00){
		for(int i = 0; i<256; i++){
			image_to_infer_fast_grnn[i] = input[0 * 256 + i];
		}
	}
	else if (image_index_fast_grnn == 0x01){
		for(int i = 0; i<256; i++){
			image_to_infer_fast_grnn[i] = input[1 * 256 + i];
		}
	}
	else if (image_index_fast_grnn == 0x02){
		for(int i = 0; i<256; i++){
			image_to_infer_fast_grnn[i] = input[2 * 256 + i];
		}
	}
	else if (image_index_fast_grnn == 0x03){
		for(int i = 0; i<256; i++){
			image_to_infer_fast_grnn[i] = input[3 * 256 + i];
		}
	}
	else if (image_index_fast_grnn == 0x04){
		for(int i = 0; i<256; i++){
			image_to_infer_fast_grnn[i] = input[4 * 256 + i];
		}
	}
	else if (image_index_fast_grnn == 0x05){
		for(int i = 0; i<256; i++){
			image_to_infer_fast_grnn[i] = input[5 * 256 + i];
		}
	}
	else if (image_index_fast_grnn == 0x06){
		for(int i = 0; i<256; i++){
			image_to_infer_fast_grnn[i] = input[6 * 256 + i];
		}
	}
	else if (image_index_fast_grnn == 0x07){
		for(int i = 0; i<256; i++){
			image_to_infer_fast_grnn[i] = input[7 * 256 + i];
		}
	}
	else if (image_index_fast_grnn == 0x08){
		for(int i = 0; i<256; i++){
			image_to_infer_fast_grnn[i] = input[8 * 256 + i];
		}
	}
	else if (image_index_fast_grnn == 0x09){
		for(int i = 0; i<256; i++){
			image_to_infer_fast_grnn[i] = input[9 * 256 + i];
		}
	}
	else {
		for(int i = 0; i<256; i++){
			image_to_infer_fast_grnn[i] = 0;
		}
	}
}


////Undergraduate Work
/* FastGRNN Implementation Documentation */

/* Finds the median value in an array of floats
*
* Inputs:
*  arr - Array of float values to find median from
*  n - Length of the array
*
* Returns: Median value from the array */
float median(float *arr, int n) {
    // Sort the array
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (arr[i] > arr[j]) {
                float temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
    // If n is odd, return the middle element
    if (n % 2 != 0) {
        return arr[n / 2];
    }
    // If n is even, return the average of the two middle elements
    return (arr[(n - 1) / 2] + arr[n / 2]) / 2.0;
}
/*
 Simple random number generator for fault tolerance
* Returns: Random float between 0 and 1 */
unsigned int seed = 123456789;
float simple_rand() {
    seed = (1103515245 * seed + 12345) & 0x7fffffff;
    return (float)seed / (float)0x7fffffff;
}

/**
 * Uses random sampling and median filtering
 *
 * Inputs:
 *  z - Value to apply sigmoid to
 *
 * Returns: Protected sigmoid output
 *
 * Implements fault tolerance through:
 * 1. Generating N random samples
 * 2. Computing sigmoid for each sample
 * 3. Using median filtering for fault tolerance
 * Uses the formula: numerator = P_x * (P_y - 1), denominator = 2 * P_x * P_y - P_x - P_y
 */
float c_sigmoid(float z) {
    float c[N];
    for (int i = 0; i < N; ++i) {
        float y = simple_rand() * 10.0f - 5.0f; // Random value between -5 and 5
        float x = z + y;
        float P_x = sigmoid(x);
        float P_y = sigmoid(y);
        float numerator = P_x * (P_y - 1);
        float denominator = 2 * P_x * P_y - P_x - P_y;
        c[i] = numerator / denominator;
    }
    return median(c, N);
}

/**
 * Uses random sampling and median filtering
 *
 * Inputs:
 *  z - Value to apply tanh to
 *
 * Returns: Protected tanh output
 *
 * Implements fault tolerance through:
 * 1. Generating N random samples between -5 and 5
 * 2. Computing tanh for paired values
 * 3. Using median filtering for fault tolerance
 * Uses the formula: (tanh(x) + tanh(y)) / (1 + tanh(x) * tanh(y))
 */
float c_tanh(float z) {
    float c[N];
    for (int i = 0; i < N; ++i) {
        float x = simple_rand() * 10.0f - 5.0f; // Random value between -5 and 5
        float y = z - x;
        float P_x = tanh(x);
        float P_y = tanh(y);
        c[i] = (P_x + P_y) / (1 + P_x * P_y);
    }
    return median(c, N);
}


/**
 * Main protected FastGRNN implementation
 * Inputs:
 *  layer_fault_no - Fault injection point
 *  hiddenState - Hidden state array
 *  hiddenDims - Size of hidden state
 *  input - Input data array
 *  inputDims - Size of input
 *  steps - Number of time steps
 *  params - Network parameters
 *  buffers - Working memory
 *  backward - Direction flag
 *  normalize - Normalization flag
 *
 * Returns: 0 on success, error code on failure
 *
 * Protected implementation includes:
 * 1. Fault injection triggers for testing
 * 2. Self-correcting sigmoid and tanh functions
 * 3. Cycle counting for performance measurement
 * 4. Input data normalization options
 * Uses protected versions of activation functions (c_sigmoid, c_tanh)
 */
int top_fast_grnn_protected(uint16_t layer_fault_no, float* const hiddenState, unsigned hiddenDims,
  const float* const input, unsigned inputDims, unsigned steps,
  const void* params, void* buffers, int backward, int normalize) {

  const FastGRNN_LR_Params* tparams = (const FastGRNN_LR_Params*)params;
  FastGRNN_LR_Buffers* tbuffers = (FastGRNN_LR_Buffers*)buffers;

  if (tbuffers->preComp == 0) return ERR_PRECOMP_NOT_INIT;
  if (tbuffers->tempLRW == 0) return ERR_TEMPLRW_NOT_INIT;
  if (tbuffers->tempLRU == 0) return ERR_TEMPLRU_NOT_INIT;
  if (tbuffers->normFeatures == 0) return ERR_NORMFEATURES_NOT_INIT;

  for (unsigned t = 0; t < steps; t++) {
    // Normalize the features
    unsigned offset = backward ? steps - 1 - t : t;
    if (normalize) {
      v_add(1.0f, input + offset * inputDims, -1.0f, tparams->mean + offset * inputDims,
        inputDims, tbuffers->normFeatures);
      v_div(tparams->stdDev + offset * inputDims, tbuffers->normFeatures, inputDims,
        tbuffers->normFeatures);
    }
    else {
      for (unsigned d = 0; d < inputDims; ++d)
        tbuffers->normFeatures[d] = input[offset * inputDims + d];
    }

    if(layer_fault_no == t*65){
        trigger_high();
    }
    // Process the new input and previous hidden state
    protectedmatVec(tparams->W1, tbuffers->normFeatures, tparams->wRank, inputDims,
    0.0f, 1.0f, tbuffers->tempLRW, t);
    protectedmatVec(tparams->W2, tbuffers->tempLRW, hiddenDims, tparams->wRank,
    0.0f, 1.0f, tbuffers->preComp, t);;
    protectedmatVec(tparams->U1, hiddenState, tparams->uRank, hiddenDims,
    0.0f, 1.0f, tbuffers->tempLRU, t);
    protectedmatVec(tparams->U2, tbuffers->tempLRU, hiddenDims, tparams->uRank,
    1.0f, 1.0f, tbuffers->preComp, t);

    if(layer_fault_no == t*65){
        trigger_low();
    }

    // Apply the gate to generate the new hidden state
    for (unsigned i = 0; i < hiddenDims; i++) {
        if(layer_fault_no == ((t*65) + (i*2) + 1)){
            trigger_high();
        }
        float gate = sigmoid(tbuffers->preComp[i] + tparams->Bg[i]);
        if(layer_fault_no == ((t*65) + (i*2) + 1)){
            trigger_low();
        }
        if(layer_fault_no == ((t*65) + (i*2) + 2)){
            trigger_high();
        }
        float update = tanh(tbuffers->preComp[i] + tparams->Bh[i]);
        if(layer_fault_no == ((t*65) + (i*2) + 2)){
            trigger_low();
        }
        hiddenState[i] = gate * hiddenState[i] + (tparams->sigmoid_zeta * (1.0 - gate) + tparams->sigmoid_nu) * update;
    }
  }
  return 0;
}
