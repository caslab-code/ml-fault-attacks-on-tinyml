#include "logistic_regression.h"
#include "weights.h"
#include "math.h"
#define N 30 // Number of iterations for self-correction

uint32_t execution_cycle;
uint8_t cycle[4];

// Captured intermediate from the unprotected predict path (read out over SimpleSerial).
// Populated in logistic_regression_predict(); persists after inference so the host can
// fetch it with the 0x10 sub-command in tiny-ml.c.
#define LR_N_FEATURES 11                    // input feature count (one product per feature)
double g_lr_products[LR_N_FEATURES] = {0};  // per-feature terms:  X[j] * w[j]


/**
 * @brief Basic sigmoid activation function
 * @param x Input value
 * @return Sigmoid output: 1 / (1 + e^(-x))
 */
double sigmoid_logistic_regression(double x) { return 1.0 / (1.0 + exp(-x)); }

void matrix_multiply(double *a, double *b, double *result, int a_rows,
                     int a_cols, int b_cols) {
  for (int i = 0; i < a_rows; i++) {
    for (int j = 0; j < b_cols; j++) {
      result[i * b_cols + j] = 0;
      for (int k = 0; k < a_cols; k++) {
        result[i * b_cols + j] += a[i * a_cols + k] * b[k * b_cols + j];
      }
    }
  }
}

void update_weights_bias(double *X, double *y, double *weights, double *bias,
                         int n_samples, int n_features, double lr) {
  double *predictions = (double *)malloc(n_samples * sizeof(double));
  double *errors = (double *)malloc(n_samples * sizeof(double));

  // Calculate predictions
  for (int i = 0; i < n_samples; i++) {
    predictions[i] = 0;
    for (int j = 0; j < n_features; j++) {
      predictions[i] += X[i * n_features + j] * weights[j];
    }
    predictions[i] += *bias;
    predictions[i] = sigmoid_logistic_regression(predictions[i]);
  }

  // Calculate gradient
  for (int j = 0; j < n_features; j++) {
    double gradient = 0;
    for (int i = 0; i < n_samples; i++) {
      errors[i] = predictions[i] - y[i];
      gradient += X[i * n_features + j] * errors[i];
    }
    gradient /= n_samples;
    weights[j] -= lr * gradient;
  }

  // Update bias
  double bias_gradient = 0;
  for (int i = 0; i < n_samples; i++) {
    bias_gradient += errors[i];
  }
  *bias -= lr * (bias_gradient / n_samples);

  free(predictions);
  free(errors);
}

void logistic_regression_fit(double *X, double *y, double *weights,
                             double *bias, int n_samples, int n_features,
                             int n_iters, double lr) {
  for (int i = 0; i < n_iters; i++) {
    update_weights_bias(X, y, weights, bias, n_samples, n_features, lr);
  }
}

void logistic_regression_predict(double *X, double *weights, double bias,
                                 uint8_t *log_reg_class_pred, int n_samples,
                                 int n_features) {
  for (int i = 0; i < n_samples; i++) {
    double linear_pred = 0;
    for (int j = 0; j < n_features; j++) {
      double prod = X[i * n_features + j] * weights[j];
      g_lr_products[j] = prod;   // capture per-feature contribution X[j]*w[j]
      linear_pred += prod;
    }
    linear_pred += bias;
    log_reg_class_pred[i] = sigmoid_logistic_regression(linear_pred) > 0.5 ? 1 : 0;
  }
}

void logistic_regression (uint8_t layer_fault_no, double *X_predict_data, uint8_t* log_reg_class_pred) {
  const int n_features = 11;  // Number of features
  const int n_predict = 1;  // Number of samples to predict

  double *X_predict_flat = (double *)malloc(n_predict * n_features * sizeof(double));
  for (int i = 0; i < n_features; i++) {
    X_predict_flat[i] = X_predict_data[i];
  }

  // Predict the classes for new data
  // int *log_reg_class_pred = (int *)malloc(n_predict * sizeof(int));
  if(layer_fault_no == 0x04){
      trigger_high();
  }
  logistic_regression_predict(X_predict_flat, log_reg_weights, log_reg_bias, log_reg_class_pred,
                              n_predict, n_features);
    
  if(layer_fault_no == 0x04){
      trigger_low();
  }

// The caller is responsible to clear the scratch buffers for security reasons if applicable.
  free(X_predict_flat);
}



// Undergraduate Work by Dustin Mazza

/**
 * Implementation of Logistic Regression with fault tolerance mechanisms
 *
 * This file contains multiple implementations of logistic regression:
 * 1. Standard implementation
 * 2. Protected implementation with fault-tolerant sigmoid
 * 3. MVP (Matrix-Vector Product) protected implementation
 */


/**
 * Protected random number generator for logistic regression
 * Returns: Random double between 0 and 1
 *
 * Uses XOR shift algorithm for efficiency:
 * 1. Left shift by 13 bits and XOR
 * 2. Right shift by 17 bits and XOR
 * 3. Left shift by 5 bits and XOR
 */
static unsigned int seed = 123456789;
double simple_rand_lr(void) {
    seed = (seed << 13) ^ seed;
    seed = (seed >> 17) ^ seed;
    seed = (seed << 5) ^ seed;
    return (double)(seed & 0x7fffffff) / (double)0x7fffffff;
}


/**
 * Inputs:
 *  arr - Array of values
 *  n - Array length
 *  Returns: Median value
 *
 * Uses bubble sort then returns:
 * - For odd n: middle element
 * - For even n: average of two middle elements
 */
double median_lr(double *arr, int n) {
    // Sort array for median calculation
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (arr[i] > arr[j]) {
                double temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }

    // Return median value
    if (n % 2 != 0) {
        return arr[n / 2];
    }
    return (arr[(n - 1) / 2] + arr[n / 2]) / 2.0;
}

/**
 *
 * Implements N-iteration protection with:
 * 1. Random sampling for error detection
 * 2. Median filtering for error correction
 * Uses mathematical identity: sigmoid(x+y) = [sigmoid(x)(sigmoid(y)-1)]/[2sigmoid(x)sigmoid(y)-sigmoid(x)-sigmoid(y)]
 */
double sigmoid_logistic_regression_protected(double x) {
    double c[N];

    // Generate N different evaluations
    for (int i = 0; i < N; ++i) {
        double y = simple_rand() * 10.0 - 5.0; // Random value between -5 and 5
        double z = x + y;

        // Calculate sigmoid values using original function
        double P_z = 1.0 / (1.0 + exp(-z));
        double P_y = 1.0 / (1.0 + exp(-y));

        // Apply the functional equation for sigmoid
        double numerator = P_z * (P_y - 1.0);
        double denominator = 2.0 * P_z * P_y - P_z - P_y;
        c[i] = numerator / denominator;
    }

    // Return median of all calculations
    return median(c, N);
}

/* Protected sigmoid activation function using N iterations
*
* Inputs:
*  x - Input value
*
* Returns: Protected sigmoid output */
void logistic_regression_protected(uint8_t layer_fault_no, double *X_predict_data, uint8_t* log_reg_class_pred) {
  const int n_features = 11;  // Number of features
  const int n_predict = 1;  // Number of samples to predict

  double *X_predict_flat = (double *)malloc(n_predict * n_features * sizeof(double));
  for (int i = 0; i < n_features; i++) {
    X_predict_flat[i] = X_predict_data[i];
  }

  // Make predictions using protected sigmoid
  if(layer_fault_no == 0x04){
    trigger_high();
  }

  for (int i = 0; i < n_predict; i++) {
    double linear_pred = 0;
    for (int j = 0; j < n_features; j++) {
      linear_pred += X_predict_flat[i * n_features + j] * log_reg_weights[j];
    }
    linear_pred += log_reg_bias;
    log_reg_class_pred[i] = sigmoid_logistic_regression_protected(linear_pred) > 0.5 ? 1 : 0;
  }
    
  if(layer_fault_no == 0x04){
      trigger_low();
  }
  free(X_predict_flat);
}


// All work below is for the Matrix Vector Protection Implementation

/* Protected dot product calculation
*
* Inputs:
*  weights - Weight vector
*  inputs - Input vector
*  size - Vector length
*
* Returns: Protected dot product */
static double linear_combination_protected(double *weights, double *inputs, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += weights[i] * inputs[i];
    }
    return result;
}

/* Protected prediction with linear combination and sigmoid
*
* Inputs:
*  weights - Weight vector
*  inputs - Input vector
*  size - Vector length
*  bias - Model bias
*
* Returns: Protected prediction value */
static double predict_protected(double *weights, double *inputs, int size, double bias) {
    double f = linear_combination_protected(weights, inputs, size);
    f += bias;
    return sigmoid_logistic_regression(f);
}

/* Random Space Redundancy for matrix operations
*
* Inputs:
*  sig_x_r - Protected result
*  sig_r - Random input result
*
* Returns: Corrected output */
double rsrprop(double sig_x_r, double sig_r) {
    return (sig_x_r * (sig_r - 1.0)) / (2.0 * sig_x_r * sig_r - sig_x_r - sig_r);
}


/* Median calculation for final result selection
*
* Inputs:
*  values - Array of computed values
*  size - Array length
*
* Returns: Median value */
double calculate_median_mvp(double *values, int size) {
    // Sort array using bubble sort
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (values[j] > values[j + 1]) {
                double temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }

    // Return median with proper handling of even/odd cases
    if (size % 2 == 0) {
        return (values[size / 2 - 1] + values[size / 2]) / 2.0;
    }
    return values[size / 2];
}

// Main protection function implementing MVP defense
/* Matrix-Vector Protection
*
* Inputs:
*  weights - Model weights
*  inputs - Input features
*  size - Input dimension
*  beta - Confidence parameter
*  bias - Model bias
*
* Returns: Protected prediction */
double correct_mvp(double *weights, double *inputs, int size, double beta, double bias) {
    // Calculate number of iterations based on desired confidence (beta)
    int N2 = (int)ceil(log(1.0 / beta));

    // Allocate memory for calculations
    double *corrected_values = (double *)malloc(N2 * sizeof(double));
    double *x_r = (double *)malloc(size * sizeof(double));
    double *r = (double *)malloc(size * sizeof(double));

    if (!corrected_values || !x_r || !r) {
        // Handle memory allocation failure
        if (corrected_values) free(corrected_values);
        if (x_r) free(x_r);
        if (r) free(r);
        return 0.0;
    }

    // Perform N iterations of protection algorithm
    for (int i = 0; i < N2; i++) {
        // Generate random perturbations
        for (int j = 0; j < size; j++) {
            r[j] = ((double)simple_rand_lr() / 0x7fffffff) * 2.0 - 1.0;
            x_r[j] = inputs[j] + r[j];
        }

        // Get predictions for both perturbed input and perturbation
        double pred_x_r = predict_protected(weights, x_r, size, bias);
        double pred_r = predict_protected(weights, r, size, 0);

        // Calculate corrected value using RSR
        corrected_values[i] = rsrprop(pred_x_r, pred_r);
    }

    // Get median of all corrected values
    double result = calculate_median_mvp(corrected_values, N2);

    // Clean up
    free(corrected_values);
    free(x_r);
    free(r);

    return result;
}

/* Main interface function for protected logistic regression for Matrix Vector Multiplication
* Inputs:
*  layer_fault_no - Fault injection identifier
*  X_predict_data - Input features
*  log_reg_class_pred - Output prediction
*/
void logistic_regression_protected_new(uint8_t layer_fault_no, double *X_predict_data, uint8_t* log_reg_class_pred) {
    const int n_features = 11;
    const int n_predict = 1;
    const double beta = 0.01;  // Confidence parameter
    uint8_t debug_msg;

    double *X_predict_flat = (double *)malloc(n_predict * n_features * sizeof(double));
    if (!X_predict_flat) {
        log_reg_class_pred[0] = 0;  // Safe default
        return;
    }

    // Copy input data
    for (int i = 0; i < n_features; i++) {
        X_predict_flat[i] = X_predict_data[i];
    }

    debug_msg = 1;

    debug_msg = 2;

    // Apply fault injection if specified
    if(layer_fault_no == 0x04) {
        trigger_high();
    }
    

    // Get MVP result with protection
    double mvp_result = correct_mvp(log_reg_weights, X_predict_flat, n_features, beta, log_reg_bias);

    if (mvp_result > 1000000 || mvp_result < -1000000) {
        debug_msg = 100;  // Extreme value
    } else if (mvp_result > 100 || mvp_result < -100) {
        debug_msg = 101;  // Large value
    } else {
        debug_msg = 102;  // Normal value
    }

    debug_msg = 3;

    // Apply sigmoid with bias
    // double final_result = sigmoid_logistic_regression(mvp_result + log_reg_bias);


    // Bound the result to [0,1]
    // if (final_result > 1.0) final_result = 1.0;
    // if (final_result < 0.0) final_result = 0.0;

    // Convert to binary prediction
    log_reg_class_pred[0] = (mvp_result > 0.5) ? 1 : 0;

    if(layer_fault_no == 0x04) {
        trigger_low();
    }



    debug_msg = 6;

    // Clean up
    free(X_predict_flat);
}
