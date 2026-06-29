#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

double sigmoid_logistic_regression(double x);

void matrix_multiply(double *a, double *b, double *result, int a_rows,
                     int a_cols, int b_cols);

void update_weights_bias(double *X, double *y, double *weights, double *bias,
                         int n_samples, int n_features, double lr);


void logistic_regression_fit(double *X, double *y, double *weights,
                             double *bias, int n_samples, int n_features,
                             int n_iters, double lr);

void logistic_regression_predict(double *X, double *weights, double bias,
                                 uint8_t *log_reg_class_pred, int n_samples,
                                 int n_features);

void logistic_regression (uint8_t layer_fault_no, double X_predict_data[2], uint8_t* log_reg_class_pred);


//Undergraduate Work by Dustin for protected sigmoid
double simple_rand_lr(void);
double sigmoid_logistic_regression_protected(double x);
double median_lr(double *arr, int n);
void logistic_regression_protected(uint8_t layer_fault_no, double X_predict_data[2], uint8_t* log_reg_class_pred);

//Undergraduate Work by Dustin for protected matvecmult
double rsrprop(double sig_x_r, double sig_r);
double calculate_median_mvp(double *values, int size);
double correct_mvp(double *weights, double *inputs, int size, double beta, double bias);
void logistic_regression_protected_new(uint8_t layer_fault_no, double *X_predict_data, uint8_t* log_reg_class_pred);
