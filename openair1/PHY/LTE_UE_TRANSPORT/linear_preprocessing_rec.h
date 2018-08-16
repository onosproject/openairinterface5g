
#include<stdio.h>
#include<math.h>
#include<complex.h>
#include <stdlib.h>
#include "PHY/defs_UE.h"


/* FUNCTIONS FOR LINEAR PREPROCESSING: MMSE, WHITENNING, etc*/
void transpose(int N, float complex *A, float complex *Result);

void conjugate_transpose (int rows_A, int col_A, float complex *A, float complex *Result);

void H_hermH_plus_sigma2I (int row_A, int col_A, float complex *A, float sigma2, float complex *Result);

void HH_herm_plus_sigma2I (int rows_A, int col_A, float complex *A, float sigma2, float complex *Result);

void eigen_vectors_values(int N, float complex *A, float complex *Vectors, float *Values_Matrix);

void lin_eq_solver(int N, float complex *A, float complex *B);

/* mutl_matrix_matrix_row_based performs multiplications when matrix is row-oriented H[0], H[1]; H[2], H[3]*/
void mutl_matrix_matrix_row_based(float complex *M0, float complex *M1, int rows_M0, int col_M0, int rows_M1, int col_M1, float complex *Result );

/* mutl_matrix_matrix_col_based performs multiplications matrix is column-oriented H[0], H[2]; H[1], H[3]*/
void mutl_matrix_matrix_col_based(float complex *M0, float complex *M1, int rows_M0, int col_M0, int rows_M1, int col_M1, float complex *Result );

void mutl_scal_matrix_matrix_col_based(float complex *M0, float complex *M1, float complex alpha, int rows_M0, int col_M0, int rows_M1, int col_M1, float complex *Result);

void compute_MMSE(float complex *H, int order_H, float sigma2, float complex *W_MMSE);

float sqrt_float(float x);

void compute_white_filter(float complex *H0_re,
                          float complex *H1_re,
                          float sigma2,
                          int n_rx,
                          int n_tx,
                          float complex *W_Wh_0_re,
                          float complex *W_Wh_1_re);

void mmse_processing_oai(LTE_UE_PDSCH *pdsch_vars,
                         LTE_DL_FRAME_PARMS *frame_parms,
                         PHY_MEASUREMENTS *measurements,
                         unsigned char first_symbol_flag,
                         MIMO_mode_t mimo_mode,
                         unsigned short mmse_flag,
                         int noise_power,
                         unsigned char symbol,
                         unsigned short nb_rb);


void precode_channel_est(int32_t **dl_ch_estimates_ext,
                        LTE_DL_FRAME_PARMS *frame_parms,
                        LTE_UE_PDSCH *pdsch_vars,
                        unsigned char symbol,
                        unsigned short nb_rb,
                        MIMO_mode_t mimo_mode);


void rxdataF_to_float(int32_t **rxdataF_ext,
                      float complex **rxdataF_f,
                      int n_rx,
                      int length,
                      int start_point);

void chan_est_to_float(int32_t **dl_ch_estimates_ext,
                       float complex **dl_ch_estimates_ext_f,
                       uint8_t n_tx,
                       uint8_t n_rx,
                       int32_t length,
                       int32_t start_point);

void float_to_chan_est(float complex **chan_est_flp,
                       int32_t **result,
                       int n_tx,
                       int n_rx,
                       int length,
                       int start_point);

void float_to_rxdataF(float complex **rxdataF_flp,
                      int32_t **result,
                      uint8_t n_tx,
                      uint8_t n_rx,
                      int32_t length,
                      int32_t start_point);

void mult_filter_chan_est(float complex **W,
                          float complex **chan_est_flp,
                          float complex **result,
                          uint8_t n_tx,
                          uint8_t n_rx,
                          int32_t n_col_chan_est_flp,
                          int32_t length,
                          int32_t start_point);

void mult_filter_rxdataF(float complex **W,
                         float complex **rxdataF_flp,
                         float complex **result,
                         uint8_t n_tx,
                         uint8_t n_rx,
                         int32_t length,
                         int32_t start_point);

void mmse_processing_core(int32_t **rxdataF_ext,
                          int32_t **dl_ch_estimates_ext,
                          int sigma2,
                          int n_tx,
                          int n_rx,
                          int length,
                          int start_point);

void mmse_processing_core_flp(float complex **rxdataF_flp,
                              float complex **chan_est_flp,
                              int32_t **rxdataF_filt_fp,
                              int32_t **chan_est_eff_fp,
                              float noise_power,
                              uint8_t n_tx,
                              uint8_t n_rx,
                              int32_t length,
                              int32_t start_point);

void whitening_processing_core_flp(float complex **rxdataF_flp,
                                   float complex **chan_est_flp_0,
                                   float complex **chan_est_flp_1,
                                   int32_t **rxdataF_filt_fp_0,
                                   int32_t **rxdataF_filt_fp_1,
                                   int32_t **chan_est_eff_fp_0,
                                   int32_t **chan_est_eff_fp_1,
                                   float sigma2,
                                   uint8_t n_tx,
                                   uint8_t n_rx,
                                   int32_t length,
                                   int32_t start_point);


