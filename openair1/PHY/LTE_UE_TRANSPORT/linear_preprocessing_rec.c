/* These functions compute linear preprocessing for
the UE using LAPACKE and CBLAS modules of
LAPACK libraries.
MMSE and MMSE whitening filters are available.
Functions are using RowMajor storage of the
matrices, like in conventional C. Traditional
Fortran functions of LAPACK employ ColumnMajor
data storage. */

#include<stdio.h>
#include<math.h>
#include<complex.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>
#include <lapacke_utils.h>
#include <lapacke.h>

//#define DEBUG_PREPROC


void transpose (int N, float complex *A, float complex *Result)
{
  // COnputes C := alpha*op(A)*op(B) + beta*C,
  enum CBLAS_TRANSPOSE transa = CblasTrans;
  enum CBLAS_TRANSPOSE transb = CblasNoTrans;
  int rows_opA = N; // number of rows in op(A) and in C
  int col_opB = N; //number of columns of op(B) and in C
  int col_opA = N; //number of columns in op(A) and rows in op(B)
  int col_B; //number of columns in B
  float complex alpha = 1.0+I*0;
  int lda  = rows_opA;
  float complex beta = 0.0+I*0;
  int ldc = rows_opA;
  int i;
  float complex* B;

  int ldb = col_opB;

  if (transb == CblasNoTrans) {
    B = (float complex*)calloc(ldb*col_opB,sizeof(float complex));
    col_B= col_opB;
  }
  else {
    B = (float complex*)calloc(ldb*col_opA, sizeof(float complex));
    col_B = col_opA;
  }
  float complex* C = (float complex*)malloc(ldc*col_opB*sizeof(float complex));

  for (i=0; i<lda*col_B; i+=N+1)
    B[i]=1.0+I*0;

  cblas_cgemm(CblasRowMajor, transa, transb, rows_opA, col_opB, col_opA, &alpha, A, lda, B, ldb, &beta, C, ldc);

  memcpy(Result, C, N*N*sizeof(float complex));

  free(B);
  free(C);
 }



void conjugate_transpose (int rows_A, int col_A, float complex *A, float complex *Result)
{
  // Computes C := alpha*op(A)*op(B) + beta*C,
  enum CBLAS_TRANSPOSE transa = CblasConjTrans;
  enum CBLAS_TRANSPOSE transb = CblasNoTrans;
  float complex alpha = 1.0;
  float complex beta = 0.0;
  int i;
  float complex* B;

  B = (float complex*)calloc(rows_A*rows_A,sizeof(float complex));


  for (i=0; i<rows_A*rows_A; i+=rows_A+1)
    B[i]=1.0;

  cblas_cgemm(CblasColMajor, transa, transb, col_A, rows_A, rows_A, &alpha, A, rows_A, B, rows_A, &beta, Result, col_A);

  free(B);
 }

void H_hermH_plus_sigma2I (int N, int M, float complex *A, float sigma2, float complex *Result)
{
  //C := alpha*op(A)*op(B) + beta*C,
  enum CBLAS_TRANSPOSE transa = CblasConjTrans;
  enum CBLAS_TRANSPOSE transb = CblasNoTrans;
  int rows_opA = N; // number of rows in op(A) and in C
  int col_opB = N; //number of columns of op(B) and in C
  int col_opA = N; //number of columns in op(A) and rows in op(B)
  int col_C = N; //number of columns in B
  float complex alpha = 1.0+I*0;
  int lda  = col_opA;
  float complex beta = 1.0 + I*0;
  int ldc = col_opA;
  int i;

  float complex* C = (float complex*)calloc(ldc*col_opB, sizeof(float complex));

  for (i=0; i<lda*col_C; i+=N+1)
    C[i]=sigma2*(1.0+I*0);

  cblas_cgemm(CblasRowMajor, transa, transb, rows_opA, col_opB, col_opA, &alpha, A, lda, A, lda, &beta, C, ldc);

  memcpy(Result, C, N*M*sizeof(float complex));
  free(C);
 }


 void HH_herm_plus_sigma2I (int rows_A, int col_A, float complex *A, float sigma2, float complex *Result)
{

  //C := alpha*op(A)*op(B) + beta*C,
  enum CBLAS_TRANSPOSE transa = CblasNoTrans;
  enum CBLAS_TRANSPOSE transb = CblasConjTrans;
  float complex alpha = 1.0+I*0;
  int i;

  for (i = 0; i < rows_A*rows_A; i += rows_A+1)
    Result[i]=1.0+I*0;


  cblas_cgemm(CblasColMajor, transa, transb, rows_A, rows_A, col_A, &alpha, A, rows_A, A, rows_A, &sigma2, Result, rows_A);

}

void eigen_vectors_values (int N, float complex *A, float complex *Vectors, float *Values_Matrix)
{
  // This function computes ORTHONORMAL eigenvectors and eigenvalues of matrix A,
  // where Values_Matrix is a diagonal matrix of eigenvalues.
  // A=Vectors*Values_Matrix*Vectors'
  char jobz = 'V'; // compute both eigenvectors and eigenvalues
  char uplo = 'U';
  int order_A = N;
  int lda = N;
  int i;
  float* Values = (float*)calloc(1*N, sizeof(float));

  LAPACKE_cheev(LAPACK_COL_MAJOR, jobz, uplo, order_A, A, lda, Values);

  memcpy(Vectors, A, N*N*sizeof(float complex));

  for (i=0; i<lda; i+=1)
    Values_Matrix[i*(lda+1)]=Values[i];

  free(Values);
}

 void lin_eq_solver (int N, float complex* A, float complex* B, float complex* Result)
{
  int n = N;
  int lda = N;
  int ldb = N;
  int nrhs = N;

  char transa = 'N';
  int* IPIV = calloc(N*N, sizeof(int));

  // Compute LU-factorization
  LAPACKE_cgetrf(LAPACK_ROW_MAJOR, n, nrhs, A, lda, IPIV);

  // Solve AX=B
  LAPACKE_cgetrs(LAPACK_ROW_MAJOR, transa, n, nrhs, A, lda, IPIV, B, ldb);

  // cgetrs( "N", N, 4, A, lda, IPIV, B, ldb, INFO )

  memcpy(Result, B, N*N*sizeof(float complex));

  free(IPIV);

}

void mutl_matrix_matrix_row_based(float complex* M0, float complex* M1, int rows_M0, int col_M0, int rows_M1, int col_M1, float complex* Result ){
  enum CBLAS_TRANSPOSE transa = CblasNoTrans;
  enum CBLAS_TRANSPOSE transb = CblasNoTrans;
  int rows_opA = rows_M0; // number of rows in op(A) and in C
  int col_opB = col_M1; //number of columns of op(B) and in C
  int col_opA = col_M0; //number of columns in op(A) and rows in op(B)
  float complex alpha =1.0;
  int lda  = col_M0;
  float complex beta = 0.0;
  int ldc = col_M1;
  int ldb = col_M1;

#ifdef DEBUG_PREPROC
  int i=0;
  printf("mutl_matrix_matrix_row_based: rows_M0 %d, col_M0 %d, rows_M1 %d, col_M1 %d\n", rows_M0, col_M0, rows_M1, col_M1);

  for(i=0; i<rows_M0*col_M0; ++i)
    printf("mutl_matrix_matrix_row_based: rows_opA = %d, col_opB = %d, W_MMSE[%d] = (%f + i%f)\n", rows_opA, col_opB,  i , creal(M0[i]), cimag(M0[i]));

  for(i=0; i<rows_M1*col_M1; ++i)
    printf("mutl_matrix_matrix_row_based: M1[%d] = (%f + i%f)\n", i , creal(M1[i]), cimag(M1[i]));
#endif

  cblas_cgemm(CblasRowMajor, transa, transb, rows_opA, col_opB, col_opA, &alpha, M0, lda, M1, ldb, &beta, Result, ldc);

#ifdef DEBUG_PREPROC
  for(i=0; i<rows_opA*col_opB; ++i)
    printf("mutl_matrix_matrix_row_based: result[%d] = (%f + i%f)\n", i , creal(Result[i]), cimag(Result[i]));
#endif

}

void mutl_matrix_matrix_col_based(float complex* M0, float complex* M1, int rows_M0, int col_M0, int rows_M1, int col_M1, float complex* Result ){
  enum CBLAS_TRANSPOSE transa = CblasNoTrans;
  enum CBLAS_TRANSPOSE transb = CblasNoTrans;
  int rows_opA = rows_M0; // number of rows in op(A) and in C
  int col_opB = col_M1; //number of columns of op(B) and in C
  int col_opA = col_M0; //number of columns in op(A) and rows in op(B)
  float complex alpha =1.0;
  int lda  = col_M0;
  float complex beta = 0.0;
  int ldc = rows_M1;
  int ldb = rows_M1;

#ifdef DEBUG_PREPROC
  int i = 0;
  printf("mutl_matrix_matrix_col_based: rows_M0 %d, col_M0 %d, rows_M1 %d, col_M1 %d\n", rows_M0, col_M0, rows_M1, col_M1);

  for(i = 0; i < rows_M0*col_M0; ++i)
    printf("mutl_matrix_matrix_col_based: rows_opA = %d, col_opB = %d, filter[%d] = (%f + i%f)\n", rows_opA, col_opB,  i , creal(M0[i]), cimag(M0[i]));

  for(i = 0; i < rows_M1*col_M1; ++i)
    printf("mutl_matrix_matrix_col_based: M1[%d] = (%f + i%f)\n", i , creal(M1[i]), cimag(M1[i]));
#endif

  cblas_cgemm(CblasColMajor, transa, transb, rows_opA, col_opB, col_opA, &alpha, M0, lda, M1, ldb, &beta, Result, ldc);

#ifdef DEBUG_PREPROC
  for(i = 0; i < rows_opA*col_opB; ++i)
    printf("mutl_matrix_matrix_col_based: result[%d] = (%f + i%f)\n", i , creal(Result[i]), cimag(Result[i]));
#endif
}

void mutl_scal_matrix_matrix_col_based(float *M0, float complex *M1, float alpha, int rows_M0, int col_M0, int rows_M1, int col_M1, float complex *Result ){

  enum CBLAS_TRANSPOSE transa = CblasNoTrans;
  enum CBLAS_TRANSPOSE transb = CblasNoTrans;
  float complex beta = 0.0;
  int i;

  // Convert float M0 into complex float D_0_complex required by cblas_cgemm
  float complex *D_0_complex = calloc(rows_M0*col_M0, sizeof(float complex));
  for(i = 0; i < rows_M0*col_M0; ++i)
  {
    D_0_complex[i] = M0[i] + I*0.00001;
#ifdef DEBUG_PREPROC
    printf("mutl_scal_matrix_matrix_col_based: D_0_complex[%d] = (%f, %f)\n", i , creal(D_0_complex[i]), cimag(D_0_complex[i]));
#endif
  }

#ifdef DEBUG_PREPROC
  printf("mutl_scal_matrix_matrix_col_based: alpha = %f\n", alpha);

  for(i = 0; i < rows_M0*col_M0; ++i){
    printf("mutl_scal_matrix_matrix_col_based M0[%d] = %f\n", i , M0[i]);
  }

  for(i = 0; i < rows_M1*col_M1; ++i)
    printf("mutl_scal_matrix_matrix_col_based: M1[%d] = (%f + i%f)\n", i , creal(M1[i]), cimag(M1[i]));
#endif

  cblas_cgemm(CblasColMajor, transa, transb, rows_M0, col_M1, col_M0, &alpha, D_0_complex, rows_M0, M1, col_M0, &beta, Result, rows_M0);

#ifdef DEBUG_PREPROC
  for(i = 0; i < rows_M0*col_M1; ++i)
    printf("mutl_scal_matrix_matrix_col_based: result[%d] = (%f + i%f)\n", i , creal(Result[i]), cimag(Result[i]));
#endif

  free(D_0_complex);
}


/*FILTERS */
void compute_MMSE(float complex* H, int order_H, float sigma2, float complex* W_MMSE)
{
  int N = order_H;
  float complex* H_hermH_sigmaI = calloc(N*N, sizeof(float complex));
  float complex* H_herm =  calloc(N*N, sizeof(float complex));

  H_hermH_plus_sigma2I(N, N, H, sigma2, H_hermH_sigmaI);

#ifdef DEBUG_PREPROC
  int i = 0;
  for(i = 0;i < N*N; ++i)
    printf("compute_MMSE: H_hermH_sigmaI[%d] = (%f + i%f)\n", i , creal(H_hermH_sigmaI[i]), cimag(H_hermH_sigmaI[i]));
#endif

  conjugate_transpose (N, N, H, H_herm); //equals H_herm

#ifdef DEBUG_PREPROC
  for(i = 0;i < N*N;i++)
    printf("compute_MMSE: H_herm[%d] = (%f + i%f)\n", i , creal(H_herm[i]), cimag(H_herm[i]));
#endif

  lin_eq_solver(N, H_hermH_sigmaI, H_herm, W_MMSE);

#ifdef DEBUG_PREPROC
  for(i = 0;i < N*N; ++i)
    printf("compute_MMSE: W_MMSE[%d] = (%f + i%f)\n", i , creal(W_MMSE[i]), cimag(W_MMSE[i]));
#endif

  free(H_hermH_sigmaI);
  free(H_herm);
}

float sqrt_float(float x)
{
  float sqrt_x = 0.0;
  sqrt_x = (float)(sqrt((double)(x)));
  return sqrt_x;
}

void compute_white_filter(float complex* H0_re,
                          float complex* H1_re,
                          float sigma2,
                          int n_rx,
                          int n_tx,
                          float complex* W_Wh_0_re,
                          float complex* W_Wh_1_re){

  float sigma = 0.0;
  int i;

  float complex *R_corr_col_n_0_re = calloc(n_rx*n_tx, sizeof(float complex));
  float complex *R_corr_col_n_1_re = calloc(n_rx*n_tx, sizeof(float complex));
  float complex *U_0_re = calloc(n_rx*n_tx, sizeof(float complex));
  float complex *U_1_re = calloc(n_rx*n_tx, sizeof(float complex));
  float complex *U_0_herm_re = calloc(n_rx*n_tx, sizeof(float complex));
  float complex *U_1_herm_re = calloc(n_rx*n_tx, sizeof(float complex));
  float *D_0_re = calloc(n_rx*n_tx, sizeof(float));
  float *D_1_re = calloc(n_rx*n_tx, sizeof(float));
  float *D_0_re_inv_sqrt = calloc(n_rx*n_tx, sizeof(float));
  float *D_1_re_inv_sqrt = calloc(n_rx*n_tx, sizeof(float));

  // Whitening filter can be computed using the following algorithm:
  // 1. Compute covariance of the colored noise: R = HH' + sigma2I.
  // 2. Compute eigen value decomposition of R = UDU'.
  // 3. W_wh = sigma sqrt(inv(D))U'.
  // 4. This function computes W_wh for both branches.

#ifdef DEBUG_PREPROC
  printf("compute_white_filter: sigma2 = %f\n", sigma2);
  for(i=0; i<n_rx*n_tx/2; i++){
    printf("compute_white_filter: H1_re[%d] = (%f + i%f)\n", i , creal(H1_re[i]), cimag(H1_re[i]));
    printf("compute_white_filter: H0_re[%d] = (%f + i%f)\n", i , creal(H0_re[i]), cimag(H0_re[i]));
}
#endif


  // 1. Compute covariance of the colored noise: R = HH' + sigma2I.
  HH_herm_plus_sigma2I(n_rx, n_tx/2, H1_re, sigma2, R_corr_col_n_0_re);
  HH_herm_plus_sigma2I(n_rx, n_tx/2, H0_re, sigma2, R_corr_col_n_1_re);

#ifdef DEBUG_PREPROC
  for(i=0;i<n_rx*n_tx;i++){
    printf("compute_white_filter: R_corr_col_n_0_re[%d] = (%f + i%f)\n", i , creal(R_corr_col_n_0_re[i]), cimag(R_corr_col_n_0_re[i]));
    printf("compute_white_filter: R_corr_col_n_1_re[%d] = (%f + i%f)\n", i , creal(R_corr_col_n_1_re[i]), cimag(R_corr_col_n_1_re[i]));
  }

#endif
  // 2. Compute eigen value decomposition of R = UDU'.
  eigen_vectors_values(n_rx, R_corr_col_n_0_re, U_0_re, D_0_re);
  eigen_vectors_values(n_rx, R_corr_col_n_1_re, U_1_re, D_1_re);

#ifdef DEBUG_PREPROC
  for(i=0;i<n_rx*n_tx;i++){
    printf("compute_white_filter: U_0_re[%d] = (%f + i%f)\n", i , creal(U_0_re[i]), cimag(U_0_re[i]));
    printf("compute_white_filter: D_0_re[%d] = (%f + i%f)\n", i , creal(D_0_re[i]), cimag(D_0_re[i]));
}
#endif

  // 3. Compute eigen value decomposition of R = UDU'.

  conjugate_transpose(n_rx, n_tx, U_0_re, U_0_herm_re);
  conjugate_transpose(n_rx, n_tx, U_1_re, U_1_herm_re);


#ifdef DEBUG_PREPROC
  for(i = 0;i < n_rx*n_tx; i++){
    printf("compute_white_filter: U_0_herm_re[%d] = (%f + i%f)\n", i , creal(U_0_herm_re[i]), cimag(U_0_herm_re[i]));
  }
#endif


  sigma = (float)(sqrt((double)(sigma2)));
  if (sigma <= 0.0001){
      sigma = 0.0001;
    }

  //The inverse of a diagonal matrix is obtained by replacing each element in the diagonal with //its reciprocal. A square root of a diagonal matrix is given by the diagonal matrix, whose //diagonal entries are just the square roots of the original matrix. However, if SNR is high,
  //the diagonal elements of D are very small, and inverse is not always possible. We thus appy a threshold to avoid too low values.

  for (i = 0; i < n_rx*n_tx; i += (n_rx + 1)){

    if (D_0_re[i] <= 0.0001){
      D_0_re[i] = 0.0001;
    }
    if (D_1_re[i] <= 0.0001){
      D_1_re[i] = 0.0001;
    }

    D_0_re_inv_sqrt[i] = sqrt_float(1/D_0_re[i]);
    D_1_re_inv_sqrt[i] = sqrt_float(1/D_1_re[i]);
  }


#ifdef DEBUG_PREPROC
  for(i = 0;i <n_rx*n_tx; i++){
    printf("compute_white_filter: D_0_re_inv_sqrt[%d] = %f\n", i , D_0_re_inv_sqrt[i]);
  }
#endif


  mutl_scal_matrix_matrix_col_based(D_0_re_inv_sqrt, U_0_herm_re, sigma, n_rx, n_tx, n_rx, n_tx, W_Wh_0_re);

#ifdef DEBUG_PREPROC
  for(i = 0;i < n_rx*n_tx; i++){
    printf("compute_white_filter: W_Wh_0_re[%d] = (%f + i%f)\n", i , creal(W_Wh_0_re[i]), cimag(W_Wh_0_re[i]));
  }
#endif

  mutl_scal_matrix_matrix_col_based(D_1_re_inv_sqrt, U_1_herm_re, sigma, n_rx, n_tx, n_rx, n_tx, W_Wh_1_re);

  free(R_corr_col_n_0_re);
  free(R_corr_col_n_1_re);
  free(U_0_herm_re);
  free(U_1_herm_re);
  free(D_0_re_inv_sqrt);
  free(D_1_re_inv_sqrt);
  free(U_0_re);
  free(U_1_re);
  free(D_0_re);
  free(D_1_re);
}
