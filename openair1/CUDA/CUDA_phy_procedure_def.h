#ifndef CUDA
#define CUDA


#if __cplusplus
extern "C" {
#endif

void CUDA_hello(void);
void CUDA_PHY_ofdm_mod(int *input, 
				int *output, 
				int fftsize, 
				unsigned char nb_symbols, 
				unsigned short nb_prefix_samples, 
				Extension_t etype);
void CUDA_multadd_cpx_vector(int* x1, int *x2, int *y, short zero_flag, unsigned int N, int output_shift);

#if __cplusplus
}
#endif

#endif
