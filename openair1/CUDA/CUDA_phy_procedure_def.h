#ifndef CUDA
#define CUDA

#include "cuda_struct.h"

#if __cplusplus
extern "C" {
#endif

void CUDA_hello(void);
void CUDA_ifft_ofdm( int **output, 
				int fftsize, 
				unsigned char nb_symbols, 
				unsigned char nb_prefix_samples,
				unsigned char nb_prefix_samples0,
				int nb_tx,
				int Ncp,
				Extension_t etype);
void CUDA_beam_precoding(int **txdataF, int ***weight, int L_ssb, int shift, int fftsize, int nb_symbols, int nb_antenna_ports, int nb_tx);

#if __cplusplus
}
#endif

#endif
