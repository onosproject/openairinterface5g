#ifndef CUDA_STRUCT_H
#define CUDA_STRUCT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#if __cplusplus
extern "C" {
#endif

typedef float2 Complex;

typedef struct cuda_cu_ru_t{
	//beamforming precoding
	int **d_txdataF;//14symb-port0, 14symb-port1, ......
	int ***d_weight;//[p][tx][symb]
	cudaStream_t *d_beam_stream;

	//ifft
	int *d_txdataF_BF;//14symb-tx0, 14symb-tx1, ......
	Complex *d_signal;
	int *d_data_wCP;
	cufftHandle plan;
}cuda_cu_ru;
extern cuda_cu_ru cu_ru;


#if __cplusplus
}
#endif

#endif
