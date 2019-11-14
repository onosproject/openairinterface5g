#ifndef INIT_CUDA_DEF
#define INIT_CUDA_DEF

#include "cuda_struct.h"

#if __cplusplus
extern "C" {
#endif


void init_cuda(int nb_tx, int nb_symbols, int fftsize);


#if __cplusplus
}
#endif

#endif
