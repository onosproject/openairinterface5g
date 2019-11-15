#ifndef INIT_CUDA_DEF
#define INIT_CUDA_DEF

#include "struct.h"

typedef cuda_ifft_t{
	Complex *d_signal;
	Complex *d_output;
	int *d_data;	
}cuda_ifft



#endif
