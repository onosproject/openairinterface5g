#ifndef CUDA_STRUCT
#define CUDA_STRUCT

#include <cuda.h>
#include <cuda_runtime.h>

typedef float2 Complex;

typedef enum {
	  CYCLIC_PREFIX,
	  CYCLIC_SUFFIX,
	  ZEROS,
	  NONE
} Extension_t;

#endif
