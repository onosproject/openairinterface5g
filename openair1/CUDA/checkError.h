__global__ void gpu_hello(void){
	printf("Hello world from GPU!\n");
}

extern "C" void CUDA_hello(void){
	printf("ready to gpu_hello\n");
	gpu_hello<<<1,1>>>();
	cudaDeviceSynchronize();
}

static const char* _cudaGetErrorEnum(cufftResult error){
	switch (error){
		case CUFFT_SUCCESS:{return "CUFFT_SUCCESS";}
		case CUFFT_INVALID_PLAN:{return "CUFFT_INVALID_PLAN";}
		case CUFFT_ALLOC_FAILED:{return "CUFFT_ALLOC_FAILED";}
		case CUFFT_INVALID_TYPE:{return "CUFFT_INVALID_TYPE";}
		case CUFFT_INVALID_VALUE:{return "CUFFT_INVALID_VALUE";}
		case CUFFT_INTERNAL_ERROR:{return "CUFFT_INTERNAL_ERROR";}
		case CUFFT_EXEC_FAILED:{return "CUFFT_EXEC_FAILED";}
		case CUFFT_SETUP_FAILED:{return "CUFFT_SETUP_FAILED";}
		case CUFFT_INVALID_SIZE:{return "CUFFT_INVALID_SIZE";}
		case CUFFT_UNALIGNED_DATA:{return "CUFFT_UNALIGNED_DATA";}
	}

	return "<unknown>";
}

#define cufftErrchk(ans) { __cufftSafeCall((ans), __FILE__, __LINE__); }
inline void __cufftSafeCall(cufftResult err, const char *file, const int line, bool abort=true){
	if (err != CUFFT_SUCCESS ){
		fprintf(stderr,"CUFFT assert: %s %s %d\n", _cudaGetErrorEnum(err), file, line);
		if (abort) exit(err);
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define CHECK_STATE(msg) {checkCudaState((msg), __FILE__, __LINE__);}
inline void checkCudaState(const char *msg, const char *file, const int line){
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) {
		fprintf(stderr, "[%s]gpu error: %s %s %d\n", msg, cudaGetErrorString(err), file, line);
	}
}








