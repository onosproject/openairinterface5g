#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "CUDA/checkError.h"
#include "CUDA/struct.h"


__global__ void cu_intToComplex(int *input, Complex *signal){
	int id = blockIdx.x*1024 + threadIdx.x;
	signal[id].x = ((short*)&input[id])[0];
	signal[id].y = ((short*)&input[id])[1];
}

__global__ void cu_ComplexToInt(int *output, Complex *signal){
	int id = blockIdx.x*1024 + threadIdx.x;
	((short*)&output[id])[0] = round(signal[id].x);
	((short*)&output[id])[1] = round(signal[id].y);
}

extern "C" void CUDA_PHY_ofdm_mod(int *input, 
				int *output, 
				int fftsize, 
				unsigned char nb_symbols, 
				unsigned short nb_prefix_samples, 
				Extension_t etype){
	if(nb_symbols==0) return ;


	Complex *d_signal;
	gpuErrchk( cudaMalloc((void**)&d_signal, fftsize*sizeof(Complex)*nb_symbols) );

	int *d_data;
	gpuErrchk( cudaMalloc((void**)&d_data, fftsize*sizeof(int)*nb_symbols) );
	gpuErrchk( cudaMemcpy(d_data, input, fftsize*sizeof(int)*nb_symbols, cudaMemcpyHostToDevice) );

	int threadNum = 1024;
	int blockNum = fftsize*nb_symbols/threadNum;
	cu_intToComplex<<<blockNum, threadNum>>>(d_data, d_signal);
	

	cufftHandle plan;
	cufftErrchk( cufftPlan1d(&plan, fftsize, CUFFT_C2C, nb_symbols) );
	cufftErrchk( cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD) );

	Complex *h_output = (Complex*)malloc(fftsize*sizeof(Complex)*nb_symbols) ;
	gpuErrchk( cudaMemcpy(h_output, d_signal, fftsize*sizeof(Complex)*nb_symbols, cudaMemcpyDeviceToHost) );

	cu_ComplexToInt<<<blockNum, threadNum>>>(d_data, d_signal);

	int *res = (int*)malloc(fftsize*sizeof(int)*nb_symbols);
	gpuErrchk( cudaMemcpy(res, d_data, fftsize*sizeof(int)*nb_symbols, cudaMemcpyDeviceToHost) );

/*	
	for(int i=0; i<fftsize*nb_symbols; i++){
		printf("res(%d) %d+%di\n", i, ((short*)&res[i])[0], ((short*)&res[i])[1]);
	}*/

	for(int symb_th=0; symb_th<nb_symbols; symb_th++){
		int* output_ptr;
		switch(etype){
			case CYCLIC_PREFIX:{
				output_ptr = &output[symb_th*fftsize + (1+symb_th)*nb_prefix_samples];
				memcpy(output_ptr, res, fftsize<<2);

				int j=fftsize;
				for(int k=-1; k>=-nb_prefix_samples; k--){
					output_ptr[k] = output_ptr[--j];
				}
				break;
			}
			case CYCLIC_SUFFIX:{
				output_ptr = &output[symb_th*fftsize + (symb_th)*nb_prefix_samples];
				memcpy(output_ptr, res, fftsize<<2);

				for(int k=0; k<nb_prefix_samples; k++){
					output_ptr[fftsize+k] = output_ptr[k];
				}
				break;
			}
			case ZEROS:{
				break;
			}
			case NONE:{
				output_ptr = &output[fftsize];
				memcpy(output_ptr, res, fftsize<<2);
				break;
			}

			default:{
				break;
			}

		}

		cufftDestroy(plan);
		free(h_output);
		free(res);
		cudaFree(d_signal);
	}
}

__global__ void conjMul(int *d_x1, int *d_x2, int *d_y, short zero_flag,unsigned int div){
	int id = blockIdx.x*1024 + threadIdx.x;
	int *x1 = &d_x1[id];
	int *x2 = &d_x2[id];
	int *y = &d_y[id];

	int re, im;

	((short*)x1)[1] *= -1;
	re = ((short*)x1)[0]*((short*)x2)[0] - ((short*)x1)[1]*((short*)x2)[1];
	im = ((short*)x1)[1]*((short*)x2)[0] + ((short*)x1)[0]*((short*)x2)[1];

	re = re / div;
	im = im / div;

	if(zero_flag){
		((short*)y)[0] = re; 
		((short*)y)[1] = im;
	}else{
		((short*)y)[0] += re; 
		((short*)y)[1] += im;	
	}

}

extern "C" void CUDA_multadd_cpx_vector(int *x1, int *x2, int *y, short zero_flag,unsigned int N, int output_shift){
	int *d_x1, *d_x2, *d_y;
 	gpuErrchk( cudaMalloc((void**)&d_x1, N*sizeof(int)) );
 	gpuErrchk( cudaMalloc((void**)&d_x2, N*sizeof(int)) );
 	gpuErrchk( cudaMalloc((void**)&d_y, N*sizeof(int)) );

	gpuErrchk( cudaMemcpy(d_x1, x1, N*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_x2, x2, N*sizeof(int), cudaMemcpyHostToDevice) );

	if(zero_flag == 0){
		gpuErrchk( cudaMemcpy(d_y, y, N*sizeof(int), cudaMemcpyHostToDevice) );	
	}else{
		gpuErrchk( cudaMemset(d_y, 0, N*sizeof(int)) );
	}

	unsigned int div = 1;
	div = div << output_shift;

	int threadNum = 1024;
	int blockNum = N / threadNum;	
	conjMul<<<blockNum, threadNum>>>(d_x1, d_x2, d_y, zero_flag, div);
	cudaDeviceSynchronize();
	CHECK_STATE("conjMul");

	gpuErrchk( cudaMemcpy(y, d_y, N*sizeof(int), cudaMemcpyDeviceToHost) );


	cudaFree(d_x1);
	cudaFree(d_x2);
	cudaFree(d_y);
}
