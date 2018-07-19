/*! \file PHY\CUDA/LTE_TRANSPORT/turbo_rx_gpu.cu
* \brief turbo decoder using gpu 
* \author TerngYin Hsu, JianYa Chu
* \date 2018
* \version 0.1
* \company ISIP LAB/NCTU CS  
* \email: tyhsu@cs.nctu.edu.tw
* \note
* \warning
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>


#include "turbo_parm.h"
#include "PHY/CODING/extern_3GPPinterleaver.h"
//#include "extern_interleaver.h"
#include "PHY/CODING/defs.h"
#include "turbo_rx_gpu.h"
//#include "crc_byte.h"
//#include "turbo_rx.h"
//#include "PHY/CODING/extern_table_gpu.h"
//#include "extern_table_gpu.h"
#include "PHY/defs.h"
//typedef int16_t llr_t;

#define CRC24_A 0
#define CRC24_B 1
#define CRC16 2
#define CRC8 3

int intable_h[188][6144];
int detable_h[188][6144];

void free_ptr()
{
    cudaFree(turbo_parm->sys_d);
	cudaFree(turbo_parm->sys1_d);
	cudaFree(turbo_parm->sys2_d);
	cudaFree(turbo_parm->ypar1_d);
	cudaFree(turbo_parm->ypar2_d);
	cudaFree(turbo_parm->alpha_d);
	cudaFree(turbo_parm->alpha_pre_1);
	cudaFree(turbo_parm->alpha_pre_2);
	cudaFree(turbo_parm->beta_pre_1);
	cudaFree(turbo_parm->beta_pre_2);
	cudaFree(turbo_parm->ext_d);
	cudaFree(turbo_parm->ext2_d);
	cudaFree(turbo_parm->decode_ext2);
	cudaFree(turbo_parm->decode_tmp);
	cudaFreeHost(turbo_parm->decode_h);
	free(turbo_parm);

	char i;
	for(i=0;i<2;i++)
	{
		cudaStreamDestroy(cuda_parm.stream[i]);
	}
}

__constant__ int alpha_table_0[32];
__constant__ int alpha_table_1[32];
__constant__ int beta_table_0[32];
__constant__ int beta_table_1[32];
__constant__ float alpha_par_table_0[32];
__constant__ float alpha_par_table_1[32];
__constant__ float beta_par_table_0[32];
__constant__ float beta_par_table_1[32];
__constant__ int interleaver[6144];
__constant__ int de_interleaver[6144];

void init_alloc()
{
	size_t pitch;
	
    cudaDeviceProp deviceprop;
    cudaGetDeviceProperties(&deviceprop,0);

    cudaSetDeviceFlags(cudaDeviceMapHost);

    if(deviceprop.canMapHostMemory!=1)
        printf("cudaError:cannot map host to device memory\n");

    cudaError_t result;
	
	turbo_parm = (turbo_parm_s*)malloc(sizeof(turbo_parm_s));
	
	// allocate CUDA memory 
	result = cudaMallocPitch((void**)&turbo_parm->sys_d, &pitch, 16*6144*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->sys_d failed, err_num=%d\n",result);
	
	result = cudaMallocPitch((void**)&turbo_parm->sys1_d, &pitch, 16*6144*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->sys1_d failed, err_num=%d\n",result);
	
	result = cudaMallocPitch((void**)&turbo_parm->sys2_d, &pitch, 16*6144*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->sys2_d failed, err_num=%d\n",result);
	
    result = cudaMallocPitch((void**)&turbo_parm->ypar1_d, &pitch, 16*6144*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->ypar1_d failed, err_num=%d\n",result);
	
	result = cudaMallocPitch((void**)&turbo_parm->ypar2_d, &pitch, 16*6144*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->ypar2_d failed, err_num=%d\n",result);
	
	result =  cudaMallocPitch((void**)&turbo_parm->alpha_d, &pitch, 16*8*(6144+648)*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->alpha failed, err_num=%d\n",result);
	
	result =  cudaMallocPitch((void**)&turbo_parm->alpha_pre_1, &pitch, 16*32*162*4*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->alpha_pre_1 failed, err_num=%d\n",result);
	
	result =  cudaMallocPitch((void**)&turbo_parm->alpha_pre_2, &pitch, 16*32*162*4*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->alpha_pre_2 failed, err_num=%d\n",result);
	
	result = cudaMallocPitch((void**)&turbo_parm->beta_pre_1, &pitch, 16*32*162*4*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->beta_pre1 failed, err_num=%d\n",result);
	
	result = cudaMallocPitch((void**)&turbo_parm->beta_pre_2, &pitch, 16*32*162*4*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->beta_pre2 failed, err_num=%d\n",result);
	
	result =  cudaMallocPitch((void**)&turbo_parm->ext_d, &pitch, 16*6144*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->ext_d failed, err_num=%d\n",result);
	
	result = cudaMallocPitch((void**)&turbo_parm->ext2_d, &pitch, 16*6144*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->ext2_d failed, err_num=%d\n",result);

	result = cudaMallocPitch((void**)&turbo_parm->decode_ext2, &pitch, 3*16*6144*sizeof(llr_t) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->ext2_d failed, err_num=%d\n",result);

	result = cudaMallocPitch((void**)&turbo_parm->decode_tmp, &pitch, 3*16*6144*sizeof(int) ,1);
	if(result!=cudaSuccess)
		printf("cudaMalloc turbo_parm->ext2_d failed, err_num=%d\n",result);

	result = cudaHostAlloc((void**)&turbo_parm->decode_h,3*16*768*sizeof(unsigned char), cudaHostAllocMapped);
	if(result!=cudaSuccess)
		printf("cudaHostAlloc turbo_parm->decode_h filaed, err_num=%d\n",result);
	
	// get device pointer
	result = cudaHostGetDevicePointer(&turbo_parm->decode_d, turbo_parm->decode_h, 0);
	if(result!=cudaSuccess)
		printf("cuda get device pinter decode_d failed, err_num=%d\n",result);

	// memset for mem
	cudaMemset(turbo_parm->ext2_d,0,16*6144*sizeof(llr_t));
	cudaMemset(turbo_parm->alpha_d,0,16*8*(6144+648)*sizeof(llr_t));
	cudaMemset(turbo_parm->decode_tmp,0,16*6144*sizeof(llr_t));
	//memset(turbo_parm->decode_h,0,16*768*sizeof(char));

	// init table for decoder
	int a_table_0[32]={0,3,4,7,1,2,5,6,8,11,12,15,9,10,13,14,16,19,20,23,17,18,21,22,24,27,28,31,25,26,29,30};
	cudaMemcpyToSymbol(alpha_table_0,a_table_0,32*sizeof(int));
	int a_table_1[32]={1,2,5,6,0,3,4,7,9,10,13,14,8,11,12,15,17,18,21,22,16,19,20,23,25,26,29,30,24,27,28,31};
	cudaMemcpyToSymbol(alpha_table_1,a_table_1,32*sizeof(int));
	
	float a_p_table_0[32] = {0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0};
	cudaMemcpyToSymbol(alpha_par_table_0,a_p_table_0,32*sizeof(llr_t));
	float a_p_table_1[32] = {1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0 ,1.0 ,0.0 ,0.0 ,1.0};
	cudaMemcpyToSymbol(alpha_par_table_1,a_p_table_1,32*sizeof(llr_t));
	
	int b_table_0[32]={0,4,5,1,2,6,7,3,8,12,13,9,10,14,15,11,16,20,21,17,18,22,23,19,24,28,29,25,26,30,31,27};
	cudaMemcpyToSymbol(beta_table_0,b_table_0,32*sizeof(int));
	int b_table_1[32]={4,0,1,5,6,2,3,7,12,8,9,13,14,10,11,15,20,16,17,21,22,18,19,23,28,24,25,29,30,26,27,31};
	cudaMemcpyToSymbol(beta_table_1,b_table_1,32*sizeof(int));
	
	float b_p_table_0[32] = {0.0 ,0.0 ,1.0 ,1.0 ,1.0 ,1.0 ,0.0 ,0.0 , 0.0 ,0.0 ,1.0 ,1.0 ,1.0 ,1.0 ,0.0 ,0.0 , 0.0 ,0.0 ,1.0 ,1.0 ,1.0 ,1.0 ,0.0 ,0.0 , 0.0 ,0.0 ,1.0 ,1.0 ,1.0 ,1.0 ,0.0 ,0.0};
	cudaMemcpyToSymbol(beta_par_table_0,b_p_table_0,32*sizeof(llr_t));
	float b_p_table_1[32] = {1.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,1.0 ,1.0 , 1.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,1.0 ,1.0 , 1.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,1.0 ,1.0 , 1.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,1.0 ,1.0};
	cudaMemcpyToSymbol(beta_par_table_1,b_p_table_1,32*sizeof(llr_t));
	
	
	// build de-interleaver table and interleaver table
	int i, j;
	unsigned long n;
	unsigned short f1, f2;
	for(j=0;j<188;j++)
	{
		n = f1f2mat[j].nb_bits;
		f1 = f1f2mat[j].f1;
		f2 = f1f2mat[j].f2;
		for(i=0;i<n;i++)
		{
			intable_h[j][i] = (((f1+f2*i)%n)*i)%n;
			detable_h[j][(((f1+f2*i)%n)*i)%n] = i;
		}
	}

	// for crc check and stream create
	for(i=0;i<2;i++)
	{
		cudaStreamCreate(&cuda_parm.stream[i]);
	}

	for(i=0;i<3;i++)
	{
		cudaEventCreate(&cuda_parm.s_check[i]);
	}
}
__device__ void compute_alpha(float* sys, float* sys1, float* sys2,
							  float* par, 
						      float* alpha, float* alpha_tmp,
						      float* alpha_pre_1, float* alpha_pre_2,
						      int num_per_block, int iteration_cnt, int decoder_id, int n, int codeword_num)
{
	int alpha_start = blockIdx.y*(n+gridDim.x*4)*8 + blockIdx.x*(num_per_block+1)*8*4;
	int index = blockIdx.y*n + blockIdx.x*num_per_block*4 + num_per_block*threadIdx.y;
	llr_t r0, r1;
	char i;
	
	alpha_tmp[threadIdx.x + 8*threadIdx.y] = 0;
	if(!(iteration_cnt==0 || (iteration_cnt==1 && decoder_id==2)))
	{
		if(!(blockIdx.x==0 && threadIdx.y==0))
		{
			if(decoder_id==1)
				alpha_tmp[threadIdx.x + 8*threadIdx.y] = alpha_pre_1[blockIdx.y*gridDim.x*32 + blockIdx.x*32 + threadIdx.x + 8*threadIdx.y -8];
			else
				alpha_tmp[threadIdx.x + 8*threadIdx.y] = alpha_pre_2[blockIdx.y*gridDim.x*32 + blockIdx.x*32 + threadIdx.x + 8*threadIdx.y -8];
		}
	}
	alpha[alpha_start + threadIdx.x + 8*threadIdx.y] = alpha_tmp[threadIdx.x + 8*threadIdx.y];
	__syncthreads();
	
	for(i=0;i<num_per_block;i++)
	{
		if(decoder_id==1)
		{
			r0 = alpha_par_table_0[threadIdx.x + 8*threadIdx.y]*par[index + i];
			r1 = sys1[index + i] + alpha_par_table_1[threadIdx.x + 8*threadIdx.y]*par[index + i];
		}
		else
		{
			r0 = alpha_par_table_0[threadIdx.x + 8*threadIdx.y]*par[index + i];
			r1 = sys2[index + i] + alpha_par_table_1[threadIdx.x + 8*threadIdx.y]*par[index + i];
		}
		alpha[alpha_start + (i+1)*32 + threadIdx.x + 8*threadIdx.y] = fmaxf(alpha_tmp[alpha_table_0[threadIdx.x+8*threadIdx.y]] + r0, alpha_tmp[alpha_table_1[threadIdx.x + 8*threadIdx.y]] + r1);
		__syncthreads();
		alpha_tmp[threadIdx.x + 8*threadIdx.y] = alpha[alpha_start + (i+1)*32 + threadIdx.x + 8*threadIdx.y];
		if(i==num_per_block-1)
		{
			if(decoder_id==1)
				alpha_pre_1[blockIdx.y*gridDim.x*32 + blockIdx.x*32 + threadIdx.x + 8*threadIdx.y] = alpha_tmp[threadIdx.x + 8*threadIdx.y];
			else
				alpha_pre_2[blockIdx.y*gridDim.x*32 + blockIdx.x*32 + threadIdx.x + 8*threadIdx.y] = alpha_tmp[threadIdx.x + 8*threadIdx.y];
		}
	}
	
}

__device__ void compute_beta_ext(float* sys, float* sys1, float* sys2,
							float* par, 
							float* alpha, float* beta_now, float* beta_next, 
							float* beta_pre_1, float* beta_pre_2,
							float* ext_tmp0, float* ext_tmp1,
							float* ext, float* ext2, float* decode_ext2,
							int num_per_block, int iteration_cnt, int decoder_id, int n, int codeword_num)
{
	llr_t a, r0, r1, max_0, max_1;
	int alpha_start = blockIdx.y*(n+gridDim.x*4)*8 + blockIdx.x*(num_per_block+1)*8*4;
	int index = blockIdx.y*n + blockIdx.x*num_per_block*4 + threadIdx.y*num_per_block;
	int index2;
	char i,j;
	
	beta_now[threadIdx.x + 8*threadIdx.y] = 0;
	if(!(iteration_cnt==0 || (iteration_cnt==1 && decoder_id==2)))
	{
		if(!(blockIdx.x==gridDim.x-1 && threadIdx.y==3))
		{
			if(decoder_id==1)
				beta_now[threadIdx.x + 8*threadIdx.y] = beta_pre_1[blockIdx.y*gridDim.x*32 + blockIdx.x*32 + threadIdx.x + 8*threadIdx.y +8];
			else
				beta_now[threadIdx.x + 8*threadIdx.y] = beta_pre_2[blockIdx.y*gridDim.x*32 + blockIdx.x*32 + threadIdx.x + 8*threadIdx.y +8];
		}
	}
	__syncthreads();
	
	for(i=num_per_block-1;i>=0;i--)
	{
		if(decoder_id==1)
		{
			r0 = beta_par_table_0[threadIdx.x + 8*threadIdx.y]*par[index + i];
			r1 = sys1[index + i] + beta_par_table_1[threadIdx.x + 8*threadIdx.y]*par[index + i];
		}
		else
		{
			r0 = beta_par_table_0[threadIdx.x + 8*threadIdx.y]*par[index + i];
			r1 = sys2[index + i] + beta_par_table_1[threadIdx.x + 8*threadIdx.y]*par[index + i];
		}
		a = alpha[alpha_start + 32*i + threadIdx.x + 8*threadIdx.y];
		beta_next[threadIdx.x + 8*threadIdx.y] = fmaxf(beta_now[beta_table_0[threadIdx.x+8*threadIdx.y]] + r0, beta_now[beta_table_1[threadIdx.x + 8*threadIdx.y]] + r1);
		if(i==0)
		{
			if(decoder_id==1)
				beta_pre_1[blockIdx.y*gridDim.x*32 + blockIdx.x*32 + threadIdx.x + 8*threadIdx.y] = beta_next[threadIdx.x + 8*threadIdx.y];
			else
				beta_pre_2[blockIdx.y*gridDim.x*32 + blockIdx.x*32 + threadIdx.x + 8*threadIdx.y] = beta_next[threadIdx.x + 8*threadIdx.y];
		}
		ext_tmp0[((num_per_block-1-i)&7)*32 + threadIdx.x + 8*threadIdx.y] = a + r0 + beta_now[beta_table_0[threadIdx.x + 8*threadIdx.y]];
		ext_tmp1[((num_per_block-1-i)&7)*32 + threadIdx.x + 8*threadIdx.y] = a + r1 + beta_now[beta_table_1[threadIdx.x + 8*threadIdx.y]];
		__syncthreads();
		beta_now[threadIdx.x + 8*threadIdx.y] = beta_next[threadIdx.x + 8*threadIdx.y];
		
		if(((num_per_block-1-i)&7)==7)
		{
			max_0 = ext_tmp0[(7-threadIdx.x)*32 + threadIdx.x + 8*threadIdx.y];
			max_1 = ext_tmp1[(7-threadIdx.x)*32 + threadIdx.x + 8*threadIdx.y];
			for(j=1;j<8;j++)
			{
				index2 = (threadIdx.x + j)&7;
				max_0 = fmaxf(max_0, ext_tmp0[(7-threadIdx.x)*32 + index2 + 8*threadIdx.y]);
				max_1 = fmaxf(max_1, ext_tmp1[(7-threadIdx.x)*32 + index2 + 8*threadIdx.y]);
			}
			index2 = blockIdx.x*num_per_block*4 + threadIdx.y*num_per_block + i + threadIdx.x;
			if(decoder_id==1)
			{
				ext[blockIdx.y*n + index2] = max_1 - max_0 - sys1[blockIdx.y*n + index2] + sys[blockIdx.y*n + index2];
				sys2[blockIdx.y*n + de_interleaver[index2]] = ext[blockIdx.y*n + index2];
			}
			else
			{
				ext2[blockIdx.y*n + index2] = max_1 - max_0;
				if(iteration_cnt >= 3)
				{
					decode_ext2[(iteration_cnt-3)*gridDim.y*n + blockIdx.y*n + interleaver[index2]] = max_1 - max_0;
				}
				sys1[blockIdx.y*n + interleaver[index2]] = max_1 - max_0 - ext[blockIdx.y*n + interleaver[index2]] + sys[blockIdx.y*n + interleaver[index2]];
			}
		}
		else if(i==0 && (num_per_block&7)!=0 && threadIdx.x < (num_per_block&7))
		{
			max_0 = ext_tmp0[((num_per_block&7)-1-threadIdx.x)*32 + threadIdx.x + 8*threadIdx.y];
			max_1 = ext_tmp1[((num_per_block&7)-1-threadIdx.x)*32 + threadIdx.x + 8*threadIdx.y];
			for(j=1;j<8;j++)
			{
				index2 = (threadIdx.x + j)&7;
				max_0 = fmaxf(max_0, ext_tmp0[((num_per_block&7)-1-threadIdx.x)*32 + index2 + 8*threadIdx.y]);
				max_1 = fmaxf(max_1, ext_tmp1[((num_per_block&7)-1-threadIdx.x)*32 + index2 + 8*threadIdx.y]);
			}
			index2 = blockIdx.x*num_per_block*4 + threadIdx.y*num_per_block + i + threadIdx.x;
			if(decoder_id==1)
			{
				ext[blockIdx.y*n + index2] = max_1 - max_0 - sys1[blockIdx.y*n + index2] + sys[blockIdx.y*n + index2];
				sys2[blockIdx.y*n + de_interleaver[index2]] = ext[blockIdx.y*n + index2];
			}
			else
			{
				ext2[blockIdx.y*n + index2] = max_1 - max_0;
				if(iteration_cnt >= 3)
				{
					decode_ext2[(iteration_cnt-3)*gridDim.y*n + blockIdx.y*n + interleaver[index2]] = max_1 - max_0;
				}
				sys1[blockIdx.y*n + interleaver[index2]] = max_1 - max_0 - ext[blockIdx.y*n + interleaver[index2]] + sys[blockIdx.y*n + interleaver[index2]];
			}
		}
	}
}

__global__ void log(float* sys, float* sys1, float* sys2,
					float* par,
					float* alpha, 
					float* alpha_pre_1, float* alpha_pre_2, float* beta_pre_1, float* beta_pre_2,
					float* ext, float* ext2, float* decode_ext2,
					int num_per_block, int iteration_cnt, int decoder_id, int n, int codeword_num
					)
{
	__shared__ llr_t alpha_tmp[32];
	__shared__ llr_t beta_tmp[32];
	__shared__ llr_t ext_tmp0[32*8];
	__shared__ llr_t ext_tmp1[32*8];
	
	compute_alpha(sys, sys1, sys2,
				   par,
				   alpha, alpha_tmp,
				   alpha_pre_1, alpha_pre_2,
				   num_per_block, iteration_cnt, decoder_id, n , codeword_num
				   );
	__syncthreads();			  
	
	compute_beta_ext(sys, sys1, sys2,
					 par,
					 alpha, alpha_tmp,beta_tmp,
					 beta_pre_1, beta_pre_2,
					 ext_tmp0, ext_tmp1,
					 ext, ext2, decode_ext2,
					 num_per_block, iteration_cnt, decoder_id, n, codeword_num
					);
		 
}

// for decoding
__global__ void decode(llr_t* decode_ext2, unsigned char* decode_d, int* decode_tmp, int n, int decode_len, int iteration_cnt)
{
	int i, j;
	for(i=threadIdx.x; i<n; i+=256)
	{
		decode_tmp[iteration_cnt*gridDim.x*n + blockIdx.x*n + i] = 0;
		if(decode_ext2[iteration_cnt*gridDim.x*n + blockIdx.x*n + i] > 0)
		{
			decode_tmp[iteration_cnt*gridDim.x*n + blockIdx.x*n + i] = 1 << (7-(i&7));
		}
	}
	__syncthreads();
	for(i=threadIdx.x; i<decode_len; i+=256)
	{
		decode_d[iteration_cnt*gridDim.x*decode_len + blockIdx.x*decode_len + i] = 0;
		for(j=0;j<8;j++)
		{
			decode_d[iteration_cnt*gridDim.x*decode_len + blockIdx.x*decode_len + i] += decode_tmp[iteration_cnt*gridDim.x*n + blockIdx.x*n + i*8 + j];
		}
	}
}

//#define TIME_EST

unsigned char phy_threegpplte_turbo_decoder_gpu(short **y,
        unsigned char **decoded_bytes,
		unsigned int codeword_num,
        unsigned short n,
        unsigned short f1,
        unsigned short f2,
        unsigned char max_iterations,
        unsigned char crc_type,
        unsigned char *f_tmp,
		unsigned char* ret)
{
	unsigned int i,j,iind,k;
#ifdef TIME_EST
	cudaEventCreate(&cuda_parm.e_start);
	cudaEventCreate(&cuda_parm.e_stop);
	cuda_parm.e_time = 0;
	cudaEventRecord(cuda_parm.e_start, cuda_parm.stream[0]);
#endif

    llr_t sys_h[n*codeword_num], ypar1_h[n*codeword_num], ypar2_h[n*codeword_num];
	
	unsigned char iteration_cnt=0;
	unsigned int crc,oldcrc,crc_len;
	uint8_t temp;
	unsigned char F;
	
	if (crc_type > 3) {
		printf("Illegal crc length!\n");
		return 255;
	}
	
	for (iind=0; f1f2mat[iind].nb_bits!=n && iind <188; iind++);
	
	if ( iind == 188 ) {
		printf("Illegal frame length!\n");
		return 255;
	}
	
	switch (crc_type) {
		case CRC24_A:
		case CRC24_B:
			crc_len=3;
			break;

		case CRC16:
			crc_len=2;
			break;

		case CRC8:
			crc_len=1;
			break;

		default:
			crc_len=3;
	}
	
	// fetch data for each codeword
#ifdef TIME_EST
	cudaEventCreate(&cuda_parm.f_start);
	cudaEventCreate(&cuda_parm.f_stop);
	cuda_parm.f_time = 0;
	cudaEventRecord(cuda_parm.f_start,cuda_parm.stream[0]);
#endif
	short* yp;
	for(i=0;i<codeword_num;i++)
	{
		yp = y[i];
		for(j=0;j<n;j++)
		{
			sys_h[j+n*i] = *yp;
			ypar1_h[j+n*i] = *(yp+1);
			ypar2_h[j+n*i] = *(yp+2);
			yp+=3;
		}
	}
#ifdef TIME_EST
	cudaEventRecord(cuda_parm.f_stop,cuda_parm.stream[0]);
	cudaEventSynchronize(cuda_parm.f_stop);
	cuda_parm.f_time = 0;
	cudaEventElapsedTime(&cuda_parm.f_time, cuda_parm.f_start, cuda_parm.f_stop);
	printf("Fetech data time = %f ms\n",cuda_parm.f_time);
#endif

	// for kernel  memcpy
#ifdef TIME_EST
	cudaEventCreate(&cuda_parm.m_start);
	cudaEventCreate(&cuda_parm.m_stop);
	cudaEventRecord(cuda_parm.m_start, cuda_parm.stream[0]);
#endif
	cudaMemcpyAsync(turbo_parm->sys_d,sys_h,codeword_num*n*sizeof(llr_t),cudaMemcpyHostToDevice, cuda_parm.stream[0]);
	cudaMemcpyAsync(turbo_parm->ypar1_d,ypar1_h,codeword_num*n*sizeof(llr_t),cudaMemcpyHostToDevice, cuda_parm.stream[0]);
	cudaMemcpyAsync(turbo_parm->ypar2_d,ypar2_h,codeword_num*n*sizeof(llr_t),cudaMemcpyHostToDevice, cuda_parm.stream[0]);
	cudaMemcpyToSymbolAsync(interleaver, intable_h[iind],n*sizeof(int), 0, cudaMemcpyHostToDevice,  cuda_parm.stream[0]);
	cudaMemcpyToSymbolAsync(de_interleaver, detable_h[iind],n*sizeof(int), 0, cudaMemcpyHostToDevice,  cuda_parm.stream[0]);
#ifdef TIME_EST
	cudaEventRecord(cuda_parm.m_stop, cuda_parm.stream[0]);
	cudaEventSynchronize(cuda_parm.m_stop);	
	cuda_parm.m_time=0;
	cudaEventElapsedTime(&cuda_parm.m_time, cuda_parm.m_start, cuda_parm.m_stop);
	printf("Memcpy Time For Kernel = %f ms\n",cuda_parm.m_time);
#endif

#ifdef TIME_EST	
	cuda_parm.t_time = 0;
	cudaEventCreate(&cuda_parm.t_start);
	cudaEventCreate(&cuda_parm.t_stop);
	cudaEventRecord(cuda_parm.t_start,cuda_parm.stream[0]);
#endif
	// decide block and thread
	int blocknum=648;
	while(blocknum!=8)
	{
		if(n%blocknum==0 && n/blocknum>=16)
		{
			break;
		}
		blocknum-=4;
	}
	
	dim3 threadnum(8,4);
	size_t s_size = 0;
	int num_per_block = n / blocknum;
	blocknum = blocknum/4;
	dim3 bb(blocknum, codeword_num);
	
	llr_t ext2[n*codeword_num];
	memset(ext2,0,sizeof(ext2));
	llr_t tmp[n*codeword_num];
	
	// for crc check
	char check=0;
	int z;
	// log map algorithm
	log<<<bb, threadnum, s_size, cuda_parm.stream[0]>>>(turbo_parm->sys_d, turbo_parm->sys_d, turbo_parm->sys2_d,
											  turbo_parm->ypar1_d,
											  turbo_parm->alpha_d,
											  turbo_parm->alpha_pre_1, turbo_parm->alpha_pre_2, turbo_parm->beta_pre_1, turbo_parm->beta_pre_2,
											  turbo_parm->ext_d, turbo_parm->ext2_d, turbo_parm->decode_ext2,
											  num_per_block, iteration_cnt, 1, n, codeword_num
											);
	
	while(iteration_cnt++ < max_iterations)
	{
		
		log<<<bb, threadnum, s_size, cuda_parm.stream[0]>>>(turbo_parm->sys_d, turbo_parm->sys1_d, turbo_parm->sys2_d,
												  turbo_parm->ypar2_d,
												  turbo_parm->alpha_d,
												  turbo_parm->alpha_pre_1, turbo_parm->alpha_pre_2, turbo_parm->beta_pre_1, turbo_parm->beta_pre_2,
												  turbo_parm->ext_d, turbo_parm->ext2_d, turbo_parm->decode_ext2,
												  num_per_block, iteration_cnt, 2, n, codeword_num
												);
		
		if(iteration_cnt>=3)
		{
			cudaEventRecord(cuda_parm.s_check[iteration_cnt-3], cuda_parm.stream[0]);	
		}

		if(iteration_cnt < max_iterations)
		{
			log<<<bb, threadnum, s_size, cuda_parm.stream[0]>>>(turbo_parm->sys_d, turbo_parm->sys1_d, turbo_parm->sys2_d,
													  turbo_parm->ypar1_d,
													  turbo_parm->alpha_d,
													  turbo_parm->alpha_pre_1, turbo_parm->alpha_pre_2, turbo_parm->beta_pre_1, turbo_parm->beta_pre_2,
													  turbo_parm->ext_d, turbo_parm->ext2_d, turbo_parm->decode_ext2,
													  num_per_block, iteration_cnt, 1, n, codeword_num
													);
		}
	}
	cudaDeviceSynchronize();
	
#ifdef TIME_EST
	cudaEventRecord(cuda_parm.t_stop, cuda_parm.stream[0]);
	cudaEventSynchronize(cuda_parm.t_stop);
	cudaEventElapsedTime(&cuda_parm.t_time, cuda_parm.t_start, cuda_parm.t_stop);
	
	printf("Time For turbo algorithm Kernel = %f ms\n",cuda_parm.t_time);	
#endif	

	int decode_len = n >> 3;
	for(i=0;i<=2;i++)
	{
		check = 0;
		// wait for turbo ext
		cudaStreamWaitEvent(cuda_parm.stream[1], cuda_parm.s_check[i], 0);
#ifdef TIME_EST
	cuda_parm.d_time = 0;
	cudaEventCreate(&cuda_parm.d_start);
	cudaEventCreate(&cuda_parm.d_stop);
	cudaEventRecord(cuda_parm.d_start, cuda_parm.stream[1]);
#endif
		// decode
		decode<<<codeword_num, 256, 0, cuda_parm.stream[1]>>>(turbo_parm->decode_ext2, turbo_parm->decode_d, turbo_parm->decode_tmp, n, decode_len, i);
		cudaStreamSynchronize(cuda_parm.stream[1]);

		// crc check
		for(j=0;j<codeword_num;j++)
		{
			F = f_tmp[1];
			if(j==0)
			{
				F = f_tmp[0];
			}
			oldcrc = *((unsigned int *)(&turbo_parm->decode_h[i*codeword_num*decode_len + j*decode_len + decode_len-crc_len]));
			switch(crc_type)
			{
				case CRC24_A:
					oldcrc&=0x00ffffff;
					crc = crc24a(&turbo_parm->decode_h[i*codeword_num*decode_len + j*decode_len + ( F>>3 )], n-24-F)>>8;
					temp=((uint8_t *)&crc)[2];
					((uint8_t *)&crc)[2] = ((uint8_t *)&crc)[0];
					((uint8_t *)&crc)[0] = temp;
					break;
				case CRC24_B:
					oldcrc&=0x00ffffff;
					crc = crc24b(&turbo_parm->decode_h[i*codeword_num*decode_len + j*decode_len],
								 n-24)>>8;
					temp=((uint8_t *)&crc)[2];
					((uint8_t *)&crc)[2] = ((uint8_t *)&crc)[0];
					((uint8_t *)&crc)[0] = temp;
					break;

				case CRC16:
					oldcrc&=0x0000ffff;
					crc = crc16(&turbo_parm->decode_h[i*codeword_num*decode_len + j*decode_len],
								n-16)>>16;
					break;

				case CRC8:
					oldcrc&=0x000000ff;
					crc = crc8(&turbo_parm->decode_h[i*codeword_num*decode_len + j*decode_len],
							   n-8)>>24;
					break;

				default:
					printf("FATAL: 3gpplte_turbo_decoder_sse.c: Unknown CRC\n");
					return(255);
					break;
			}

			if ((crc == oldcrc) && (crc!=0)) {
				ret[j] = i+3-1;
			}
			else
			{
				check = 1;
			}			
		}

#ifdef TIME_EST
	cudaEventRecord(cuda_parm.d_stop, cuda_parm.stream[1]);
	cudaEventSynchronize(cuda_parm.d_stop);
	cudaEventElapsedTime(&cuda_parm.d_time, cuda_parm.d_start, cuda_parm.d_stop);
	printf("Time For decode & crc check = %f ms\n",cuda_parm.d_time);
#endif	
		if(check==0)
		{
			for(j=0;j<codeword_num;j++)
			{
				for(k=0;k<decode_len;k++)
				{
					decoded_bytes[j][k] = turbo_parm->decode_h[i*codeword_num*decode_len + j*decode_len + k];
				}
			}
			
#ifdef TIME_EST
	cudaEventRecord(cuda_parm.e_stop, cuda_parm.stream[0]);
	cudaEventSynchronize(cuda_parm.e_stop);
	cudaEventElapsedTime(&cuda_parm.e_time, cuda_parm.e_start, cuda_parm.e_stop);
	printf("Time For CUDA = %f ms\n",cuda_parm.e_time);
#endif
			return i+3-1;
		}
	}

	// crc check fail
	for(i=0;i<codeword_num;i++)
	{
		for(j=0;j<decode_len;j++)
		{
			decoded_bytes[i][j] = turbo_parm->decode_h[2*codeword_num*decode_len + i*decode_len + j];
		}
		ret[i] = 5;
		//return 5;
	}
	
#ifdef TIME_EST
	cudaEventRecord(cuda_parm.e_stop, cuda_parm.stream[0]);
	cudaEventSynchronize(cuda_parm.e_stop);
	cudaEventElapsedTime(&cuda_parm.e_time, cuda_parm.e_start, cuda_parm.e_stop);
	printf("Time For CUDA = %f ms\n",cuda_parm.e_time);
#endif	
	for(i=0;i<2;i++)
	{
		cudaStreamSynchronize(cuda_parm.stream[i]);
	}
    return 5;
}