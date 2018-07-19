/*! \file PHY\CUDA/LTE_TRANSPORT/turbo_parm.h
* \brief turbo decoder using gpu 
* \author TerngYin Hsu, JianYa Chu
* \date 2018
* \version 0.1
* \company ISIP LAB/NCTU CS
* \email: tyhsu@cs.nctu.edu.tw
* \note
* \warning  
*/


#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
typedef float llr_t;
typedef struct
{
	llr_t *sys_d;
	llr_t *sys1_d;
	llr_t *sys2_d;
	llr_t *ypar1_d;
    llr_t *ypar2_d;
	llr_t *ext_d;
	llr_t *ext2_d;
	llr_t *alpha_d;
	int *decode_tmp;
	llr_t *decode_ext2;
	llr_t *alpha_pre_1;
	llr_t *alpha_pre_2;
	llr_t *beta_pre_1;
	llr_t *beta_pre_2;
	unsigned char *decode_h;
	unsigned char *decode_d;
}turbo_parm_s;

typedef struct 
{
	// for excution time
	cudaEvent_t e_start; 
	cudaEvent_t e_stop;
	float e_time;
	// for fetch time
	cudaEvent_t f_start, f_stop;
	float f_time;
	// for memcpy
	cudaEvent_t m_start, m_stop;
	float m_time;
	// for turbo kernel excution time
	cudaEvent_t t_start, t_stop;
	float t_time;
	// for decode excution time
	cudaEvent_t d_start, d_stop;
	float d_time;
	// for crc check
	cudaEvent_t s_check[3];
	// for algorithm & decode stream
	cudaStream_t stream[2];


}cuda_parm_s;

turbo_parm_s* turbo_parm;
cuda_parm_s cuda_parm; 