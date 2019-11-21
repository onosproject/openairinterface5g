/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

 /*! \file CUDA_phy_procedure.cu
 * \brief Create and Implementation of beamforming and ifft in gpu(resource allocate)
 * \author TY Hsu, CW Chang
 * \date 2018
 * \version 0.2
 * \company ISIP@NCTU and Eurecom
 * \email: tyhsu@cs.nctu.edu.tw, zhang0756107.cs07g@nctu.edu.tw
 * \note
 * \warning
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "CUDA/checkError.h"
#include "CUDA/struct.h"
#include "CUDA/cuda_struct.h"

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

__global__ void cu_CP_fft_resultin(unsigned char nb_prefix_samples, int *input, int *output, int fftsize, int nb_symbols){
	int id = blockIdx.x*1024 + threadIdx.x;
	int elementId = id%fftsize;
	int symbolId = id/fftsize;
	int slotId = symbolId/nb_symbols;
	int symbIdinSlot = symbolId%nb_symbols;
	int slotElmtNum = fftsize*(nb_symbols+1);
	int CPElmtNum = fftsize+nb_prefix_samples;

	int offset = slotId*slotElmtNum + symbIdinSlot*CPElmtNum;
	output[offset + nb_prefix_samples + elementId] = input[id];
	if(elementId >= fftsize-nb_prefix_samples){
		output[offset + (fftsize-nb_prefix_samples)] = input[id];
	}
}

__global__ void cu_CP0_fft_resultin(unsigned char nb_prefix_samples0, unsigned char nb_prefix_samples, 
								int *input, int *output, int fftsize, int nb_symbols){
	int id = blockIdx.x*1024 + threadIdx.x;
	int elementId = id%fftsize;
	int symbolId = id/fftsize;
	int slotId = symbolId/nb_symbols;
	int symbIdinSlot = symbolId%nb_symbols;
	int slotElmtNum = fftsize*(nb_symbols+1);
	int CP0ElmtNum = fftsize+nb_prefix_samples0;
	int CPElmtNum = fftsize+nb_prefix_samples;

	if(symbIdinSlot==0){
		int offset = slotId*slotElmtNum;
		output[offset + nb_prefix_samples0+ elementId] = input[id];
		if(elementId >= fftsize-nb_prefix_samples0){
			output[offset + (fftsize-nb_prefix_samples0)] = input[id];
		} 
	}else{
		int offset = slotId*slotElmtNum + CP0ElmtNum + (symbIdinSlot-1)*CPElmtNum;
		output[offset + nb_prefix_samples + elementId] = input[id];
		if(elementId >= fftsize-nb_prefix_samples){
			output[offset + (fftsize-nb_prefix_samples)] = input[id];
		}
	}
}

extern "C" void CUDA_ifft_ofdm( int **output, 
				int fftsize, 
				unsigned char nb_symbols, 
				unsigned char nb_prefix_samples,
				unsigned char nb_prefix_samples0,
				int nb_tx,
				int Ncp,
				Extension_t etype){
	//for(int i=0; i<fftsize; i++) printf("%d+%di\n", ((short*)&input[0][i])[0], ((short*)&input[0][i])[1]);
	
	int *d_txdataF_BF = cu_ru.d_txdataF_BF;
	int *d_data_wCP = cu_ru.d_data_wCP;
	Complex *d_signal = cu_ru.d_signal;
	cufftHandle plan = cu_ru.plan; 

	/*
	for(int aa=0; aa<nb_tx; aa++){
		int elementNum = fftsize*nb_symbols;
		gpuErrchk( cudaMemcpy(&d_data[aa*elementNum], input[aa], sizeof(int)*elementNum, cudaMemcpyHostToDevice) );
	}*/

	int threadNum = 1024;
	int blockNum = fftsize*nb_symbols*nb_tx / threadNum;
	//cu_intToComplex<<<blockNum, threadNum>>>(d_txdataF_BF, d_signal);
	//CHECK_STATE("cu_intToComplex");

	cufftErrchk( cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE));

	cu_ComplexToInt<<<blockNum, threadNum>>>(d_txdataF_BF, d_signal);
	//CHECK_STATE("cu_ComplexToInt");


	//only do cyclic_prefix, suffix/none not finish yet.
	if(Ncp==1){
		cu_CP_fft_resultin<<<blockNum, threadNum>>>(nb_prefix_samples, d_txdataF_BF, d_data_wCP, fftsize, nb_symbols);
		//CHECK_STATE("cu_CP_fft_resultin");	
	}else{
		cu_CP0_fft_resultin<<<blockNum, threadNum>>>(nb_prefix_samples0, nb_prefix_samples, d_txdataF_BF, d_data_wCP, fftsize, nb_symbols);
		//CHECK_STATE("cu_CP0_fft_resultin");	
	}

	//write back gpu->cpu
	for(int aa=0; aa<nb_tx; aa++){
		gpuErrchk( cudaMemcpy(output[aa], &d_data_wCP[aa*(nb_symbols+1)*fftsize], fftsize*(nb_symbols+1)*sizeof(int), cudaMemcpyDeviceToHost) );
	}
	cudaDeviceSynchronize();
	

}

__device__ inline void beamComp(int *res, int *x1, int *x2){
	((short*)res)[0] += ((short*)x1)[0]*((short*)x2)[0] + ((short*)x1)[1]*((short*)x2)[1];
	((short*)res)[1] += ((short*)x1)[0]*((short*)x2)[1] - ((short*)x1)[1]*((short*)x2)[0];
}

extern __constant__ int PORTSIZE;
extern __constant__ int SUBTXSIZE;
extern __constant__ int BW_PSIZE;

__global__ void	conjMulAll(int* txdataF, int* weight, int* sub,
		int fftsize, int nb_symbols, int nb_tx, int nb_antenna_ports){
	__shared__ int x1[2048*4];
	__shared__ int res[2048];
		
	int id = threadIdx.x;
	int id2 = id+1024;
	int symbId = blockIdx.x;
	int portStart = blockIdx.y*4;
	int subtxId = blockIdx.y;
	

	int s1 = 0;
	for(int p=portStart; p<portStart+4; p++){
		x1[s1*2048+id] = txdataF[p*PORTSIZE + symbId*fftsize + id];
		x1[s1*2048+id2] = txdataF[p*PORTSIZE + symbId*fftsize + id2];
		s1++;
	}

	for(int aa=0; aa<nb_tx; aa++){
		res[id] = 0;
		res[id2] = 0;
		s1 = 0;
		for(int p=portStart; p<portStart+4; p++){
			beamComp(&res[id], &x1[s1*2048+id], &weight[p*BW_PSIZE+aa*fftsize+id]);
			beamComp(&res[id2], &x1[s1*2048+id2], &weight[p*BW_PSIZE+aa*fftsize+id2]);
			/*
			if(id==0){
				printf("%5d+%5di mul %5d+%5di = %5d+%5di\n", 
						((short*)&x1[s1*2048+id])[0], ((short*)&x1[s1*2048+id])[1],
						((short*)&weight[p*BW_PSIZE+aa*fftsize+id])[0],((short*)&weight[p*BW_PSIZE+aa*fftsize+id])[1],
						((short*)&res[id])[0], ((short*)&res[id])[1]);
			}*/
			s1++;
		}

		int offset = subtxId*SUBTXSIZE + aa*PORTSIZE + symbId*fftsize;
		sub[offset+id] = res[id];
		sub[offset+id2] = res[id2];
	} 	
}

__device__ inline void partAdd(Complex *res, int *x1, int *x2){
	res->x = ((short*)x1)[0] + ((short*)x2)[0];
	res->y = ((short*)x1)[1] + ((short*)x2)[1];
}

__global__ void	combine(int* subtx, Complex* d_signal, int fftsize, int nb_symbols, int nb_tx, int nb_antenna_ports){
	int id = threadIdx.x;
	int id2 = id+1024;
	int aa = blockIdx.x;
	int symbStart = blockIdx.y*7;
	int symbEnd = symbStart + 7;

	for(int symb=symbStart; symb<symbEnd; symb++){
		int offset = aa*PORTSIZE + symb*fftsize;
		partAdd(&d_signal[offset+id], &subtx[offset+id], &subtx[SUBTXSIZE+offset+id]);
		partAdd(&d_signal[offset+id2], &subtx[offset+id2], &subtx[SUBTXSIZE+offset+id2]);
		//if(id==0) printf("%5.5f+%5.5fi\n", d_signal[offset+id].x, d_signal[offset+id].y);
		
	}

}

extern "C" void CUDA_beam_precoding(int **txdataF, int ***weight, int L_ssb, int shift, int fftsize, int nb_symbols, int nb_antenna_ports, int nb_tx){
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//initial BF data;
	gpuErrchk( cudaMemset(cu_ru.d_txdataF_BF, 0, fftsize*nb_symbols*sizeof(int)*nb_tx) );
	gpuErrchk( cudaMemset(cu_ru.d_subtx, 0, fftsize*nb_symbols*nb_tx*2*sizeof(int)) );
	//move data to gpu
	int slotsize = fftsize*nb_symbols;
	for(int p=0; p<nb_antenna_ports; p++){
		gpuErrchk( cudaMemcpy(&cu_ru.d_txdataF[p*slotsize], txdataF[p], slotsize*sizeof(int), cudaMemcpyHostToDevice) );	
	}
	
	cudaEventRecord(start);
	int div = 1<<shift;
	for(int aa=0; aa<nb_tx; aa++){
		for(int p=0; p<nb_antenna_ports; p++){
			if((L_ssb>>p) & 0x01){
				gpuErrchk( cudaMemcpy(&cu_ru.d_weight[p*(nb_tx*fftsize)+aa*fftsize], 
							weight[p][aa], fftsize*sizeof(int), cudaMemcpyHostToDevice) );
			}
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,  start, stop);
	printf("HTD: %f\n", time*1000.0);

	cudaEventRecord(start);

	dim3 block(14,2,1);
	dim3 thread(1024);
	conjMulAll<<<block, thread>>>(cu_ru.d_txdataF, cu_ru.d_weight, cu_ru.d_subtx,
		fftsize, nb_symbols, nb_tx, nb_antenna_ports);
	block = dim3(8, 2, 1);
	combine<<<block, thread>>>(cu_ru.d_subtx, cu_ru.d_signal, 
		fftsize, nb_symbols, nb_tx, nb_antenna_ports);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,  start, stop);
	printf("conjMul+comb: %f\n", time*1000.0);


}

