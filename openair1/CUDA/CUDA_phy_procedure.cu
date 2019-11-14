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
 * \brief Create and Implementation of beamforming and ifft in gpu
 * \author TY Hsu, CW Chang
 * \date 2018
 * \version 0.1
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
	cu_intToComplex<<<blockNum, threadNum>>>(d_txdataF_BF, d_signal);
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

__global__ void conjMul(int *d_x1, int *d_x2, int *d_y, int aa, int div, int fftsize, int nb_symbols){
	int id = blockIdx.x*1024 + threadIdx.x;
	int *x1 = &d_x1[id];
	int *x2 = &d_x2[id%fftsize];
	int *y = &d_y[aa*fftsize*nb_symbols + id];

	int re, im;
	//conj(x1) * x2
	re = ((short*)x1)[0]*((short*)x2)[0] + ((short*)x1)[1]*((short*)x2)[1];
	im = ((short*)x1)[0]*((short*)x2)[1] - ((short*)x1)[1]*((short*)x2)[0];

	re = re / div;
	im = im / div;

	((short*)y)[0] += re; 
	((short*)y)[1] += im;	

}


extern "C" void CUDA_beam_precoding(int **txdataF, int ***weight, int L_ssb, int shift, int fftsize, int nb_symbols, int nb_antenna_ports, int nb_tx){

	//initial BF data;
	gpuErrchk( cudaMemset(cu_ru.d_txdataF_BF, 0, fftsize*nb_symbols*sizeof(int)*nb_tx) );
	//move data to gpu
	for(int p=0; p<nb_antenna_ports; p++){
		gpuErrchk( cudaMemcpy(cu_ru.d_txdataF[p], txdataF[p], fftsize*sizeof(int)*nb_symbols, cudaMemcpyHostToDevice) );	
	}

	
	int threadNum = 1024;
	int blockNum = fftsize*nb_symbols/threadNum;
	int div = 1<<shift;
	for(int aa=0; aa<nb_tx; aa++){
		for(int p=0; p<nb_antenna_ports; p++){
			if((L_ssb>>p) & 0x01){
				gpuErrchk( cudaMemcpy(cu_ru.d_weight[p][aa], weight[p][aa], fftsize*sizeof(int), cudaMemcpyHostToDevice) );
				conjMul<<<blockNum, threadNum>>>(cu_ru.d_txdataF[p], cu_ru.d_weight[p][aa], 
							cu_ru.d_txdataF_BF, aa, div, fftsize, nb_symbols);
			}
		}
	}


}

