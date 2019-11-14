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

 /*! \file init_cuda.cu
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

cuda_cu_ru cu_ru;

extern "C" void init_cuda(int nb_tx, int nb_symbols, int fftsize){
	printf("init_cuda %d %d %d  \n\n\n", nb_tx, nb_symbols, fftsize);

	int nb_antenna_ports = 8;

	//beamforming precoding
	cu_ru.d_txdataF = (int**)malloc(sizeof(int*) * nb_antenna_ports);
	for(int p=0; p<nb_antenna_ports; p++){
		gpuErrchk( cudaMalloc((void**)&cu_ru.d_txdataF[p], fftsize*sizeof(int)*nb_symbols) );
	}
	cu_ru.d_beam_stream = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nb_tx);
	for(int aa=0; aa<nb_tx; aa++){
		gpuErrchk( cudaStreamCreate(&cu_ru.d_beam_stream[aa]) );	
	}


	cu_ru.d_weight = (int***)malloc(sizeof(int**) * nb_antenna_ports);
	for(int p=0; p<nb_antenna_ports; p++){
		cu_ru.d_weight[p] = (int**)malloc(sizeof(int*) * nb_tx);
		for(int aa=0; aa<nb_tx; aa++){
			gpuErrchk( cudaMalloc((void**)&cu_ru.d_weight[p][aa], fftsize*sizeof(int)) );
		}
	}

	//ifft	
	gpuErrchk( cudaMalloc((void**)&cu_ru.d_txdataF_BF, fftsize*sizeof(int)*nb_symbols*nb_tx) );
	gpuErrchk( cudaMalloc((void**)&cu_ru.d_signal, fftsize*sizeof(Complex)*nb_symbols*nb_tx) );
	gpuErrchk( cudaMalloc((void**)&cu_ru.d_data_wCP, fftsize*(nb_symbols+1)*nb_tx*sizeof(int)) );
	cufftErrchk( cufftPlan1d(&cu_ru.plan, fftsize, CUFFT_C2C, nb_symbols*nb_tx) );



}
