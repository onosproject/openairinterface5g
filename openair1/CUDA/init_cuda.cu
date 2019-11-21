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

cuda_cu_ru cu_ru;

__constant__ int PORTSIZE;
__constant__ int SUBTXSIZE;
__constant__ int BW_PSIZE;

extern "C" void init_cuda(int nb_tx, int nb_symbols, int fftsize){
	printf("init_cuda %d %d %d  \n\n\n", nb_tx, nb_symbols, fftsize);

	int nb_antenna_ports = 8;

	//beamforming precoding
	gpuErrchk( cudaMalloc((void**)&cu_ru.d_txdataF, sizeof(int) * nb_tx*nb_antenna_ports*nb_symbols*fftsize) );
	gpuErrchk( cudaMalloc((void**)&cu_ru.d_weight, sizeof(int) * nb_tx*nb_antenna_ports*fftsize) );
	gpuErrchk( cudaMalloc((void**)&cu_ru.d_subtx, sizeof(int) * nb_tx*fftsize*nb_symbols*2) );

	//ifft	
	gpuErrchk( cudaMalloc((void**)&cu_ru.d_txdataF_BF, fftsize*sizeof(int)*nb_symbols*nb_tx) );
	gpuErrchk( cudaMalloc((void**)&cu_ru.d_signal, fftsize*sizeof(Complex)*nb_symbols*nb_tx) );
	gpuErrchk( cudaMalloc((void**)&cu_ru.d_data_wCP, fftsize*(nb_symbols+1)*nb_tx*sizeof(int)) );
	cufftErrchk( cufftPlan1d(&cu_ru.plan, fftsize, CUFFT_C2C, nb_symbols*nb_tx) );

	int portSize  = fftsize*nb_symbols;
	int subtxsize = nb_tx * nb_symbols * fftsize;
	int bw_psize = nb_tx * fftsize;
	gpuErrchk( cudaMemcpyToSymbol(PORTSIZE, &portSize, sizeof(int)) );
	gpuErrchk( cudaMemcpyToSymbol(SUBTXSIZE, &subtxsize, sizeof(int)) );
	gpuErrchk( cudaMemcpyToSymbol(BW_PSIZE, &bw_psize, sizeof(int)) );

}
