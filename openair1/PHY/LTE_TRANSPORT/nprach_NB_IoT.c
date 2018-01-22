/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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
/*! \file PHY/LTE_TRANSPORT/nprach_eNb_NB_IoT.c
 * function for NPRACH signal detection and Timing Advance estimation
 * \author V. Savaux
 * \date 2017
 * \version 0.1
 * \company b<>com
 * \email: vincent.savaux@b-com.com
 * \note
 * \warning
 */

//#include "PHY/sse_intrin.h"
#include "PHY/defs_NB_IoT.h"
#include "PHY/TOOLS/defs.h" // to take into account the dft functions
//#include "PHY/extern.h"
//#include "prach.h"
//#include "PHY/LTE_TRANSPORT/if4_tools.h"
//#include "SCHED/defs.h"
//#include "SCHED/extern.h"
//#include "UTIL/LOG/vcd_signal_dumper.h"

uint8_t NPRACH_detection_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB, 
								int16_t *Rx_sub_sampled_buffer, 
								uint16_t sub_sampling_rate, 
								uint32_t FRAME_LENGTH_COMPLEX_SUB_SAMPLES){

	//  uint32_t P_noise ; // to uncomment when needed // needs to be defined or calculated
	uint16_t FFT_size=0; 
	uint16_t delta_t=0; // size, in samples, between 2 successive FFTs
	uint16_t Nb_packets,n_subcarriers; 
	uint16_t N_sample_per_sc=0; // number of samples per subcarrier
	int16_t **mat_from_buffer,**mat_to_detector; 
	uint32_t **mat_energy;  
	uint64_t energy_per_subcarrier;  
	uint32_t threshold_gamma=0; // threshold for signal detection
	int k,n,m; 
	uint8_t is_NPRACH_present = 0; 

	switch (sub_sampling_rate){
		case 16: // fs = 1.92 MHz
			FFT_size = 16384; 
			//dft_size = dft16384; 
			delta_t = 80; 
			N_sample_per_sc = 256; 
		break; 
		case 32: // fs = 960 kHz 
			FFT_size = 8192; 
			//dft_size = dft8192; 
			delta_t = 40; 
			N_sample_per_sc = 128;  
		break; 
		case 64: // fs = 480 kHz 
			FFT_size = 4096; 
			//dft_size = dft4096; 
			delta_t = 20; 
			N_sample_per_sc = 64; 
		break;
		case 128: // fs = 240 kHz 
			FFT_size = 2048; 
			//dft_size = dft2048; 
			delta_t = 10; 
			N_sample_per_sc = 32; 
		break; 
	}

	Nb_packets = (uint16_t)(FRAME_LENGTH_COMPLEX_SUB_SAMPLES-FFT_size)/delta_t + 1; // Number of sections of FFT_size complex samples
	n_subcarriers = FFT_size/N_sample_per_sc; // number of subcarriers on which the energy is calculated

	// Create matrices where lines correspond to one sample, columns are fft of inputs
	mat_to_detector = (int16_t **)malloc(Nb_packets*sizeof(int16_t *)); 
	mat_from_buffer = (int16_t **)malloc(Nb_packets*sizeof(int16_t *)); 
	mat_energy = (uint32_t **)malloc(Nb_packets*sizeof(uint32_t *)); 
	 
	for (k=0;k<Nb_packets;k++){
		mat_to_detector[k] = (int16_t *)malloc(2*FFT_size*sizeof(int16_t)); 
		mat_from_buffer[k] = (int16_t *)malloc(2*FFT_size*sizeof(int16_t)); 
		mat_energy[k] = (uint32_t *)malloc(n_subcarriers*sizeof(uint32_t)); 
	}
	for (k=0;k<Nb_packets;k++){
		for (n=0;n<FFT_size;n++){
			mat_from_buffer[k][2*n] = Rx_sub_sampled_buffer[k*delta_t+2*n]; // Real part
			mat_from_buffer[k][2*n+1] = Rx_sub_sampled_buffer[k*delta_t+2*n+1]; // Imag part
		}
		// dft_size(int16_t mat_from_buffer[k],int16_t mat_to_detector[k],int scale=1); 	
		switch (sub_sampling_rate){
			case 16: // fs = 1.92 MHz
				printf("dft16384 not yet implemented\n");
				//dft16384(int16_t mat_from_buffer[k],int16_t mat_to_detector[k],int scale=1); 
			break; 
			case 32: // fs = 960 kHz 
				dft8192( mat_from_buffer[k], mat_to_detector[k],1);   
			break; 
			case 64: // fs = 480 kHz 
				dft4096( mat_from_buffer[k], mat_to_detector[k],1);  
			break;
			case 128: // fs = 240 kHz 
				dft2048( mat_from_buffer[k], mat_to_detector[k],1);
			break; 
		}
	}

	// Calculate the energy of the samples of mat_to_detector 
	// and concatenate by N_sample_per_sc samples 

	for (k=0;k<Nb_packets;k++){
		for (n=0;n<n_subcarriers;n++){
			energy_per_subcarrier = 0; 
			for (m=0;m<N_sample_per_sc;m++){
				energy_per_subcarrier += (uint64_t)mat_to_detector[k][2*n*N_sample_per_sc+2*m]*mat_to_detector[k][2*n*N_sample_per_sc+2*m] 
										+ (uint64_t)mat_to_detector[k][2*n*N_sample_per_sc+2*m+1]*mat_to_detector[k][2*n*N_sample_per_sc+2*m+1];
			}	
			mat_energy[k][n] = energy_per_subcarrier/N_sample_per_sc; 

			if (energy_per_subcarrier >= threshold_gamma){
				is_NPRACH_present = 1; 
			}
		}		
	} 

	for (k=0;k<Nb_packets;k++){
		free(mat_to_detector[k]); 
		free(mat_from_buffer[k]); 
		free(mat_energy[k]);
	}
	free(mat_to_detector); 
	free(mat_from_buffer); 
	free(mat_energy); 

	return is_NPRACH_present; 

}

uint32_t TA_estimation_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB, 
							  int16_t *Rx_sub_sampled_buffer, 
							  uint16_t sub_sampling_rate, 
							  uint16_t FRAME_LENGTH_COMPLEX_SUB_SAMPLES, 
							  uint32_t estimated_TA_coarse, 
							  char coarse){

	uint16_t length_seq_NPRACH,length_CP,length_symbol; // in number of samples, per NPRACH preamble: 4 sequences ; length of CP in number of samples 
	uint16_t length_CP_0 = eNB->frame_parms.nprach_config_common.nprach_CP_Length; //NB-IoT: 0: short, 1: long 
	uint32_t fs=0; //NB-IoT: sampling frequency of Rx_buffer, must be defined somewhere
	uint32_t fs_sub_sampled; 
	uint16_t length_correl_window,base_length; 
	int64_t *vec_correlation; 
	double max_correlation = 0; 
	int16_t **matrix_received_signal_re, **matrix_received_signal_im; 
	uint16_t offset_estimation, offset_start; // offset due to first coarse estimation
	// double *** mat_to_phase_estimation_re, *** mat_to_phase_estimation_im; 
	double average_mat_to_phase_re, average_mat_to_phase_im; 
	double estimated_phase, estimated_CFO; 
	// int16_t *vec_CFO_compensation_re, *vec_CFO_compensation_im; 
	// int16_t *vec_received_signal_re, *vec_received_signal_im; 
	int32_t *signal_CFO_compensed_re, *signal_CFO_compensed_im; 
	int32_t **sub_sequence_reference_re, **sub_sequence_reference_im; 
	int32_t *sequence_reference_re, *sequence_reference_im; 
	uint32_t TA_sample_estimated = 0; 
	int n,k,m,o; 

	length_seq_NPRACH = (length_CP_0+5*8192)/sub_sampling_rate; 
	length_CP = length_CP_0/sub_sampling_rate; 

	length_symbol = 8192/sub_sampling_rate;  

	if (coarse){ // coarse = 1: first estimation 

		offset_start = 0; 
		length_correl_window = 20512/sub_sampling_rate; // corresponds to the max TA, i.e. 667.66 micro s //FRAME_LENGTH_COMPLEX_SUB_SAMPLES - 4*length_seq_NPRACH+1; 

	}else{

		offset_estimation = 8 * estimated_TA_coarse; 
		base_length = 32; 
		// we arbitrarily define the length of correl window as base_length samples. 
		// Check if offset_estimation is close to zero or 1282 (max lentgh of delays) 

		if (offset_estimation-base_length/2 <0){
			offset_start = 0; 
			length_correl_window = offset_estimation + base_length/2; 
		}
		if (offset_estimation+base_length/2 >1281){
			offset_start = offset_estimation-base_length/2; 
			length_correl_window = base_length;// 512 - (1282-offset_estimation); 
		}
		if ((offset_estimation-base_length/2 >=0) && (offset_estimation+base_length/2 <=1281)){
			offset_start = offset_estimation-base_length/2; 
			length_correl_window = base_length; 
		}
		
	}

	fs_sub_sampled = (uint32_t)fs/sub_sampling_rate; 

	// Method: MMSE (sub-optimal) CFO estimation -> CFO compensation -> ML (sub-optimal) TA estimation /============================================================/

		matrix_received_signal_re = (int16_t **)malloc(4*sizeof(int16_t *)); 
		matrix_received_signal_im = (int16_t **)malloc(4*sizeof(int16_t *)); 
		for (k=0;k<4;k++){ // # sequence
			matrix_received_signal_re[k] = (int16_t *)malloc((length_seq_NPRACH-length_CP)*sizeof(int16_t)); // avoid CP in this process
			matrix_received_signal_im[k] = (int16_t *)malloc((length_seq_NPRACH-length_CP)*sizeof(int16_t)); // avoid CP in this process
		}
		signal_CFO_compensed_re = (int32_t *)malloc(4*length_seq_NPRACH*sizeof(int32_t)); 
		signal_CFO_compensed_im = (int32_t *)malloc(4*length_seq_NPRACH*sizeof(int32_t)); 
		sub_sequence_reference_re = (int32_t **)malloc(4*sizeof(int32_t *));  
		sub_sequence_reference_im = (int32_t **)malloc(4*sizeof(int32_t *)); 
		for (k=0;k<4;k++){
			sub_sequence_reference_re[k] = (int32_t *)calloc(length_symbol,sizeof(int32_t)); 
			sub_sequence_reference_im[k] = (int32_t *)calloc(length_symbol,sizeof(int32_t)); 
		} 
		sequence_reference_re = (int32_t *)malloc(4*length_seq_NPRACH*sizeof(int32_t)); 
		sequence_reference_im = (int32_t *)malloc(4*length_seq_NPRACH*sizeof(int32_t)); 		
		vec_correlation = (int64_t *)malloc(length_correl_window*sizeof(int64_t));  

	for (n=0;n<length_correl_window;n++){ // loops over samples %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		// MMSE (sub-optimal) CFO estimation /============================================================/ 

		// for (k=0;k<4;k++){ // # sequence
		// 	for (m=0;m<(length_seq_NPRACH-length_CP);m++){
		// 		matrix_received_signal_re[k][m] = Rx_sub_sampled_buffer[2*(n+k*length_seq_NPRACH+length_CP+m)]; // avoid CP for CFO estimation
		// 		matrix_received_signal_im[k][m] = Rx_sub_sampled_buffer[2*(n+k*length_seq_NPRACH+length_CP+m)+1]; 
		// 	} 
		// } 
		average_mat_to_phase_re = 0; 
		average_mat_to_phase_im = 0; 
		for (k=0;k<4;k++){ // # sequence
			for (o=0;o<4;o++){ // # symbol in sequence
				for (m=0;m<length_symbol;m++){
					// average_mat_to_phase_re = average_mat_to_phase_re 
					// 						- (double)matrix_received_signal_re[k][o*length_symbol+m] * matrix_received_signal_re[k][(o+1)*length_symbol+m]
					// 						- (double)matrix_received_signal_im[k][o*length_symbol+m] * matrix_received_signal_im[k][(o+1)*length_symbol+m]; 
					// average_mat_to_phase_im = average_mat_to_phase_im
					// 						-	(double)matrix_received_signal_im[k][o*length_symbol+m]	* matrix_received_signal_re[k][(o+1)*length_symbol+m] 
					// 						+ (double)matrix_received_signal_re[k][o*length_symbol+m] * matrix_received_signal_im[k][(o+1)*length_symbol+m]; 

					average_mat_to_phase_re = average_mat_to_phase_re 
												- (double)Rx_sub_sampled_buffer[2*(n+offset_start+k*length_seq_NPRACH+length_CP+o*length_symbol+m)]
												* Rx_sub_sampled_buffer[2*(n+offset_start+k*length_seq_NPRACH+length_CP+(o+1)*length_symbol+m)]
												- (double)Rx_sub_sampled_buffer[2*(n+offset_start+k*length_seq_NPRACH+length_CP+o*length_symbol+m)+1]
												* Rx_sub_sampled_buffer[2*(n+offset_start+k*length_seq_NPRACH+length_CP+(o+1)*length_symbol+m)+1];
												// - (double)matrix_received_signal_re[k][o*length_symbol+m] * matrix_received_signal_re[k][(o+1)*length_symbol+m]
												// - (double)matrix_received_signal_im[k][o*length_symbol+m] * matrix_received_signal_im[k][(o+1)*length_symbol+m]; 
					average_mat_to_phase_im = average_mat_to_phase_im
												- (double)Rx_sub_sampled_buffer[2*(n+offset_start+k*length_seq_NPRACH+length_CP+o*length_symbol+m)+1]
												* Rx_sub_sampled_buffer[2*(n+offset_start+k*length_seq_NPRACH+length_CP+(o+1)*length_symbol+m)]
												+ (double)Rx_sub_sampled_buffer[2*(n+offset_start+k*length_seq_NPRACH+length_CP+o*length_symbol+m)]
												* Rx_sub_sampled_buffer[2*(n+offset_start+k*length_seq_NPRACH+length_CP+(o+1)*length_symbol+m)+1];
				}
			} 
		} 

		average_mat_to_phase_re = average_mat_to_phase_re/(16*length_symbol); 
		average_mat_to_phase_im = average_mat_to_phase_im/(16*length_symbol); 
		estimated_phase = atan2(average_mat_to_phase_im,average_mat_to_phase_re); 
		estimated_CFO = ((double)fs*estimated_phase)/(8192*2*M_PI); 

		// CFO compensation /============================================================/ 

		for (k=0;k<4*length_seq_NPRACH;k++){
			signal_CFO_compensed_re[k] = Rx_sub_sampled_buffer[2*(n+k)] * (int16_t)((double)cos(2*M_PI*estimated_CFO*k/fs_sub_sampled)*32767) 
										- Rx_sub_sampled_buffer[2*(n+k)+1] * (int16_t)((double)sin(2*M_PI*estimated_CFO*k/fs_sub_sampled)*32767); 
			signal_CFO_compensed_im[k] = Rx_sub_sampled_buffer[2*(n+k)] * (int16_t)((double)sin(2*M_PI*estimated_CFO*k/fs_sub_sampled)*32767) 
										+ Rx_sub_sampled_buffer[2*(n+k)+1] * (int16_t)((double)cos(2*M_PI*estimated_CFO*k/fs_sub_sampled)*32767); 
		} 

		// sub-optimal ML TA estimation /============================================================/ 
 
		for (k=0;k<4;k++){ // loop over the 4 sequence of a preamble 
			for (o=0;o<5;o++){ // loop over the symbols of a sequence 
				for (m=0;m<length_symbol;m++){
					sub_sequence_reference_re[k][m] = sub_sequence_reference_re[k][m] + (int32_t)pow(-1,o) * signal_CFO_compensed_re[k*length_seq_NPRACH + o*length_symbol + length_CP + m] / 5; // average over the 5 symbols of a group
					sub_sequence_reference_im[k][m] = sub_sequence_reference_im[k][m] + (int32_t)pow(-1,o) * signal_CFO_compensed_im [k*length_seq_NPRACH + o*length_symbol + length_CP + m]/ 5; // average over the 5 symbols of a group
				}
			}
		} 
		for (k=0;k<4;k++){ // re-initialize sub_sequence_reference matrices
			for (m=0;m<length_symbol;m++){ 
				sub_sequence_reference_re[k][m] = 0; 
				sub_sequence_reference_im[k][m] = 0; 
			}
		} 

		for (m=0;m<length_seq_NPRACH;m++){
			vec_correlation[n] = vec_correlation[n] + (double)signal_CFO_compensed_re[m] * sequence_reference_re[m] + (double)signal_CFO_compensed_im[m] * sequence_reference_im[m];  
		}

	} 
	for (n=0;n<length_correl_window;n++){ 
		if(vec_correlation[n]>=max_correlation){ 
			max_correlation = vec_correlation[n]; 
			TA_sample_estimated = n; 
		}
	}

	free(vec_correlation); 
	for (k=0;k<4;k++){ // # sequence
		free(matrix_received_signal_re[k]); 
		free(matrix_received_signal_im[k]); 
		free(sub_sequence_reference_re[k]); 
		free(sub_sequence_reference_im[k]); 
	} 
	free(matrix_received_signal_re); 
	free(matrix_received_signal_im); 
	free(signal_CFO_compensed_re); 
	free(signal_CFO_compensed_im); 
	free(sub_sequence_reference_re); 
	free(sub_sequence_reference_im); 

	return TA_sample_estimated; 

} 

int16_t* sub_sampling_NB_IoT(int16_t *input_buffer, uint32_t length_input, uint32_t *length_ouput, uint16_t sub_sampling_rate){

	int k; 
	uint32_t L; 
	int16_t *output_buffer; 

	L = (uint32_t)((double)length_input / sub_sampling_rate); 
	*length_ouput = L; 

	output_buffer = (int16_t *)malloc(2*L*sizeof(int16_t)); 

	for (k=0;k<L;k++){
		output_buffer[2*k] = input_buffer[sub_sampling_rate*(2*k)]; 
		output_buffer[2*k+1] = input_buffer[sub_sampling_rate*(2*k)+1]; 
	} 
	// for (k=0;k<2*L;k++){
	// 	 printf("%i\n",output_buffer[k]); 
	// }

	return output_buffer;

}

void RX_NPRACH_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB, int16_t *Rx_buffer){ 

	uint32_t estimated_TA =0;
	uint32_t estimated_TA_coarse=0;  
	int16_t *Rx_sub_sampled_buffer_128,*Rx_sub_sampled_buffer_16; 
	uint16_t sub_sampling_rate; //NB-IoT: to be defined somewhere
	uint32_t FRAME_LENGTH_COMPLEX_SAMPLESx=0; // NB-IoT: length of input buffer, to be defined somewhere 
	uint32_t FRAME_LENGTH_COMPLEX_SUB_SAMPLES; // Length of buffer after sub-sampling
	uint32_t *length_ouput; // Length of buffer after sub-sampling 
	char coarse=1; // flag that indicate the level of TA estimation

	/* 1. Coarse TA estimation using sub sampling rate = 128, i.e. fs = 240 kHz  */

	// Sub-sampling stage /============================================================/ 

	sub_sampling_rate = 128; 
	length_ouput = &FRAME_LENGTH_COMPLEX_SUB_SAMPLES; 
	Rx_sub_sampled_buffer_128 = sub_sampling_NB_IoT(Rx_buffer,FRAME_LENGTH_COMPLEX_SAMPLESx,length_ouput, sub_sampling_rate); 

	// Detection and TA estimation stage  /============================================================/ 

	if (NPRACH_detection_NB_IoT(eNB, Rx_sub_sampled_buffer_128, sub_sampling_rate,FRAME_LENGTH_COMPLEX_SUB_SAMPLES)){

		estimated_TA_coarse = TA_estimation_NB_IoT(eNB, 
												   Rx_sub_sampled_buffer_128, 
												   sub_sampling_rate, 
												   FRAME_LENGTH_COMPLEX_SUB_SAMPLES, 
												   estimated_TA_coarse, 
												   coarse); 

		/* 2. Fine TA estimation using sub sampling rate = 16, i.e. fs = 240 MHz */ 
	
		// Sub-sampling stage /============================================================/
		sub_sampling_rate = 16; 
		Rx_sub_sampled_buffer_16 = sub_sampling_NB_IoT(Rx_buffer,FRAME_LENGTH_COMPLEX_SAMPLESx,length_ouput, sub_sampling_rate); 


		// Fine TA estimation stage  /============================================================/ 
		// start1 = clock();
		coarse = 0;
		estimated_TA = TA_estimation_NB_IoT(eNB, 
											Rx_sub_sampled_buffer_16, 
											sub_sampling_rate, 
											FRAME_LENGTH_COMPLEX_SUB_SAMPLES, 
											estimated_TA_coarse, 
											coarse); 
		// Needs to be stored in a variable in PHY_VARS_eNB_NB_IoT structure
	}


}

