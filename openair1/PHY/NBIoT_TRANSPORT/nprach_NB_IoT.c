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
 * \date 2018
 * \version 0.1
 * \company b<>com
 * \email: vincent.savaux@b-com.com
 * \note
 * \warning
 */

//#include "PHY/sse_intrin.h"
#include "PHY/defs_L1_NB_IoT.h"
#include "PHY/TOOLS/tools_defs.h" // to take into account the dft functions
#include "tables_nprach_NB_IoT.h"
#include "first_sc_NB_IoT.h"
//#include "PHY/extern.h"
//#include "prach.h"
//#include "PHY/LTE_TRANSPORT/if4_tools.h"
//#include "SCHED/defs.h"
//#include "SCHED/extern.h"
//#include "UTIL/LOG/vcd_signal_dumper.h"

int filter_xx[40] = {-2161, 453, 489, 570, 688, 838, 1014, 1209, 1420, 1639,
		  1862, 2082, 2295, 2495, 2677, 2837, 2969, 3072, 3142, 3178, 
		  3178, 3142, 3072, 2969, 2837, 2677, 2495, 2295, 2082, 1862, 
		  1639, 1420, 1209, 1014, 838, 688, 570, 489, 453, -2161}; // this is a low-pass filter

int16_t buffer_nprach[153600];
int16_t filtered_buffer[153600];
int16_t signal_compensed_re[153600];
int16_t signal_compensed_im[153600];
int16_t output_buffer[4800];

uint8_t NPRACH_detection_NB_IoT(int16_t *input_buffer,uint32_t input_length){

	uint8_t cp_type = 0; // 0: short ; 1: extended
	uint32_t nb_signal_samples,nb_noise_samples,n1,n2; 
	uint64_t energy_signal=0,energy_noise=0; 
	uint32_t n;

	if(cp_type){

	}else{
		nb_signal_samples = (uint32_t)(((uint64_t) 62670*input_length)/100000); 
		nb_noise_samples = input_length - nb_signal_samples;
	}
	n1 = nb_signal_samples; 
	n2 = nb_noise_samples;

	// printf("n samples = %i %i\n",FRAME_LENGTH_COMPLEX_SAMPLESx,nb_signal_samples); 

	for(n=0;n<nb_signal_samples;n++){
		energy_signal += (((uint64_t)input_buffer[2*n]*input_buffer[2*n] + (uint64_t)input_buffer[2*n+1]*input_buffer[2*n+1])/n1); 
		//energy_signal += (uint64_t)(((uint64_t)input_buffer[2*n]*input_buffer[2*n] + (uint64_t)input_buffer[2*n+1]*input_buffer[2*n+1])/10);
	}
	for(n=nb_signal_samples;n<input_length;n++){
		energy_noise += (((uint64_t)input_buffer[2*n]*input_buffer[2*n] + (uint64_t)input_buffer[2*n+1]*input_buffer[2*n+1])/n2); 
		//energy_noise += (uint64_t)(((uint64_t)input_buffer[2*n]*input_buffer[2*n] + (uint64_t)input_buffer[2*n+1]*input_buffer[2*n+1])/10); 
	}

	 //printf("energies = %ld %ld\n",energy_signal,energy_noise);
	if ((uint64_t)(((uint64_t) energy_signal))<(uint64_t)energy_noise>>6){
			
		return 1;
	}else{
		return 0;
	}
}

/*uint32_t TA_estimation_NB_IoT(PHY_VARS_eNB *eNB, 
							  int16_t *Rx_sub_sampled_buffer, 
							  uint16_t sub_sampling_rate, 
							  uint16_t FRAME_LENGTH_COMPLEX_SUB_SAMPLES, 
							  uint32_t estimated_TA_coarse, 
							  uint8_t coarse){

	uint16_t length_seq_NPRACH,length_CP,length_symbol; // in number of samples, per NPRACH preamble: 4 sequences ; length of CP in number of samples 
	uint16_t length_CP_0 = 2048;//eNB->frame_parms.prach_config_common.nprach_CP_Length; //NB-IoT: 0: short, 1: long 
	uint32_t fs=30720000; //NB-IoT: sampling frequency of Rx_buffer, must be defined somewhere
	uint32_t fs_sub_sampled; 
	uint16_t length_correl_window,base_length; 
	int64_t *vec_correlation; 
	int64_t max_correlation = 0; 
	//int16_t **matrix_received_signal_re, **matrix_received_signal_im; 
	uint16_t offset_estimation, offset_start; // offset due to first coarse estimation
	// double *** mat_to_phase_estimation_re, *** mat_to_phase_estimation_im; 
	int64_t average_mat_to_phase_re, average_mat_to_phase_im; 
	float estimated_phase, estimated_CFO, CFO_correction, CFO_correction_k; 
	// int16_t *vec_CFO_compensation_re, *vec_CFO_compensation_im; 
	// int16_t *vec_received_signal_re, *vec_received_signal_im; 
	int16_t *signal_CFO_compensed_re, *signal_CFO_compensed_im; 
	int32_t **sub_sequence_reference_re, **sub_sequence_reference_im; 
	int32_t *sequence_reference_re, *sequence_reference_im; 
	uint32_t TA_sample_estimated = 0; 
	int32_t A;//,B; 
	int n,k,m,o; 
	int32_t pow_n1 = 1; 
	uint32_t index_av_ph1, index_av_ph2; 

	if (coarse){ // coarse = 1: first estimation at 240 kHz

		length_seq_NPRACH = (length_CP_0+5*8192)/128; 
		length_CP = length_CP_0/128; 
		length_symbol = 64;
		offset_start = 0; 
		length_correl_window = 80; //20512/sub_sampling_rate; // corresponds to the max TA, i.e. 667.66 micro s //FRAME_LENGTH_COMPLEX_SUB_SAMPLES - 4*length_seq_NPRACH+1; 
		fs_sub_sampled = (uint32_t)fs/128; 

	}else{

		length_seq_NPRACH = (length_CP_0+5*8192)/16; 
		length_CP = length_CP_0/16; 
		length_symbol = 8192/16;  

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
err
		fs_sub_sampled = (uint32_t)fs/16;
		
	}

	//fs_sub_sampled = (uint32_t)fs/sub_sampling_rate; 

	// Method: MMSE (sub-optimal) CFO estimation -> CFO compensation -> ML (sub-optimal) TA estimation /============================================================/

		//matrix_received_signal_re = (int16_t **)malloc(4*sizeof(int16_t *)); 
		//matrix_received_signal_im = (int16_t **)malloc(4*sizeof(int16_t *)); 
		// for (k=0;k<4;k++){ // # sequence
		// 	matrix_received_signal_re[k] = (int16_t *)malloc((length_seq_NPRACH-length_CP)*sizeof(int16_t)); // avoid CP in this process
		// 	matrix_received_signal_im[k] = (int16_t *)malloc((length_seq_NPRACH-length_CP)*sizeof(int16_t)); // avoid CP in this process
		// }
		signal_CFO_compensed_re = (int16_t *)malloc(4*length_seq_NPRACH*sizeof(int16_t));   /////to do : exact size of tables 
		signal_CFO_compensed_im = (int16_t *)malloc(4*length_seq_NPRACH*sizeof(int16_t)); 
		sub_sequence_reference_re = (int32_t **)malloc(4*sizeof(int32_t *));  
		sub_sequence_reference_im = (int32_t **)malloc(4*sizeof(int32_t *)); 
		for (k=0;k<4;k++){
			sub_sequence_reference_re[k] = (int32_t *)calloc(length_symbol,sizeof(int32_t)); 
			sub_sequence_reference_im[k] = (int32_t *)calloc(length_symbol,sizeof(int32_t)); 
		} 
		sequence_reference_re = (int32_t *)malloc(4*length_seq_NPRACH*sizeof(int32_t)); 
		sequence_reference_im = (int32_t *)malloc(4*length_seq_NPRACH*sizeof(int32_t)); 		
		vec_correlation = (int64_t *)calloc(length_correl_window,sizeof(int64_t));  

	for (n=0;n<length_correl_window;n++){ // loops over samples %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		// MMSE (sub-optimal) CFO estimation /============================================================/ 
		average_mat_to_phase_re = 0; 
		average_mat_to_phase_im = 0; 
		for (k=0;k<4;k++){ // # sequence
			for (o=0;o<4;o++){ // # symbol in sequence
				for (m=0;m<length_symbol;m++){        ///// creation of two variables for tab indexes "n+offset_start+k*length_seq_NPRACH+length_CP+o*length_symbol+m"
					index_av_ph1 = (n+offset_start+k*length_seq_NPRACH+length_CP+o*length_symbol+m)<<1; 
					index_av_ph2 = index_av_ph1 + (length_symbol<<1);
					average_mat_to_phase_re = average_mat_to_phase_re 
												- (int64_t)(Rx_sub_sampled_buffer[index_av_ph1]
												* Rx_sub_sampled_buffer[index_av_ph2])
												- (int64_t)(Rx_sub_sampled_buffer[index_av_ph1+1]
												* Rx_sub_sampled_buffer[index_av_ph2+1]);

					average_mat_to_phase_im = average_mat_to_phase_im
												- (int64_t)(Rx_sub_sampled_buffer[index_av_ph1+1]
												* Rx_sub_sampled_buffer[index_av_ph2])
												+ (int64_t)(Rx_sub_sampled_buffer[index_av_ph1]
												* Rx_sub_sampled_buffer[index_av_ph2+1]);
				}
			} 
		} 

		average_mat_to_phase_re = average_mat_to_phase_re/(16*length_symbol); 
		average_mat_to_phase_im = average_mat_to_phase_im/(16*length_symbol); 
		estimated_phase = atan2f(average_mat_to_phase_im,average_mat_to_phase_re); 
		estimated_CFO = ((float)fs*estimated_phase)/(8192*2*(float)M_PI); 
		CFO_correction = 2*(float)M_PI*estimated_CFO/fs_sub_sampled;
		// CFO compensation /============================================================/ 

		for (k=0;k<4*length_seq_NPRACH;k++){     ///// creation of two variables for tab indexes /// replace "2*(float)M_PI*estimated_CFO*k/fs_sub_sampled" and "2*(n+offset_start+k)"
			CFO_correction_k = (float)k*CFO_correction;
		    signal_CFO_compensed_re[k] = (int16_t)((Rx_sub_sampled_buffer[(n+offset_start+k)<<1] * (int32_t)(cosf(CFO_correction_k)*32767) 
							- Rx_sub_sampled_buffer[((n+offset_start+k)<<1)+1] * (int32_t)(sinf(CFO_correction_k)*32767))>>15);
		    signal_CFO_compensed_im[k] = (int16_t)((Rx_sub_sampled_buffer[(n+offset_start+k)<<1] * (int32_t)(sinf(CFO_correction_k)*32767) 
							+ Rx_sub_sampled_buffer[((n+offset_start+k)<<1)+1] * (int32_t)(cosf(CFO_correction_k)*32767))>>15);
	
		} 

		// sub-optimal ML TA estimation /============================================================/ 
 
		
		for (k=0;k<4;k++){ // loop over the 4 sequences of a preamble 
			pow_n1 = 1;
			for (o=0;o<5;o++){ // loop over the symbols of a sequence 
				for (m=0;m<length_symbol;m++){
                         //  mon_variable=k*length_seq_NPRACH + o*length_symbol + length_CP + m ///////////////////////////////////////////////////////////////////////////////////////////////
				    sub_sequence_reference_re[k][m] = sub_sequence_reference_re[k][m] +  pow_n1 * signal_CFO_compensed_re[k*length_seq_NPRACH + o*length_symbol + length_CP + m] / 5; // average over the 5 symbols of a group
				    sub_sequence_reference_im[k][m] = sub_sequence_reference_im[k][m] +  pow_n1 * signal_CFO_compensed_im [k*length_seq_NPRACH + o*length_symbol + length_CP + m]/ 5; // average over the 5 symbols of a group
				}
				pow_n1 = -pow_n1;
			}
		} 

		pow_n1 = 1;
		for (k=0;k<4;k++){ // loop over the 4 sequences of a preamble 
			pow_n1 = 1;
			for (o=0;o<5;o++){ // loop over the symbols of a sequence   //  mon_variable=k*length_seq_NPRACH+o*length_symbol +length_CP +m///////////////////////////////////////////////
				for (m=0;m<length_symbol;m++){
				    sequence_reference_re[k*length_seq_NPRACH+o*length_symbol +length_CP +m] = pow_n1 * sub_sequence_reference_re[k][m]; 
				    sequence_reference_im[k*length_seq_NPRACH+o*length_symbol +length_CP +m] = pow_n1 * sub_sequence_reference_im[k][m];
				}
				pow_n1 = -pow_n1;
			}
		}
		for (k=0;k<4;k++){ // loop over the 4 sequences of a preamble 
			for (m=0;m<length_CP;m++){
				sequence_reference_re[k*length_seq_NPRACH+m] = -sub_sequence_reference_re[k][length_symbol-length_CP+m]; 
				sequence_reference_im[k*length_seq_NPRACH+m] = -sub_sequence_reference_im[k][length_symbol-length_CP+m]; 
			}
		} 

		// for (m=0;m<length_seq_NPRACH;m++){
		// 	vec_correlation[n] = vec_correlation[n] + (double)signal_CFO_compensed_re[m] * sequence_reference_re[m] + (double)signal_CFO_compensed_im[m] * sequence_reference_im[m];  
		// 	printf("seq=%i\n",sequence_reference_re[m]); 
		// }
		
		for (m=0;m<4*length_seq_NPRACH;m++){
			A = (int64_t)((signal_CFO_compensed_re[m] * sequence_reference_re[m] 
				+ signal_CFO_compensed_im[m] * sequence_reference_im[m])); 
			//B = -(int32_t)(((int64_t)signal_CFO_compensed_re[m] * (int64_t)sequence_reference_im[m] 
			//	- (int64_t)signal_CFO_compensed_im[m] * (int64_t)sequence_reference_re[m])>>32); 
			vec_correlation[n] = vec_correlation[n] + A;//(int32_t)(((int64_t)A*(int64_t)A + 2*(int64_t)B*(int64_t)B)>>32);
		}

		for (k=0;k<4;k++){ // re-initialize sub_sequence_reference matrices   ////////////////////////////////////////////
			for (m=0;m<length_symbol;m++){ 
				sub_sequence_reference_re[k][m] = 0; 
				sub_sequence_reference_im[k][m] = 0; 
			}
		} 

	} 
	for (n=0;n<length_correl_window;n++){ 
		//printf("\ncorr=%li \n",vec_correlation[n]);
		if(vec_correlation[n]>=max_correlation){ 
			max_correlation = vec_correlation[n]; 
			TA_sample_estimated = offset_start+ n; 
		}
	}

	free(vec_correlation);       
	for (k=0;k<4;k++){ // # sequence
		//free(matrix_received_signal_re[k]); 
		err//free(matrix_received_signal_im[k]); 
		free(sub_sequence_reference_re[k]); 
		free(sub_sequence_reference_im[k]); 
	} 
	//free(matrix_received_signal_re); 
	//free(matrix_received_signal_im); 
	free(signal_CFO_compensed_re); 
	free(signal_CFO_compensed_im); 
	free(sub_sequence_reference_re); 
	free(sub_sequence_reference_im); 

	return TA_sample_estimated; 

} */ 


uint16_t subcarrier_estimation(int16_t *input_buffer){
	
	uint16_t estimated_sc=0; 
	int16_t *s_n_re, *s_n_im; 
	uint16_t k,m,n; 
	int64_t max_correl_sc_m = 0; 
	int64_t max_correl_sc_k = 0; 
	int64_t max_correl_sc_glob = 0; 
	int n_start_offset = 1920; // start at t=8 ms

	for (k=0;k<12;k++){
		s_n_re = &s_n_12_re[k*336]; 
		s_n_im = &s_n_12_im[k*336]; 

		for (m=0;m<20;m++){
			for (n=0;n<336;n++){
				max_correl_sc_m = max_correl_sc_m + 
							(int16_t)(((int32_t)input_buffer[(m<<1)+((n+n_start_offset)<<1)]*(int32_t)s_n_re[n] )>>15) 
							+ (int16_t)(((int32_t)input_buffer[(m<<1)+((n+n_start_offset)<<1)+1]*(int32_t)s_n_im[n])>>15);
			}

			if (max_correl_sc_m>max_correl_sc_k){
				max_correl_sc_k = max_correl_sc_m;
			}
			max_correl_sc_m = 0;
		}

		//printf("correl = %li\n",max_correl_sc_k);

		if (max_correl_sc_k>max_correl_sc_glob){
			max_correl_sc_glob = max_correl_sc_k; 
			estimated_sc = k; 
		}
		max_correl_sc_k = 0; 
	} 

	return estimated_sc;

}

int16_t* sub_sampling_NB_IoT(int16_t *input_buffer, uint32_t length_input, uint32_t *length_ouput, uint16_t sub_sampling_rate){  // void function ////// adding flag for switching between output_buffers 

	int k; 
	uint32_t L; 
	//int16_t *output_buffer;    
	int16_t *p_output_buffer;
	L = (uint32_t)(length_input / sub_sampling_rate);  
	*length_ouput = L;    ///// to remove

	

	for (k=0;k<L;k++){
		output_buffer[k<<1] = input_buffer[sub_sampling_rate*(k<<1)]; 
		output_buffer[(k<<1)+1] = input_buffer[sub_sampling_rate*(k<<1)+1]; 
	} 
	// for (k=0;k<2*L;k++){
	// 	 printf("%i\n",output_buffer[k]); 
	// }
p_output_buffer=output_buffer;
	return p_output_buffer;

} 

void filtering_signal(int16_t *input_buffer, int16_t *filtered_buffer, uint32_t FRAME_LENGTH_COMPLEX_SAMPLESx){

	int n;
	//int k; 
	//float f_s_RB22 = 1807.5; 
	//float f_s = 7680; 
	//int16_t *signal_compensed_re, *signal_compensed_im; 
	int16_t *cos_x, *sin_x; 

	cos_x = cos_x_rb22; 
	sin_x = sin_x_rb22; 

	
	for (n=0;n<FRAME_LENGTH_COMPLEX_SAMPLESx;n++){

		signal_compensed_re[n] = (int16_t)((input_buffer[n<<1] * (int32_t)(cos_x[n])      
								+ input_buffer[(n<<1)+1] * (int32_t)(sin_x[n]))>>15); 
		signal_compensed_im[n] = (int16_t)((- input_buffer[n<<1] * (int32_t)(sin_x[n]) 
								+ input_buffer[(n<<1)+1] * (int32_t)(cos_x[n]))>>15); 
		
		filtered_buffer[n<<1] = signal_compensed_re[n]; 
		filtered_buffer[(n<<1)+1] = signal_compensed_im[n]; 


	}

	/*for (n=0;n<FRAME_LENGTH_COMPLEX_SAMPLESx-10;n++){
		if (n<20){
			for (k=-n;k<20;k++){
				filtered_buffer[n<<1] = filtered_buffer[n<<1] + (int16_t)(((int32_t)filter_xx[20+k]*(int32_t)signal_compensed_re[n+k])>>15); 
				filtered_buffer[(n<<1)+1] = filtered_buffer[(n<<1)+1] + (int16_t)(((int32_t)filter_xx[20+k]*(int32_t)signal_compensed_im[n+k])>>15); 
			}
		}else{
			for (k=-20;k<20;k++){
				filtered_buffer[n<<1] = filtered_buffer[n<<1] + (int16_t)(((int32_t)filter_xx[20+k]*(int32_t)signal_compensed_re[n+k])>>15); 
				filtered_buffer[(n<<1)+1] = filtered_buffer[(n<<1)+1] + (int16_t)(((int32_t)filter_xx[20+k]*(int32_t)signal_compensed_im[n+k])>>15); 
			}
		}
	}*/
	

}

uint32_t process_nprach_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB, int frame, uint8_t subframe, uint16_t *rnti, uint16_t *preamble_index, uint16_t *timing_advance){ 


	//uint32_t estimated_TA_coarse=0;  
	//uint32_t estimated_TA;
	int16_t *Rx_sub_sampled_buffer_128; //       *Rx_sub_sampled_buffer_16; 
	uint16_t sub_sampling_rate; //NB-IoT: to be defined somewhere
	uint32_t FRAME_LENGTH_COMPLEX_SAMPLESx; // NB-IoT: length of input buffer, to be defined somewhere 
	uint32_t FRAME_LENGTH_COMPLEX_SUB_SAMPLES; // Length of buffer after sub-sampling
	uint32_t *length_ouput; // Length of buffer after sub-sampling 
	// uint8_t coarse=1; // flag that indicate the level of TA estimation
	int16_t *Rx_buffer;
	//int16_t *filtered_buffer;
	//int n;
 	
	//// 1. Coarse TA estimation using sub sampling rate = 128, i.e. fs = 240 kHz  

	FRAME_LENGTH_COMPLEX_SAMPLESx = 10*eNB->frame_parms.samples_per_tti; 
	Rx_buffer = (int16_t*)&eNB->common_vars.rxdata[0][0][0]; // get the whole frame

        memcpy(&buffer_nprach[0],&Rx_buffer[0],307200);
	
	
	//filtered_buffer = (int16_t *)calloc(2*FRAME_LENGTH_COMPLEX_SAMPLESx,sizeof(int16_t));  // calcule du taille exacte du tableau 76800
        memset(filtered_buffer,0,307200);
	filtering_signal(buffer_nprach,filtered_buffer,FRAME_LENGTH_COMPLEX_SAMPLESx); 

	// Sub-sampling stage /============================================================/ 

	sub_sampling_rate = FRAME_LENGTH_COMPLEX_SAMPLESx/2400; // gives the sub-sampling rate leading to f=240 kHz
	length_ouput = &FRAME_LENGTH_COMPLEX_SUB_SAMPLES; 
	Rx_sub_sampled_buffer_128 = sub_sampling_NB_IoT(filtered_buffer,FRAME_LENGTH_COMPLEX_SAMPLESx,length_ouput, sub_sampling_rate);  

	// Detection and TA estimation stage  /============================================================/ 

	if (NPRACH_detection_NB_IoT(Rx_sub_sampled_buffer_128,*length_ouput)){
		
		
		/*  estimated_TA_coarse = TA_estimation_NB_IoT(eNB, 
													   Rx_sub_sampled_buffer_128, 
													   sub_sampling_rate, 
													   FRAME_LENGTH_COMPLEX_SUB_SAMPLES, 
													   estimated_TA_coarse, 
													   coarse); 


		// 2. Fine TA estimation using sub sampling rate = 16, i.e. fs = 1.92 MHz  
	
		// Sub-sampling stage /============================================================/
		//// sub_sampling_rate = FRAME_LENGTH_COMPLEX_SAMPLESx/(2400*8); 
		Rx_sub_sampled_buffer_16 = sub_sampling_NB_IoT(filtered_buffer,FRAME_LENGTH_COMPLEX_SAMPLESx,length_ouput, sub_sampling_rate); 


		// Fine TA estimation stage  /============================================================/ 
		// start1 = clock();
		coarse = 0;
		estimated_TA = TA_estimation_NB_IoT(eNB, 
											Rx_sub_sampled_buffer_16, 
											sub_sampling_rate, 
											FRAME_LENGTH_COMPLEX_SUB_SAMPLES, 
											estimated_TA_coarse, 
											coarse); //
		// Needs to be stored in a variable in PHY_VARS_eNB_NB_IoT structure

		//for (n=0;n<FRAME_LENGTH_COMPLEX_SAMPLESx;n++){
			//printf("   buf%i= %i",n,Rx_sub_sampled_buffer_128[2*n]);
		//	fprintf(f," %i %i ",Rx_buffer[2*n],Rx_buffer[2*n+1]);
			//fprintf(f,"%i \n",Rx_buffer[2*n+1]);
		//}*/

		printf("\ndetection !!!   at frame %i \n",frame);
		//eNB->preamble_index_NB_IoT = subcarrier_estimation(Rx_sub_sampled_buffer_128);    // c'est un uint16_t
		*preamble_index = subcarrier_estimation(Rx_sub_sampled_buffer_128);
		*timing_advance = 0;
		*rnti = 1 + frame/4;
		printf("estimated subaccier = %i\n",*preamble_index);
		return 1;//estimated_TA;
	}else{

		return 0;
	}

// }
 return 0;
}

