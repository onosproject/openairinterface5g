/*! \file PHY\CUDA/LTE_TRANSPORT/turbo_rx_gpu.h
* \brief turbo decoder using gpu 
* \author TerngYin Hsu, JianYa Chu
* \date 2018
* \version 0.1
* \company ISIP LAB/NCTU CS  
* \email: tyhsu@cs.nctu.edu.tw
* \note
* \warning
*/


#ifndef __TURBO_RX_GPU__H__
#define __TURBO_RX_GPU__H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define CRC24_A 0
#define CRC24_B 1
#define CRC16 2
#define CRC8 3

typedef char Binary;
typedef float llr_t;
typedef short channel_t;

#ifdef __cplusplus
extern "C"
#endif
unsigned char phy_threegpplte_turbo_decoder_gpu(short **y,
        unsigned char **decoded_bytes,
		unsigned int codeword_num,
        unsigned short n,
        unsigned short f1,
        unsigned short f2,
        unsigned char max_iterations,
        unsigned char crc_type,
        unsigned char* f_tmp,
		unsigned char* ret);

#ifdef __cplusplus
extern "C"
#endif
void free_ptr(void);

#ifdef __cplusplus
extern "C"
#endif
void init_alloc(void);

#endif
