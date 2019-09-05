/* ============================================================================
 *
 *    file             ldpc_decoder_offload.h
 *    author           B.ROBERT
 *    date             Jan 10, 2018
 *
 *    brief            LDPC Decoder Offload interface header.
 *
 * Infos:
 *   - Project         : syr_ldpc_offload_lib
 *   - Software        :
 *   - CVS domain      : syrtemplatform5g
 *   - CVS component   :
 *
 * ============================================================================
 *   Statement of Ownership
 *   ----------------------
 * Copyright (c) 2018-2019 SYRTEM S.a.r.l All Rights Reserved
 *
 * This software is furnished under licence and may be used and copied only
 * in accordance with the terms of such license and with the inclusion of the
 * above COPYRIGHT notice. This SOFTWARE or any other copies thereof may not
 * be provided or otherwise made available to any other person. No title to
 * and ownership of the SOFTWARE is hereby transferred.
 *
 * The information in this SOFTWARE is subject to change without notice and
 * should not be constructed as a commitment by SYRTEM.
 *
 * SYRTEM assumes no responsability for the use or reliability of its SOFTWARE
 * on equipment or platform not explicitly validated by SYRTEM.
 *
 * ============================================================================
 * Reference documents :
 * -------------------
 *
 * ==========================================================================*/
#ifndef __LDPC_DECODER_OFFLOAD_LIB_H__
#define __LDPC_DECODER_OFFLOAD_LIB_H__



#define LDPC_DECODER_OFFLOAD_OK						(0)
#define LDPC_DECODER_OFFLOAD_ERROR					(-1)
#define LDPC_DECODER_OFFLOAD_BAD_PARAMETER			(-2)
#define LDPC_DECODER_OFFLOAD_PRECONDITION_NOT_MET	(-3)
#define LDPC_DECODER_OFFLOAD_NOT_SUPPORTED			(-4)
#define LDPC_DECODER_OFFLOAD_NO_CONTEXT				(-5)
#define LDPC_DECODER_OFFLOAD_DEVICE_NOT_FOUND		(-6)
#define LDPC_DECODER_OFFLOAD_CHANNEL_NOT_FOUND		(-7)
#define LDPC_DECODER_OFFLOAD_BUFFER_NOT_CREATED		(-8)
#define LDPC_DECODER_OFFLOAD_INITCFG_FAILED			(-9)
#define LDPC_DECODER_OFFLOAD_SELF_TEST_FAILED		(-10)
#define LDPC_DECODER_OFFLOAD_SYRIQ_INIT_FAILED		(-11)
#define LDPC_DECODER_OFFLOAD_TH_INIT_FAILED			(-12)
#define LDPC_DECODER_OFFLOAD_DECOD_REQ_NULL			(-20)
#define LDPC_DECODER_OFFLOAD_ERROR_TIMEOUT 			(-30)
 

typedef void	*session_t;

typedef struct session_desc_s
{
	uint32_t	proc_nb;
	uint32_t	frame;
	uint32_t	nr_tti_rx;			// ~ tti number
	uint32_t	BG;
	uint32_t	R;

	uint8_t		coderate;			// 0x01:BG1 8/9 27.Zc
									// 0x2A:BG1 1/3 68.Zc, 
									// 0x43:BG2 2/3 17.Zc, 
									// 0x66:BG2 1/5 52.Zc, 
	uint32_t	Zc;					// 128, 160, 224, 256, 384
	uint8_t		max_decoding_iterations;
	uint32_t	C;
	uint8_t 	crc_type;
	uint8_t		mcs;
} session_desc_t;


typedef enum ldpc_msgtype_e
{
	LDPC_DECODING_OFFLOAD_REQ,
	LDPC_DECODING_OFFLOAD_CONF
} ldpc_msgtype_t;

typedef struct decode_req_s
{
	ldpc_msgtype_t	msg_type;
	uint32_t		proc_nb;
	uint32_t		r;			// seg. index from 1 to 16 max
	int16_t			*data;		// 16bit LLR buffer pointer
} decode_req_t;

typedef struct decode_conf_s
{
	ldpc_msgtype_t	msg_type;
	uint32_t		proc_nb;
	uint32_t		segment_no;		// from 1 to 16 max
	uint32_t		crc24_check; 
	uint32_t 		nb_iterations;
	uint8_t			*data;			// Hard decoded bits buffer pointer
} decode_conf_t;



session_t threegpp_nr_ldpc_decode_start(session_desc_t *session_desc);

int32_t threegpp_nr_ldpc_decode_putq(	session_t 	fd, 
										int16_t 	*y_16bits,
										uint8_t 	*decoded_bytes,
										uint8_t		r,
										uint16_t 	n,
										uint8_t		max_iterations,
										uint8_t		crc_type
									);


int32_t threegpp_nr_ldpc_decode_getq(	session_t 	fd, 
										uint8_t 	*decoded_bytes,
										uint8_t		r,
										uint8_t		crc_type,
										uint8_t		*crc_status,
										uint8_t		*nb_iterations
									); 

int32_t threegpp_nr_ldpc_decode_stop(session_t fd);


int32_t threegpp_nr_ldpc_decode_run(session_t 	fd);



/*!
\brief This routine performs max-logmap detection for the 3GPP turbo code (with termination).  It is optimized for SIMD processing and 16-bit
LLR arithmetic, and requires SSE2,SSSE3 and SSE4.1 (gcc >=4.3 and appropriate CPU)
@param y LLR input (16-bit precision)
@param decoded_bytes Pointer to decoded output
@param n number of coded bits (including tail bits)
@param max_iterations The maximum number of iterations to perform
@param interleaver_f1 F1 generator
@param interleaver_f2 F2 generator
@param crc_type Length of 3GPPLTE crc (CRC24a,CRC24b,CRC16,CRC8)
@param F Number of filler bits at start of packet
@returns number of iterations used (this is 1+max if incorrect crc or if crc_len=0)
*/
uint8_t phy_threegppnr_ldpc_decoder_offload(int16_t		*y,
										uint8_t			*decoded_bytes,
										uint16_t		n,
										uint16_t		interleaver_f1,
										uint16_t		interleaver_f2,
										uint8_t			max_iterations,
										uint8_t			crc_type,
										uint8_t			F,
										time_stats_t	*init_stats,
										time_stats_t	*alpha_stats,
										time_stats_t	*beta_stats,
										time_stats_t	*gamma_stats,
										time_stats_t	*ext_stats,
										time_stats_t	*intl1_stats,
										time_stats_t	*intl2_stats);



uint8_t phy_threegppnr_ldpc_decoder_offload_init(void);

uint8_t phy_threegppnr_ldpc_decoder_offload_fini(void);



#endif // __LDPC_DECODER_OFFLOAD_LIB_H__
