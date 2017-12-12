/*==============================================================================
* nrLDPC_decoder.h
*
* Defines the LDPC decoder core prototypes
*
* Author: Sebastian Wagner
* Date: 17-11-2017
*
===============================================================================*/

#ifndef _NR_LDPC_DECODER_
#define _NR_LDPC_DECODER_

#include "nrLDPC_types.h"

void nrLDPC_decoder(t_nrLDPC_dec_params* p_decParams, int16_t* p_llr, int8_t* p_llrOut, t_nrLDPC_proc_time* p_procTime);

#endif
