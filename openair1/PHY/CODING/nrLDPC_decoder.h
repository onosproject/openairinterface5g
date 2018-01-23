/*==============================================================================
* nrLDPC_decoder.h
*
* Defines the LDPC decoder core prototypes
*
* Author: Sebastian Wagner
* Date: 17-11-2017
*
===============================================================================*/

#ifndef __NR_LDPC_DECODER__H__
#define __NR_LDPC_DECODER__H__

#include "nrLDPC_types.h"

int32_t nrLDPC_decoder(t_nrLDPC_dec_params* p_decParams, int8_t* p_llr, int8_t* p_llrOut, t_nrLDPC_time_stats* p_profiler);

#endif
