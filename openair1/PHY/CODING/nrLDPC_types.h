/*==============================================================================
* nrLDPC_defs.h
*
* Defines LDPC decoder types
*
* Author: Sebastian Wagner
* Date: 17-11-2017
*
===============================================================================*/

#ifndef __NR_LDPC_TYPES__H__
#define __NR_LDPC_TYPES__H__

#include "./nrLDPC_tools/time_meas.h"

// ==============================================================================
// TYPES

typedef struct nrLDPC_lut {
    const uint32_t* startAddrCnGroups;
    const uint8_t*  numCnInCnGroups;
    const uint8_t*  numBnInBnGroups;
    const uint32_t* startAddrBnGroups;
    const uint16_t* startAddrBnGroupsLlr;
    const uint32_t* llr2CnProcBuf;
    const uint8_t*  numEdgesPerBn;
    const uint32_t* cn2bnProcBuf;
    const uint16_t* llr2llrProcBuf;
} t_nrLDPC_lut;

typedef enum nrLDPC_outMode {
    nrLDPC_outMode_BIT,
    nrLDPC_outMode_BITINT8,
    nrLDPC_outMode_LLRINT8
} e_nrLDPC_outMode;

typedef struct nrLDPC_dec_params {
    uint8_t BG;
    uint16_t Z;
    uint8_t R; // Format 15,13,... for code rates 1/5, 1/3,...
    uint8_t numMaxIter;
    e_nrLDPC_outMode outMode;
} t_nrLDPC_dec_params;

typedef struct nrLDPC_time_stats {
    time_stats_t llr2llrProcBuf;
    time_stats_t llr2CnProcBuf;
    time_stats_t cnProc;
    time_stats_t bnProcPc;
    time_stats_t bnProc;
    time_stats_t cn2bnProcBuf;
    time_stats_t bn2cnProcBuf;
    time_stats_t llrRes2llrOut;
    time_stats_t llr2bit;
    time_stats_t total;
} t_nrLDPC_time_stats;

#endif
