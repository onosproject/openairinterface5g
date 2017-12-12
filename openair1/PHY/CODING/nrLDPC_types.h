/*==============================================================================
* nrLDPC_defs.h
*
* Defines all constant variables for the LDPC decoder
*
* Author: Sebastian Wagner
* Date: 17-11-2017
*
===============================================================================*/

#ifndef _NR_LDPC_TYPES_
#define _NR_LDPC_TYPES_

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

typedef struct nrLDPC_dec_params {
    uint8_t BG;
    uint16_t Z;
    uint8_t R; // Format 15,13,... for code rates 1/5, 1/3,...
    uint8_t numMaxIter;
} t_nrLDPC_dec_params;

typedef struct nrLDPC_proc_time {
    double llr2llrProcBuf;
    double llr2CnProcBuf;
    double cnProc;
    double bnProcPc;
    double bnProc;
    double cn2bnProcBuf;
    double bn2cnProcBuf;
    double llrRes2llrOut;
    double llr2bit;
    double total;
} t_nrLDPC_proc_time;

#endif
