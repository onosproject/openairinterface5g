/*==============================================================================
* nrLDPC_init.h
*
* Defines the function to initialize the LDPC decoder.
* Set correct LUTs
*
* Author: Sebastian Wagner
* Date: 17-11-2017
*
===============================================================================*/

#ifndef __NR_LDPC_INIT__H__
#define __NR_LDPC_INIT__H__

#include "./nrLDPC_lut/nrLDPC_lut.h"
#include "nrLDPC_defs.h"

static inline uint32_t nrLDPC_init(t_nrLDPC_dec_params* p_decParams, t_nrLDPC_lut* p_lut)
{
    uint32_t numLLR = 0;
    uint8_t BG = p_decParams->BG;
    uint16_t Z = p_decParams->Z;
    uint8_t  R = p_decParams->R;

    if (BG == 2)
    {
        // LUT that only depend on BG
        p_lut->startAddrCnGroups = lut_startAddrCnGroups_BG2;

        // LUT that only depend on R
        if (R == 15)
        {
            p_lut->numCnInCnGroups = lut_numCnInCnGroups_BG2_R15;
            p_lut->numBnInBnGroups = lut_numBnInBnGroups_BG2_R15;
            p_lut->startAddrBnGroups = lut_startAddrBnGroups_BG2_R15;
            p_lut->startAddrBnGroupsLlr = lut_startAddrBnGroupsLlr_BG2_R15;
            p_lut->numEdgesPerBn = lut_numEdgesPerBn_BG2_R15;
            numLLR = NR_LDPC_NCOL_BG2_R15*Z;
        }
        else if (R == 13)
        {
            p_lut->numCnInCnGroups = lut_numCnInCnGroups_BG2_R13;
            p_lut->numBnInBnGroups = lut_numBnInBnGroups_BG2_R13;
            p_lut->startAddrBnGroups = lut_startAddrBnGroups_BG2_R13;
            p_lut->startAddrBnGroupsLlr = lut_startAddrBnGroupsLlr_BG2_R13;
            p_lut->numEdgesPerBn = lut_numEdgesPerBn_BG2_R13;
            numLLR = NR_LDPC_NCOL_BG2_R13*Z;
        }
        else if (R == 23)
        {
            p_lut->numCnInCnGroups = lut_numCnInCnGroups_BG2_R23;
            p_lut->numBnInBnGroups = lut_numBnInBnGroups_BG2_R23;
            p_lut->startAddrBnGroups = lut_startAddrBnGroups_BG2_R23;
            p_lut->startAddrBnGroupsLlr = lut_startAddrBnGroupsLlr_BG2_R23;
            p_lut->numEdgesPerBn = lut_numEdgesPerBn_BG2_R23;
            numLLR = NR_LDPC_NCOL_BG2_R23*Z;
        }

        // LUT that depend on Z and R
        if (Z == 2)
        {
            if (R == 15)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z2_R15;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z2_R15;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z2_R15;
            }
            else if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z2_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z2_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z2_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z2_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z2_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z2_R23;
            }

        }
        else if (Z == 80)
        {
            if (R == 15)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z80_R15;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z80_R15;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z80_R15;
            }
            else if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z80_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z80_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z80_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z80_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z80_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z80_R23;
            }

        }
        else if (Z == 128)
        {
            if (R == 15)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z128_R15;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z128_R15;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z128_R15;
            }
            else if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z128_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z128_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z128_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z128_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z128_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z128_R23;
            }

        }
        else if (Z == 384)
        {
            if (R == 15)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z384_R15;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z384_R15;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z384_R15;
            }
            else if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z384_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z384_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z384_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG2_Z384_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG2_Z384_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG2_Z384_R23;
            }
        }
    }
    else
    {   // BG == 1
        // LUT that only depend on BG
        p_lut->startAddrCnGroups = lut_startAddrCnGroups_BG1;

        // LUT that only depend on R
        if (R == 13)
        {
            p_lut->numCnInCnGroups = lut_numCnInCnGroups_BG1_R13;
            p_lut->numBnInBnGroups = lut_numBnInBnGroups_BG1_R13;
            p_lut->startAddrBnGroups = lut_startAddrBnGroups_BG1_R13;
            p_lut->startAddrBnGroupsLlr = lut_startAddrBnGroupsLlr_BG1_R13;
            p_lut->numEdgesPerBn = lut_numEdgesPerBn_BG1_R13;
            numLLR = NR_LDPC_NCOL_BG1_R13*Z;
        }
        else if (R == 23)
        {
            p_lut->numCnInCnGroups = lut_numCnInCnGroups_BG1_R23;
            p_lut->numBnInBnGroups = lut_numBnInBnGroups_BG1_R23;
            p_lut->startAddrBnGroups = lut_startAddrBnGroups_BG1_R23;
            p_lut->startAddrBnGroupsLlr = lut_startAddrBnGroupsLlr_BG1_R23;
            p_lut->numEdgesPerBn = lut_numEdgesPerBn_BG1_R23;
            numLLR = NR_LDPC_NCOL_BG1_R23*Z;
        }
        else if (R == 89)
        {
            p_lut->numCnInCnGroups = lut_numCnInCnGroups_BG1_R89;
            p_lut->numBnInBnGroups = lut_numBnInBnGroups_BG1_R89;
            p_lut->startAddrBnGroups = lut_startAddrBnGroups_BG1_R89;
            p_lut->startAddrBnGroupsLlr = lut_startAddrBnGroupsLlr_BG1_R89;
            p_lut->numEdgesPerBn = lut_numEdgesPerBn_BG1_R89;
            numLLR = NR_LDPC_NCOL_BG1_R89*Z;
        }

        // LUT that depend on Z and R
        if (Z == 2)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z2_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z2_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z2_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z2_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z2_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z2_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z2_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z2_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z2_R89;
            }
        }
        else if (Z == 3)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z3_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z3_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z3_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z3_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z3_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z3_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z3_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z3_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z3_R89;
            }
        }
        else if (Z == 4)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z4_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z4_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z4_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z4_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z4_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z4_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z4_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z4_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z4_R89;
            }
        }
        else if (Z == 5)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z5_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z5_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z5_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z5_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z5_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z5_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z5_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z5_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z5_R89;
            }
        }
        else if (Z == 6)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z6_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z6_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z6_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z6_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z6_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z6_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z6_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z6_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z6_R89;
            }
        }
        else if (Z == 7)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z7_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z7_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z7_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z7_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z7_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z7_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z7_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z7_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z7_R89;
            }
        }
        else if (Z == 8)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z8_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z8_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z8_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z8_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z8_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z8_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z8_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z8_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z8_R89;
            }
        }
        else if (Z == 9)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z9_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z9_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z9_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z9_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z9_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z9_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z9_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z9_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z9_R89;
            }
        }
        else if (Z == 10)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z10_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z10_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z10_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z10_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z10_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z10_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z10_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z10_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z10_R89;
            }
        }
        else if (Z == 11)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z11_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z11_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z11_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z11_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z11_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z11_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z11_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z11_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z11_R89;
            }
        }
        else if (Z == 12)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z12_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z12_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z12_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z12_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z12_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z12_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z12_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z12_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z12_R89;
            }
        }
        else if (Z == 13)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z13_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z13_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z13_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z13_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z13_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z13_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z13_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z13_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z13_R89;
            }
        }
        else if (Z == 14)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z14_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z14_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z14_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z14_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z14_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z14_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z14_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z14_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z14_R89;
            }
        }
        else if (Z == 15)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z15_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z15_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z15_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z15_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z15_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z15_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z15_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z15_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z15_R89;
            }
        }
        else if (Z == 16)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z16_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z16_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z16_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z16_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z16_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z16_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z16_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z16_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z16_R89;
            }
        }
        else if (Z == 18)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z18_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z18_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z18_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z18_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z18_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z18_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z18_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z18_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z18_R89;
            }
        }
        else if (Z == 20)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z20_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z20_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z20_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z20_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z20_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z20_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z20_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z20_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z20_R89;
            }
        }
        else if (Z == 22)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z22_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z22_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z22_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z22_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z22_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z22_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z22_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z22_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z22_R89;
            }
        }
        else if (Z == 24)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z24_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z24_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z24_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z24_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z24_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z24_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z24_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z24_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z24_R89;
            }
        }
        else if (Z == 26)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z26_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z26_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z26_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z26_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z26_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z26_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z26_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z26_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z26_R89;
            }
        }
        else if (Z == 28)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z28_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z28_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z28_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z28_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z28_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z28_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z28_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z28_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z28_R89;
            }
        }
        else if (Z == 30)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z30_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z30_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z30_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z30_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z30_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z30_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z30_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z30_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z30_R89;
            }
        }
        else if (Z == 32)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z32_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z32_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z32_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z32_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z32_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z32_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z32_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z32_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z32_R89;
            }
        }
        else if (Z == 36)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z36_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z36_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z36_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z36_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z36_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z36_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z36_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z36_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z36_R89;
            }
        }
        else if (Z == 40)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z40_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z40_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z40_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z40_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z40_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z40_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z40_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z40_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z40_R89;
            }
        }
        else if (Z == 44)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z44_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z44_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z44_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z44_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z44_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z44_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z44_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z44_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z44_R89;
            }
        }
        else if (Z == 48)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z48_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z48_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z48_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z48_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z48_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z48_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z48_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z48_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z48_R89;
            }
        }
        else if (Z == 52)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z52_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z52_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z52_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z52_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z52_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z52_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z52_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z52_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z52_R89;
            }
        }
        else if (Z == 56)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z56_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z56_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z56_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z56_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z56_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z56_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z56_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z56_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z56_R89;
            }
        }
        else if (Z == 60)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z60_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z60_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z60_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z60_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z60_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z60_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z60_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z60_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z60_R89;
            }
        }
        else if (Z == 64)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z64_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z64_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z64_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z64_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z64_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z64_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z64_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z64_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z64_R89;
            }
        }
        else if (Z == 72)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z72_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z72_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z72_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z72_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z72_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z72_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z72_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z72_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z72_R89;
            }
        }
        else if (Z == 80)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z80_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z80_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z80_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z80_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z80_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z80_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z80_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z80_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z80_R89;
            }
        }
        else if (Z == 88)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z88_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z88_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z88_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z88_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z88_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z88_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z88_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z88_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z88_R89;
            }
        }
        else if (Z == 96)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z96_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z96_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z96_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z96_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z96_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z96_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z96_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z96_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z96_R89;
            }
        }
        else if (Z == 104)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z104_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z104_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z104_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z104_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z104_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z104_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z104_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z104_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z104_R89;
            }
        }
        else if (Z == 112)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z112_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z112_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z112_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z112_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z112_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z112_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z112_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z112_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z112_R89;
            }
        }
        else if (Z == 120)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z120_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z120_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z120_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z120_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z120_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z120_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z120_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z120_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z120_R89;
            }
        }
        else if (Z == 128)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z128_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z128_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z128_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z128_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z128_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z128_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z128_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z128_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z128_R89;
            }
        }
        else if (Z == 144)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z144_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z144_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z144_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z144_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z144_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z144_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z144_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z144_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z144_R89;
            }
        }
        else if (Z == 160)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z160_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z160_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z160_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z160_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z160_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z160_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z160_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z160_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z160_R89;
            }
        }
        else if (Z == 176)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z176_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z176_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z176_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z176_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z176_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z176_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z176_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z176_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z176_R89;
            }
        }
        else if (Z == 192)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z192_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z192_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z192_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z192_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z192_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z192_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z192_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z192_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z192_R89;
            }
        }
        else if (Z == 208)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z208_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z208_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z208_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z208_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z208_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z208_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z208_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z208_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z208_R89;
            }
        }
        else if (Z == 224)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z224_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z224_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z224_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z224_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z224_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z224_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z224_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z224_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z224_R89;
            }
        }
        else if (Z == 240)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z240_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z240_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z240_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z240_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z240_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z240_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z240_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z240_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z240_R89;
            }
        }
        else if (Z == 256)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z256_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z256_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z256_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z256_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z256_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z256_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z256_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z256_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z256_R89;
            }
        }
        else if (Z == 288)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z288_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z288_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z288_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z288_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z288_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z288_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z288_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z288_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z288_R89;
            }
        }
        else if (Z == 320)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z320_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z320_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z320_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z320_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z320_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z320_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z320_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z320_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z320_R89;
            }
        }
        else if (Z == 352)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z352_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z352_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z352_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z352_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z352_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z352_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z352_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z352_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z352_R89;
            }
        }
        else if (Z == 384)
        {
            if (R == 13)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z384_R13;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z384_R13;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z384_R13;
            }
            else if (R == 23)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z384_R23;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z384_R23;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z384_R23;
            }
            else if (R == 89)
            {
                p_lut->llr2CnProcBuf  = lut_llr2CnProcBuf_BG1_Z384_R89;
                p_lut->cn2bnProcBuf   = lut_cn2bnProcBuf_BG1_Z384_R89;
                p_lut->llr2llrProcBuf = lut_llr2llrProcBuf_BG1_Z384_R89;
            }
        }

    }

    return numLLR;
}

#endif
