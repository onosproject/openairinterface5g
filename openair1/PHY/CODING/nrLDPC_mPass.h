/*==============================================================================
* nrLDPC_mPass.h
*
* Defines the functions for message passing
*
* Author: Sebastian Wagner
* Date: 17-11-2017
*
===============================================================================*/

#ifndef _NR_LDPC_MPASS_
#define _NR_LDPC_MPASS_

static inline void nrLDPC_llr2llrProcBuf(t_nrLDPC_lut* p_lut, int8_t* llr, uint16_t numLLR)
{
    const uint16_t* lut_llr2llrProcBuf = p_lut->llr2llrProcBuf;
    uint32_t i;

    for (i=0; i<numLLR; i++)
    {
        llrProcBuf[lut_llr2llrProcBuf[i]] = llr[i];
    }
}

static inline void nrLDPC_llr2CnProcBuf(t_nrLDPC_lut* p_lut, int8_t* llr, uint16_t numLLR)
{
    const uint32_t* lut_llr2CnProcBuf = p_lut->llr2CnProcBuf;
    const uint8_t* lut_numEdgesPerBn  = p_lut->numEdgesPerBn;
    uint32_t idxLut = 0;
    uint32_t idxCnProcBuf = 0;
    int8_t curLLR;
    uint32_t i;
    uint32_t j;

    for (i=0; i<numLLR; i++)
    {
        curLLR = llr[i];

        for (j=0; j<lut_numEdgesPerBn[i]; j++)
        {
            idxCnProcBuf = lut_llr2CnProcBuf[idxLut];
            cnProcBuf[idxCnProcBuf] = curLLR;
            idxLut++;
        }
    }
}

static inline void nrLDPC_cn2bnProcBuf(t_nrLDPC_lut* p_lut, uint16_t Z)
{
    const uint32_t* lut_cn2bnProcBuf = p_lut->cn2bnProcBuf;
    const uint8_t*  lut_numCnInCnGroups = p_lut->numCnInCnGroups;
    const uint32_t* lut_startAddrCnGroups = p_lut->startAddrCnGroups;

    const uint32_t* p_lut_cn2bn;
    int8_t* p_cnProcBufRes;
    uint32_t bitOffsetInGroup;
    uint32_t i;
    uint32_t j;
    uint32_t M;

    // =====================================================================
    // CN group with 3 BNs

    p_lut_cn2bn = &lut_cn2bnProcBuf[0];
    M = lut_numCnInCnGroups[0]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[0]*NR_LDPC_ZMAX;

    for (j=0; j<3; j++)
    {
        p_cnProcBufRes = &cnProcBufRes[lut_startAddrCnGroups[0] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            bnProcBuf[p_lut_cn2bn[j*M + i]] = p_cnProcBufRes[i];
        }
    }

    // =====================================================================
    // CN group with 4 BNs

    p_lut_cn2bn += (M*3); // Number of elements of previous group
    M = lut_numCnInCnGroups[1]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[1]*NR_LDPC_ZMAX;

    for (j=0; j<4; j++)
    {
        p_cnProcBufRes = &cnProcBufRes[lut_startAddrCnGroups[1] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            bnProcBuf[p_lut_cn2bn[j*M + i]] = p_cnProcBufRes[i];
        }
    }

    // =====================================================================
    // CN group with 5 BNs

    p_lut_cn2bn += (M*4); // Number of elements of previous group
    M = lut_numCnInCnGroups[2]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[2]*NR_LDPC_ZMAX;

    for (j=0; j<5; j++)
    {
        p_cnProcBufRes = &cnProcBufRes[lut_startAddrCnGroups[2] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            bnProcBuf[p_lut_cn2bn[j*M + i]] = p_cnProcBufRes[i];
        }
    }

    // =====================================================================
    // CN group with 6 BNs

    p_lut_cn2bn += (M*5); // Number of elements of previous group
    M = lut_numCnInCnGroups[3]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[3]*NR_LDPC_ZMAX;

    for (j=0; j<6; j++)
    {
        p_cnProcBufRes = &cnProcBufRes[lut_startAddrCnGroups[3] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            bnProcBuf[p_lut_cn2bn[j*M + i]] = p_cnProcBufRes[i];
        }
    }

    // =====================================================================
    // CN group with 8 BNs

    p_lut_cn2bn += (M*6); // Number of elements of previous group
    M = lut_numCnInCnGroups[4]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[4]*NR_LDPC_ZMAX;

    for (j=0; j<8; j++)
    {
        p_cnProcBufRes = &cnProcBufRes[lut_startAddrCnGroups[4] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            bnProcBuf[p_lut_cn2bn[j*M + i]] = p_cnProcBufRes[i];
        }
    }

    // =====================================================================
    // CN group with 10 BNs

    p_lut_cn2bn += (M*8); // Number of elements of previous group
    M = lut_numCnInCnGroups[5]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[5]*NR_LDPC_ZMAX;

    for (j=0; j<10; j++)
    {
        p_cnProcBufRes = &cnProcBufRes[lut_startAddrCnGroups[5] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            bnProcBuf[p_lut_cn2bn[j*M + i]] = p_cnProcBufRes[i];
        }
    }

}

static inline void nrLDPC_bn2cnProcBuf(t_nrLDPC_lut* p_lut, uint16_t Z)
{
    const uint32_t* lut_cn2bnProcBuf = p_lut->cn2bnProcBuf;
    const uint8_t*  lut_numCnInCnGroups = p_lut->numCnInCnGroups;
    const uint32_t* lut_startAddrCnGroups = p_lut->startAddrCnGroups;

    int8_t* p_cnProcBuf;
    const uint32_t* p_lut_cn2bn;
    uint32_t bitOffsetInGroup;
    uint32_t i;
    uint32_t j;
    uint32_t M;

    // For CN groups 3 to 6 no need to send the last BN back since it's single edge
    // and BN processing does not change the value already in the CN proc buf

    // =====================================================================
    // CN group with 3 BNs

    p_lut_cn2bn = &lut_cn2bnProcBuf[0];
    M = lut_numCnInCnGroups[0]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[0]*NR_LDPC_ZMAX;

    for (j=0; j<2; j++)
    {
        p_cnProcBuf = &cnProcBuf[lut_startAddrCnGroups[0] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            p_cnProcBuf[i] = bnProcBufRes[p_lut_cn2bn[j*M + i]];
        }
    }

    // =====================================================================
    // CN group with 4 BNs

    p_lut_cn2bn += (M*3); // Number of elements of previous group
    M = lut_numCnInCnGroups[1]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[1]*NR_LDPC_ZMAX;

    for (j=0; j<3; j++)
    {
        p_cnProcBuf = &cnProcBuf[lut_startAddrCnGroups[1] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            p_cnProcBuf[i] = bnProcBufRes[p_lut_cn2bn[j*M + i]];
        }
    }

    // =====================================================================
    // CN group with 5 BNs

    p_lut_cn2bn += (M*4); // Number of elements of previous group
    M = lut_numCnInCnGroups[2]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[2]*NR_LDPC_ZMAX;

    for (j=0; j<4; j++)
    {
        p_cnProcBuf = &cnProcBuf[lut_startAddrCnGroups[2] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            p_cnProcBuf[i] = bnProcBufRes[p_lut_cn2bn[j*M + i]];
        }
    }

    // =====================================================================
    // CN group with 6 BNs

    p_lut_cn2bn += (M*5); // Number of elements of previous group
    M = lut_numCnInCnGroups[3]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[3]*NR_LDPC_ZMAX;

    for (j=0; j<5; j++)
    {
        p_cnProcBuf = &cnProcBuf[lut_startAddrCnGroups[3] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            p_cnProcBuf[i] = bnProcBufRes[p_lut_cn2bn[j*M + i]];
        }
    }

    // =====================================================================
    // CN group with 8 BNs

    p_lut_cn2bn += (M*6); // Number of elements of previous group
    M = lut_numCnInCnGroups[4]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[4]*NR_LDPC_ZMAX;

    for (j=0; j<8; j++)
    {
        p_cnProcBuf = &cnProcBuf[lut_startAddrCnGroups[4] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            p_cnProcBuf[i] = bnProcBufRes[p_lut_cn2bn[j*M + i]];
        }
    }

    // =====================================================================
    // CN group with 10 BNs

    p_lut_cn2bn += (M*8); // Number of elements of previous group
    M = lut_numCnInCnGroups[5]*Z;
    bitOffsetInGroup = lut_numCnInCnGroups_BG2_R15[5]*NR_LDPC_ZMAX;

    for (j=0; j<10; j++)
    {
        p_cnProcBuf = &cnProcBuf[lut_startAddrCnGroups[5] + j*bitOffsetInGroup];

        for (i=0; i<M; i++)
        {
            p_cnProcBuf[i] = bnProcBufRes[p_lut_cn2bn[j*M + i]];
        }
    }

}

static inline void nrLDPC_llrRes2llrOut(t_nrLDPC_lut* p_lut, int8_t* llrOut, uint16_t numLLR)
{
    const uint16_t* lut_llr2llrProcBuf = p_lut->llr2llrProcBuf;
    uint32_t i;

    for (i=0; i<numLLR; i++)
    {
        llrOut[i] = llrRes[lut_llr2llrProcBuf[i]];
    }
}

#endif
