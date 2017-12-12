/*==============================================================================
* nrLDPC_bnProc.h
*
* Defines the functions for bit node processing
*
* Author: Sebastian Wagner
* Date: 17-11-2017
*
===============================================================================*/

#ifndef _NR_LDPC_BNPROC_
#define _NR_LDPC_BNPROC_

static inline void nrLDPC_bnProcPc(t_nrLDPC_lut* p_lut, uint16_t Z)
{
    const uint8_t*  lut_numBnInBnGroups = p_lut->numBnInBnGroups;
    const uint32_t* lut_startAddrBnGroups = p_lut->startAddrBnGroups;
    const uint16_t* lut_startAddrBnGroupsLlr = p_lut->startAddrBnGroupsLlr;

    __m128i* p_bnProcBuf;
    __m256i* p_bnProcBufRes;
    __m128i* p_llrProcBuf;
    __m256i* p_llrProcBuf256;
    __m256i* p_llrRes;

    // Number of BNs in Groups
    uint32_t M;
    //uint32_t M32rem;
    uint32_t i,j;
    uint32_t k;
    // Offset to each bit within a group in terms of 32 Byte
    uint32_t cnOffsetInGroup;
    uint8_t idxBnGroup = 0;

    __m256i ymm0, ymm1, ymmRes0, ymmRes1;

    // =====================================================================
    // Process group with 1 CN

    // There is always a BN group with 1 CN
    // Number of groups of 32 BNs for parallel processing
    M = (lut_numBnInBnGroups[0]*Z)>>5;

    p_bnProcBuf     = (__m128i*) &bnProcBuf    [lut_startAddrBnGroups   [idxBnGroup]];
    p_bnProcBufRes  = (__m256i*) &bnProcBufRes [lut_startAddrBnGroups   [idxBnGroup]];
    p_llrProcBuf    = (__m128i*) &llrProcBuf   [lut_startAddrBnGroupsLlr[idxBnGroup]];
    p_llrProcBuf256 = (__m256i*) &llrProcBuf   [lut_startAddrBnGroupsLlr[idxBnGroup]];
    p_llrRes        = (__m256i*) &llrRes       [lut_startAddrBnGroupsLlr[idxBnGroup]];

    // Loop over BNs
    for (i=0,j=0; i<M; i++,j+=2)
    {
        // Store results in bnProcBufRes of first CN for further processing for next iteration
        // In case parity check fails
        p_bnProcBufRes[i] = p_llrProcBuf256[i];

        // First 16 LLRs of first CN
        ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf [j]);
        ymm1 = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);

        ymmRes0 = _mm256_adds_epi16(ymm0, ymm1);

        // Second 16 LLRs of first CN
        ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf [j+1]);
        ymm1 = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);

        ymmRes1 = _mm256_adds_epi16(ymm0, ymm1);

        // Pack results back to epi8
        ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
        // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
        // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
        *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

        // Next result
        p_llrRes++;
    }

    // =====================================================================
    // Process group with 2 CNs

    if (lut_numBnInBnGroups[1] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[1]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[1]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 2
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<2; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 3 CNs

    if (lut_numBnInBnGroups[2] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[2]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[2]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 3
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<3; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 4 CNs

    if (lut_numBnInBnGroups[3] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[3]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[3]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 4
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<4; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 5 CNs

    if (lut_numBnInBnGroups[4] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[4]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[4]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 5
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<5; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 6 CNs

    if (lut_numBnInBnGroups[5] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[5]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[5]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 6
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<6; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 7 CNs

    if (lut_numBnInBnGroups[6] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[6]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[6]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 7
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<7; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 8 CNs

    if (lut_numBnInBnGroups[7] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[7]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[7]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 8
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<8; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 9 CNs

    if (lut_numBnInBnGroups[8] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[8]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[8]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 9
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<9; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 10 CNs

    if (lut_numBnInBnGroups[9] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[9]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[9]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 10
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<10; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 11 CNs

    if (lut_numBnInBnGroups[10] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[10]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[10]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 11
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<11; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 12 CNs

    if (lut_numBnInBnGroups[11] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[11]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[11]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 12
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<12; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 13 CNs

    if (lut_numBnInBnGroups[12] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[12]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[12]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 13
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<13; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 14 CNs

    if (lut_numBnInBnGroups[13] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[13]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[13]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 14
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<14; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 15 CNs

    if (lut_numBnInBnGroups[14] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[14]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[14]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 15
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<15; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 16 CNs

    if (lut_numBnInBnGroups[15] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[15]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[15]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 16
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<16; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 17 CNs

    if (lut_numBnInBnGroups[16] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[16]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[16]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 16
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<17; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 18 CNs

    if (lut_numBnInBnGroups[17] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[17]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[17]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 18
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<18; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 19 CNs

    if (lut_numBnInBnGroups[18] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[18]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[18]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 19
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<19; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 20 CNs

    if (lut_numBnInBnGroups[19] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[19]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[19]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 20
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<20; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 21 CNs

    if (lut_numBnInBnGroups[20] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[20]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[20]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 21
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<21; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 22 CNs

    if (lut_numBnInBnGroups[21] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[21]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[21]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 22
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<22; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

    // =====================================================================
    // Process group with 23 CNs

    if (lut_numBnInBnGroups[22] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[22]*Z)>>5;

        // Set the offset to each CN within a group in terms of 16 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[22]*NR_LDPC_ZMAX)>>4;

        // Set pointers to start of group 23
        p_bnProcBuf  = (__m128i*) &bnProcBuf  [lut_startAddrBnGroups   [idxBnGroup]];
        p_llrProcBuf = (__m128i*) &llrProcBuf [lut_startAddrBnGroupsLlr[idxBnGroup]];
        p_llrRes     = (__m256i*) &llrRes     [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
        for (i=0,j=0; i<M; i++,j+=2)
        {
            // First 16 LLRs of first CN
            ymmRes0 = _mm256_cvtepi8_epi16(p_bnProcBuf[j]);
            ymmRes1 = _mm256_cvtepi8_epi16(p_bnProcBuf[j+1]);

            // Loop over CNs
            for (k=1; k<23; k++)
            {
                ymm0 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j]);
                ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

                ymm1 = _mm256_cvtepi8_epi16(p_bnProcBuf[k*cnOffsetInGroup + j+1]);
                ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);
            }

            // Add LLR from receiver input
            ymm0    = _mm256_cvtepi8_epi16(p_llrProcBuf[j]);
            ymmRes0 = _mm256_adds_epi16(ymmRes0, ymm0);

            ymm1    = _mm256_cvtepi8_epi16(p_llrProcBuf[j+1]);
            ymmRes1 = _mm256_adds_epi16(ymmRes1, ymm1);

            // Pack results back to epi8
            ymm0 = _mm256_packs_epi16(ymmRes0, ymmRes1);
            // ymm0     = [ymmRes1[255:128] ymmRes0[255:128] ymmRes1[127:0] ymmRes0[127:0]]
            // p_llrRes = [ymmRes1[255:128] ymmRes1[127:0] ymmRes0[255:128] ymmRes0[127:0]]
            *p_llrRes = _mm256_permute4x64_epi64(ymm0, 0xD8);

            // Next result
            p_llrRes++;
        }
    }

}

static inline void nrLDPC_bnProc(t_nrLDPC_lut* p_lut, uint16_t Z)
{
    // BN Processing calculating the values to send back to the CNs for next iteration
    // bnProcBufRes contains the sum of all edges to each BN at the start of each group

    const uint8_t*  lut_numBnInBnGroups = p_lut->numBnInBnGroups;
    const uint32_t* lut_startAddrBnGroups = p_lut->startAddrBnGroups;
    const uint16_t* lut_startAddrBnGroupsLlr = p_lut->startAddrBnGroupsLlr;

    __m256i* p_bnProcBuf;
    __m256i* p_bnProcBufRes;
    __m256i* p_llrRes;
    __m256i* p_res;

    // Number of BNs in Groups
    uint32_t M;
    //uint32_t M32rem;
    uint32_t i;
    uint32_t k;
    // Offset to each bit within a group in terms of 32 Byte
    uint32_t cnOffsetInGroup;
    uint8_t idxBnGroup = 0;

    // =====================================================================
    // Process group with 1 CN
    // Already done in bnProcBufPc

    // =====================================================================
    // Process group with 2 CNs

    if (lut_numBnInBnGroups[1] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[1]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[1]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 2
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<2; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes[lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 3 CNs

    if (lut_numBnInBnGroups[2] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[2]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[2]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 3
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<3; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes[lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 4 CNs

    if (lut_numBnInBnGroups[3] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[3]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[3]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 4
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<4; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes[lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 5 CNs

    if (lut_numBnInBnGroups[4] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[4]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[4]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 5
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<5; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 6 CNs

    if (lut_numBnInBnGroups[5] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[5]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[5]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 6
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<6; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes[lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 7 CNs

    if (lut_numBnInBnGroups[6] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[6]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[6]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 7
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<7; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 8 CNs

    if (lut_numBnInBnGroups[7] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[7]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[7]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 8
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<8; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 9 CNs

    if (lut_numBnInBnGroups[8] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[8]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[8]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 9
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<9; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 10 CNs

    if (lut_numBnInBnGroups[9] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[9]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[9]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 10
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<10; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 11 CNs

    if (lut_numBnInBnGroups[10] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[10]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[10]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 10
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<11; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 12 CNs

    if (lut_numBnInBnGroups[11] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[11]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[11]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 12
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<12; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

        // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 13 CNs

    if (lut_numBnInBnGroups[12] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[12]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[12]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 13
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<13; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 14 CNs

    if (lut_numBnInBnGroups[13] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[13]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[13]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 14
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<14; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 15 CNs

    if (lut_numBnInBnGroups[14] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[14]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[14]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 15
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<15; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 16 CNs

    if (lut_numBnInBnGroups[15] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[15]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[15]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 16
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<16; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 17 CNs

    if (lut_numBnInBnGroups[16] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[16]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[16]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 17
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<17; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 18 CNs

    if (lut_numBnInBnGroups[17] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[17]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[17]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 18
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<18; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 19 CNs

    if (lut_numBnInBnGroups[18] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[18]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[18]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 19
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<19; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 20 CNs

    if (lut_numBnInBnGroups[19] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[19]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[19]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 20
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<20; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 21 CNs

    if (lut_numBnInBnGroups[20] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[20]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[20]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 21
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<21; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 22 CNs

    if (lut_numBnInBnGroups[21] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[21]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[21]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 22
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<22; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

    // =====================================================================
    // Process group with 23 CNs

    if (lut_numBnInBnGroups[22] > 0)
    {
        // If elements in group move to next address
        idxBnGroup++;

        // Number of groups of 32 BNs for parallel processing
        M = (lut_numBnInBnGroups[22]*Z)>>5;

        // Set the offset to each CN within a group in terms of 32 Byte
        cnOffsetInGroup = (lut_numBnInBnGroups[22]*NR_LDPC_ZMAX)>>5;

        // Set pointers to start of group 23
        p_bnProcBuf    = (__m256i*) &bnProcBuf   [lut_startAddrBnGroups[idxBnGroup]];
        p_bnProcBufRes = (__m256i*) &bnProcBufRes[lut_startAddrBnGroups[idxBnGroup]];

        // Loop over CNs
        for (k=0; k<23; k++)
        {
            p_res = &p_bnProcBufRes[k*cnOffsetInGroup];
            p_llrRes = (__m256i*) &llrRes [lut_startAddrBnGroupsLlr[idxBnGroup]];

            // Loop over BNs
            for (i=0; i<M; i++)
            {
                *p_res = _mm256_subs_epi8(*p_llrRes, p_bnProcBuf[k*cnOffsetInGroup + i]);

                p_res++;
                p_llrRes++;
            }
        }
    }

}

static inline void nrLDPC_llr2bit(int8_t* llrOut, uint16_t numLLR)
{
    __m256i* p_llrOut = (__m256i*) llrOut;
    int8_t* p_llrOut8;
    uint32_t i;
    uint32_t M  = numLLR>>5;
    uint32_t Mr = numLLR&32;

    const __m256i* p_zeros = (__m256i*) zeros256_epi8;
    const __m256i* p_ones  = (__m256i*) ones256_epi8;

    for (i=0; i<M; i++)
    {
        *p_llrOut = _mm256_and_si256(*p_ones, _mm256_cmpgt_epi8(*p_zeros, *p_llrOut));
        p_llrOut++;
    }

    if (Mr > 0)
    {
        // Remaining LLRs that do not fit in multiples of 32 bytes
        p_llrOut8 = (int8_t*) p_llrOut;

        for (i=0; i<Mr; i++)
        {
            if (p_llrOut8[i] < 0)
            {
                p_llrOut8[i] = 1;
            }
            else
            {
                p_llrOut8[i] = 0;
            }
        }
    }
}

#endif
