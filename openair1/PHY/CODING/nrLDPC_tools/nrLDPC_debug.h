/*==============================================================================
* nrLDPC_debug.h
*
* Defines the debugging functions
*
* Author: Sebastian Wagner
* Date: 15-01-2018
*
===============================================================================*/

#ifndef __NR_LDPC_DEBUG__H__
#define __NR_LDPC_DEBUG__H__

#include <stdio.h>

typedef enum nrLDPC_buffers {
    nrLDPC_buffers_LLR_PROC,
    nrLDPC_buffers_CN_PROC,
    nrLDPC_buffers_CN_PROC_RES,
    nrLDPC_buffers_BN_PROC,
    nrLDPC_buffers_BN_PROC_RES,
    nrLDPC_buffers_LLR_RES
} e_nrLDPC_buffers;

static inline void nrLDPC_writeFile(const char* fileName, int8_t* p_data, const uint32_t N)
{
    FILE *f;
    uint32_t i;

    f = fopen(fileName, "a");

    // Newline indicating new data
    fprintf(f, "\n");
    for (i=0; i < N; i++)
    {
        fprintf(f, "%d, ", p_data[i]);
    }

    fclose(f);
}

static inline void nrLDPC_initFile(const char* fileName)
{
    FILE *f;

    f = fopen(fileName, "w");

    fprintf(f, " ");

    fclose(f);
}

static inline void nrLDPC_debug_writeBuffer2File(e_nrLDPC_buffers buffer)
{
    switch (buffer)
    {
    case nrLDPC_buffers_LLR_PROC:
    {
        nrLDPC_writeFile("llrProcBuf.txt", llrProcBuf, NR_LDPC_MAX_NUM_LLR);
        break;
    }
    case nrLDPC_buffers_CN_PROC:
    {
        nrLDPC_writeFile("cnProcBuf.txt", cnProcBuf, NR_LDPC_SIZE_CN_PROC_BUF);
        break;
    }
    case nrLDPC_buffers_CN_PROC_RES:
    {
        nrLDPC_writeFile("cnProcBufRes.txt", cnProcBufRes, NR_LDPC_SIZE_CN_PROC_BUF);
        break;
    }
    case nrLDPC_buffers_BN_PROC:
    {
        nrLDPC_writeFile("bnProcBuf.txt", bnProcBuf, NR_LDPC_SIZE_BN_PROC_BUF);
        break;
    }
    case nrLDPC_buffers_BN_PROC_RES:
    {
        nrLDPC_writeFile("bnProcBufRes.txt", bnProcBufRes, NR_LDPC_SIZE_BN_PROC_BUF);
        break;
    }
    case nrLDPC_buffers_LLR_RES:
    {
        nrLDPC_writeFile("llrRes.txt", llrRes, NR_LDPC_MAX_NUM_LLR);
        break;
    }
    }
}

static inline void nrLDPC_debug_initBuffer2File(e_nrLDPC_buffers buffer)
{
    switch (buffer)
    {
    case nrLDPC_buffers_LLR_PROC:
    {
        nrLDPC_initFile("llrProcBuf.txt");
        break;
    }
    case nrLDPC_buffers_CN_PROC:
    {
        nrLDPC_initFile("cnProcBuf.txt");
        break;
    }
    case nrLDPC_buffers_CN_PROC_RES:
    {
        nrLDPC_initFile("cnProcBufRes.txt");
        break;
    }
    case nrLDPC_buffers_BN_PROC:
    {
        nrLDPC_initFile("bnProcBuf.txt");
        break;
    }
    case nrLDPC_buffers_BN_PROC_RES:
    {
        nrLDPC_initFile("bnProcBufRes.txt");
        break;
    }
    case nrLDPC_buffers_LLR_RES:
    {
        nrLDPC_initFile("llrRes.txt");
        break;
    }
    }
}

static inline void nrLDPC_debug_print256i_epi8(__m256i* in)
{
    uint32_t i;
    
    for (i=0; i<32; i++)
    {
        mexPrintf("%d ", ((int8_t*)&in)[i]);
    }
}

#endif
