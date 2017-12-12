/*==============================================================================
* nrLDPC_decoder.c
*
* Defines the LDPC decoder
* p_llrOut = output LLRs aligned on 32 byte boundaries
*
* Author: Sebastian Wagner
* Date: 17-11-2017
*
===============================================================================*/

#include <stdint.h>
#include <immintrin.h>
#include "nrLDPC_defs.h"
#include "nrLDPC_types.h"
#include "nrLDPC_init.h"
#include "nrLDPC_mPass.h"
#include "nrLDPC_cnProc.h"
#include "nrLDPC_bnProc.h"

#ifdef MEAS_TIME
#include <time.h>
#include <string.h>
#endif

static inline void nrLDPC_decoder_core(int8_t* p_llr, int8_t* p_llrOut, uint32_t numLLR, t_nrLDPC_lut* p_lut, t_nrLDPC_dec_params* p_decParams, t_nrLDPC_proc_time* p_procTime);

void nrLDPC_decoder(t_nrLDPC_dec_params* p_decParams, int16_t* p_llr, int8_t* p_llrOut, t_nrLDPC_proc_time* p_procTime)
{
    uint32_t numLLR;
    t_nrLDPC_lut lut;
    t_nrLDPC_lut* p_lut = &lut;
    //int8_t llrOut_inter;
    //int8_t* p_llrOut_inter=&llrOut_inter;

    // Initialize decoder core(s) with correct LUTs
    numLLR = nrLDPC_init(p_decParams, p_lut);

    // Launch LDPC decoder core for one segment
    nrLDPC_decoder_core((int8_t *)p_llr, p_llrOut, numLLR, p_lut, p_decParams, p_procTime);
    //nrLDPC_decoder_core((int8_t *) p_llr, p_llrOut_inter, numLLR, p_lut, p_decParams, p_procTime);
    //p_llrOut = (uint8_t *) p_llrOut_inter;
}

static inline void nrLDPC_decoder_core(int8_t* p_llr, int8_t* p_llrOut, uint32_t numLLR, t_nrLDPC_lut* p_lut, t_nrLDPC_dec_params* p_decParams, t_nrLDPC_proc_time* p_procTime)
{
	printf("LDPC core p_llr %d\n",*p_llr);
    uint16_t Z          = p_decParams->Z;
    uint8_t  numMaxIter = p_decParams->numMaxIter;

    uint32_t i = 0;

#ifdef MEAS_TIME
    clock_t start_all, end_all, start, end;
    memset(p_procTime, 0, sizeof(t_nrLDPC_proc_time));
#endif

#ifdef MEAS_TIME
    start_all = clock();
#endif

    // Initialization
#ifdef MEAS_TIME
    start = clock();
#endif
    nrLDPC_llr2llrProcBuf(p_lut, p_llr, numLLR);
#ifdef MEAS_TIME
    end = clock();
    p_procTime->llr2llrProcBuf += ((double) (end - start));
#endif

#ifdef MEAS_TIME
    start = clock();
#endif
    nrLDPC_llr2CnProcBuf(p_lut, p_llr, numLLR);
#ifdef MEAS_TIME
    end = clock();
    p_procTime->llr2CnProcBuf += ((double) (end - start));
#endif

    // First iteration
#ifdef MEAS_TIME
    start = clock();
#endif
    nrLDPC_cnProc(p_lut, Z);

#ifdef MEAS_TIME
    end = clock();
    p_procTime->cnProc += ((double) (end - start));
#endif

#ifdef MEAS_TIME
    start = clock();
#endif
    nrLDPC_cn2bnProcBuf(p_lut, Z);
#ifdef MEAS_TIME
    end = clock();
    p_procTime->cn2bnProcBuf += ((double) (end - start));
#endif

#ifdef MEAS_TIME
    start = clock();
#endif
    nrLDPC_bnProcPc(p_lut, Z);
#ifdef MEAS_TIME
    end = clock();
    p_procTime->bnProcPc += ((double) (end - start));
#endif

    // Parity check should occur here

    while (i < numMaxIter)
    {
        // BN processing
#ifdef MEAS_TIME
        start = clock();
#endif
        nrLDPC_bnProc(p_lut, Z);
#ifdef MEAS_TIME
        end = clock();
        p_procTime->bnProc += ((double) (end - start));
#endif

        // BN results to CN processing buffer
#ifdef MEAS_TIME
        start = clock();
#endif
        nrLDPC_bn2cnProcBuf(p_lut, Z);
#ifdef MEAS_TIME
        end = clock();
        p_procTime->bn2cnProcBuf += ((double) (end - start));
#endif

        // CN processing
#ifdef MEAS_TIME
        start = clock();
#endif
        nrLDPC_cnProc(p_lut, Z);
#ifdef MEAS_TIME
        end = clock();
        p_procTime->cnProc += ((double) (end - start));
#endif

        // Send CN results back to BNs
#ifdef MEAS_TIME
        start = clock();
#endif
        nrLDPC_cn2bnProcBuf(p_lut, Z);
#ifdef MEAS_TIME
        end = clock();
        p_procTime->cn2bnProcBuf += ((double) (end - start));
#endif

#ifdef MEAS_TIME
        start = clock();
#endif
        nrLDPC_bnProcPc(p_lut, Z);
#ifdef MEAS_TIME
        end = clock();
        p_procTime->bnProcPc += ((double) (end - start));
#endif

        // Do parity check

        i++;
    }

    // Assign results from processing buffer to output
#ifdef MEAS_TIME
    start = clock();
#endif
    nrLDPC_llrRes2llrOut(p_lut, p_llrOut, numLLR);
#ifdef MEAS_TIME
    end = clock();
    p_procTime->llrRes2llrOut += ((double) (end - start));
#endif

    // Hard-decision
#ifdef MEAS_TIME
    start = clock();
#endif
    nrLDPC_llr2bit(p_llrOut, numLLR);
#ifdef MEAS_TIME
    end = clock();
    p_procTime->llr2bit += ((double) (end - start));
#endif


#ifdef MEAS_TIME
    end_all = clock();
    p_procTime->total = ((double) (end_all - start_all));

    p_procTime->llr2llrProcBuf /= CLOCKS_PER_SEC;
    p_procTime->llr2CnProcBuf  /= CLOCKS_PER_SEC;
    p_procTime->cnProc         /= CLOCKS_PER_SEC;
    p_procTime->bnProcPc       /= CLOCKS_PER_SEC;
    p_procTime->bnProc         /= CLOCKS_PER_SEC;
    p_procTime->cn2bnProcBuf   /= CLOCKS_PER_SEC;
    p_procTime->bn2cnProcBuf   /= CLOCKS_PER_SEC;
    p_procTime->llrRes2llrOut  /= CLOCKS_PER_SEC;
    p_procTime->llr2bit        /= CLOCKS_PER_SEC;
    p_procTime->total          /= CLOCKS_PER_SEC;
#endif

}
