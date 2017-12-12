/*==============================================================================
* nrLDPC_defs.h
*
* Defines all constant variables for the LDPC decoder
*
* Author: Sebastian Wagner
* Date: 17-11-2017
*
===============================================================================*/

#ifndef _NR_LDPC_DEFS_
#define _NR_LDPC_DEFS_

// ==============================================================================
// DEFINES

// Maximum lifting size
#define NR_LDPC_ZMAX 384
// Row and column dimensions of BG1 and BG2
#define NR_LDPC_NCOL_BG1 68
#define NR_LDPC_NROW_BG1 46
#define NR_LDPC_NROW_BG2 42
#define NR_LDPC_NCOL_BG2 52
#define NR_LDPC_NUM_EDGE_BG1 316
#define NR_LDPC_NUM_EDGE_BG2 197

#define NR_LDPC_NUM_CN_GROUPS_BG1 9
#define NR_LDPC_NUM_CN_GROUPS_BG2 6

#define NR_LDPC_NCOL_BG2_R15 NR_LDPC_NCOL_BG2
#define NR_LDPC_NCOL_BG2_R13 32
#define NR_LDPC_NCOL_BG2_R23 17

#define NR_LDPC_NUM_BN_GROUPS_BG1 13
#define NR_LDPC_NUM_BN_GROUPS_BG2 23
#define NR_LDPC_SIZE_CN_PROC_BUF NR_LDPC_NUM_EDGE_BG1*NR_LDPC_ZMAX
#define NR_LDPC_SIZE_BN_PROC_BUF NR_LDPC_NUM_EDGE_BG1*NR_LDPC_ZMAX

#define NR_LDPC_MAX_NUM_LLR 26112 // NR_LDPC_NCOL_BG1*NR_LDPC_ZMAX

#define NR_LDPC_NUM_MAX_ITER 1
#define MEAS_TIME

// ==============================================================================
// GLOBAL VARIABLES

// Aligned on 32 bytes = 256 bits for AVX2
static int8_t cnProcBuf   [NR_LDPC_SIZE_CN_PROC_BUF] __attribute__ ((aligned(32)));
static int8_t cnProcBufRes[NR_LDPC_SIZE_CN_PROC_BUF] __attribute__ ((aligned(32)));

static int8_t bnProcBuf   [NR_LDPC_SIZE_BN_PROC_BUF] __attribute__ ((aligned(32)));
static int8_t bnProcBufRes[NR_LDPC_SIZE_BN_PROC_BUF] __attribute__ ((aligned(32)));

static int8_t llrRes    [NR_LDPC_MAX_NUM_LLR] __attribute__ ((aligned(32)));
static int8_t llrProcBuf[NR_LDPC_MAX_NUM_LLR] __attribute__ ((aligned(32)));

// Start addresses for the cnProcBuf for each CN group
static const uint32_t lut_startAddrCnGroups_BG1[NR_LDPC_NUM_CN_GROUPS_BG1] = {0, 1152, 8832, 43392, 61824, 75264, 81408, 88320, 92160};
static const uint32_t lut_startAddrCnGroups_BG2[NR_LDPC_NUM_CN_GROUPS_BG2] = {0, 6912, 37632, 54912, 61824, 67968};

// Number of groups for check node processing
static const uint8_t lut_numCnInCnGroups_BG1[NR_LDPC_NUM_CN_GROUPS_BG1] = {1, 5, 18, 8, 5, 2, 2, 1, 4};

static const uint8_t lut_numBnInCnGroups_BG2_R15[NR_LDPC_NUM_CN_GROUPS_BG2] = {3,  4, 5, 6, 8, 10};
static const uint8_t lut_numCnInCnGroups_BG2_R15[NR_LDPC_NUM_CN_GROUPS_BG2] = {6, 20, 9, 3, 2, 2};
static const uint8_t lut_numCnInCnGroups_BG2_R23[NR_LDPC_NUM_CN_GROUPS_BG2] = {0,  1, 0, 2, 2, 2};
static const uint8_t lut_numCnInCnGroups_BG2_R13[NR_LDPC_NUM_CN_GROUPS_BG2] = {0,  8, 7, 3, 2, 2};

// Number of groups for bit node processing
static const uint8_t lut_numBnInBnGroups_BG1[NR_LDPC_NUM_BN_GROUPS_BG1] = {42, 1, 1, 2, 4, 3, 1, 4, 3, 4, 1, 1, 1};


                                                                        // BG2: 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
static const uint8_t lut_numBnInBnGroups_BG2_R15[NR_LDPC_NUM_BN_GROUPS_BG2] = {38, 0, 0, 0, 2, 1, 1, 1, 2, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1};
static const uint8_t lut_numBnInBnGroups_BG2_R13[NR_LDPC_NUM_BN_GROUPS_BG2] = {18, 1, 0, 2, 1, 5, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0};
static const uint8_t lut_numBnInBnGroups_BG2_R23[NR_LDPC_NUM_BN_GROUPS_BG2] = { 3, 3, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


// Start addresses for the bnProcBuf for each BN group
static const uint32_t lut_startAddrBnGroups_BG1[NR_LDPC_NUM_BN_GROUPS_BG1] = {0, 16128, 17664, 19584, 24192, 34944, 44160, 47616, 62976, 75648, 94080, 99072, 109824};
static const uint32_t lut_startAddrBnGroups_BG2_R15[NR_LDPC_NUM_BN_GROUPS_BG2] = {0, 14592, 18432, 20736, 23424, 26496, 33408, 37248, 41856, 46848, 52224, 58368, 66816};
static const uint32_t lut_startAddrBnGroups_BG2_R13[NR_LDPC_NUM_BN_GROUPS_BG2] = {0, 6912, 7680, 10752, 12672, 24192, 26880, 29952, 34944, 40320};
static const uint32_t lut_startAddrBnGroups_BG2_R23[NR_LDPC_NUM_BN_GROUPS_BG2] = {0, 1152, 3456, 9216, 13824, 17664};

static const uint16_t lut_startAddrBnGroupsLlr_BG2_R15[NR_LDPC_NUM_BN_GROUPS_BG2] = {0, 14592, 15360, 15744, 16128, 16512, 17280, 17664, 18048, 18432, 18816, 19200, 19584};
static const uint16_t lut_startAddrBnGroupsLlr_BG2_R13[NR_LDPC_NUM_BN_GROUPS_BG2] = {0, 6912, 7296, 8064, 8448, 10368, 10752, 11136, 11520, 11904};
static const uint16_t lut_startAddrBnGroupsLlr_BG2_R23[NR_LDPC_NUM_BN_GROUPS_BG2] = {0,  1152,  2304,  4224,  5376, 6144};

static const int8_t ones256_epi8[32] __attribute__ ((aligned(32))) = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
static const int8_t zeros256_epi8[32] __attribute__ ((aligned(32))) = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static const int8_t maxLLR256_epi8[32] __attribute__ ((aligned(32))) = {127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127};

#endif
