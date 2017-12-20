#ifndef __INIT__DEFS_CU__H__
#define __INIT__DEFS_CU__H__
#include <stdint.h>
#include <stdio.h>
#include "PHY/impl_defs_lte.h"
#include "PHY/defs.h"

#ifdef __cplusplus
extern "C"
#endif
void init_cuda( PHY_VARS_eNB *phy_vars_eNB, LTE_DL_FRAME_PARMS frame_parms );
#ifdef __cplusplus
extern "C"
#endif
void free_cufft();


#endif
