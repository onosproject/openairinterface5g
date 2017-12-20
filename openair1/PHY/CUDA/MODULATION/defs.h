#ifndef __DEFS_CU__H__
#define __DEFS_CU__H__
#include <stdint.h>
#include <stdio.h>

#ifndef CUFFT_H
#define CUFFT_H
#include "cufft.h"
#endif

//typedef float2 Complex;

#ifdef __cplusplus
extern "C"
#endif
void idft512ad_cu( short *, short *, int );

#ifdef __cplusplus
extern "C"
#endif
void dft512rm_cu( short *, short *, int  );

#endif
