#include <stdint.h>
#include <stdio.h>

#ifndef CUFFT_H
#define CUFFT_H
#include "cufft.h"
#endif

typedef struct {
  float2 *d_ul_ref_sigs_rx[30][2][33];
} estimation_const_t;

typedef struct {
  short u1;
  short v1;
  short u2;
  short v2;
  short Msc_RS_idx;
  short cyclic_shift1;
  short cyclic_shift2;
  short Msc_RS;
} para_ulsch;

typedef struct {
unsigned int first_rb; 
unsigned short first_carrier_offset;
short N_RB_UL; 
unsigned short nb_rb1; 
unsigned short nb_rb2; 
short fftsize;
} ext_rbs;

typedef struct {
  cufftHandle fft12;
  cufftHandle fft24;
  cufftHandle fft36;
  cufftHandle fft48;
  cufftHandle fft60;
  cufftHandle fft72;
  cufftHandle fft84;
  cufftHandle fft96;
  cufftHandle fft108;
  cufftHandle fft120;
  cufftHandle fft132;
  cufftHandle fft144;
  cufftHandle fft156;
  cufftHandle fft168;
  cufftHandle fft180;
  cufftHandle fft192;
  cufftHandle fft204;
  cufftHandle fft216;
  cufftHandle fft228;
  cufftHandle fft240;
  cufftHandle fft252;
  cufftHandle fft264;
  cufftHandle fft276;
  cufftHandle fft288;
  cufftHandle fft300;
} fftHandle;

typedef struct {
  cudaStream_t stream_ul;
  cudaStream_t timing_advance;
  cudaStream_t tempstrm;
  cufftHandle fft;
  cufftHandle ifft_timing_advance;
  fftHandle idft;
  int    **d_rxdata;
  float2 **d_rxdata_fft;
  int    **d_rxdataF;
  int    **d_rxdata_comp_int;
  float2 **d_rxdata_comp;
  float2 **d_drs_ch;
  int    **d_drs_ch_int;
  int  **d_ulsch_power;
  float2 **d_rxdata_ext;
  int    **d_rxdata_ext_int;
  short  N_RB_UL;
  short  nb_antennas_rx;
  short  symbols_per_tti;
  short  samples_per_tti;
  short  Ncp;
  short  fftsize;
  short  CP;
  short  CP0;
} ul_cu_t;

typedef struct {
  cudaStream_t stream_dl;
  cufftHandle ifft;
  short  *d_txdata;
  short  *d_txdata_o;
  float2 *d_txdata_ifft;
  short  *h_txdata;
  short  symbols_per_tti;
  short  samples_per_tti;
  short  Ncp;
  short  ifftsize;
  short  CP;
  short  CP0;
} dl_cu_t;