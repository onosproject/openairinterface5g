/*
	Author: Shi-Yi Oh, Judy Wang, Shao-Ying Yeh, Yuan-Te Liao, Jian-Ya Chu, Wei-Shin Huang, Ying-Liang Chen, Terng-Yin Hsu
	From ISIP CS/NCTU, Hsinchu, Taiwan
*/
#include "defs.h"
#include "PHY/CUDA/extern.h"
#include "PHY/LTE_TRANSPORT/extern.h"

#include <stdio.h>


int device_count;
dl_cu_t dl_cu[10];
ul_cu_t ul_cu[10];
estimation_const_t esti_const;
para_ulsch ulsch_para[10];
ext_rbs ext_rbs_para[10];

void init_cufft( void );
void free_cufft( void );

__global__ void generate_ul_ref_sigs_rx_cu( float2 *x, unsigned int Msc_RS, unsigned int u, unsigned int v )
{
  unsigned short ref_primes[33] = {11,23,31,47,50,71,89,107,113,139,179,191,211,239,283,293,317,359,383,431,479,523,571,599,647,719,863,887,953,971,1069,1151,1193};
  float qbar;
  float phase;
  unsigned short dftsizes[33] = { 12,24,36,48,60,72,96,108,120,144,180,192,216,240,288,300,324,360,384,432,480,540,576,600,648,720,864,900,960,972,1080,1152,1200 };
  char ref24[720] = {
  -1,3,1,-3,3,-1,1,3,-3,3,1,3,-3,3,1,1,-1,1,3,-3,3,-3,-1,-3,-3,3,-3,-3,-3,1,-3,-3,3,-1,1,1,1,3,1,-1,3,-3,-3,1,3,1,1,-3,3,-1,3,3,1,1,-3,3,3,3,3,1,-1,3,-1,1,1,-1,-3,-1,-1,1,3,3,-1,-3,1,1,3,-3,1,1,-3,-1,-1,1,3,1,3,1,-1,3,1,1,-3,-1,-3,-1,-1,-1,-1,-3,-3,-1,1,1,3,3,-1,3,-1,1,-1,-3,1,-1,-3,-3,1,-3,-1,-1,-3,1,1,3,-1,1,3,1,-3,1,-3,1,1,-1,-1,3,-1,-3,3,-3,-3,-3,1,1,1,1,-1,-1,3,-3,-3,3,-3,1,-1,-1,1,-1,1,1,-1,-3,-1,1,-1,3,-1,-3,-3,3,3,-1,-1,-3,-1,3,1,3,1,3,1,1,-1,3,1,-1,1,3,-3,-1,-1,1,-3,1,3,-3,1,-1,-3,3,-3,3,-1,-1,-1,-1,1,-3,-3,-3,1,-3,-3,-3,1,-3,1,1,-3,3,3,-1,-3,-1,3,-3,3,3,3,-1,1,1,-3,1,-1,1,1,-3,1,1,-1,1,-3,-3,3,-1,3,-1,-1,-3,-3,-3,-1,-3,-3,1,-1,1,3,3,-1,1,-1,3,1,3,3,-3,-3,1,3,1,-1,-3,-3,-3,3,3,-3,3,3,-1,-3,3,-1,1,-3,1,1,3,3,1,1,1,-1,-1,1,-3,3,-1,1,1,-3,3,3,-1,-3,3,-3,-1,-3,-1,-1,-1,-1,-1,-3,-1,3,3,1,-1,1,3,3,3,-1,1,1,-3,1,3,-1,-3,3,-3,-3,3,1,3,1,-3,3,1,3,1,1,3,3,-1,-1,-3,1,-3,-1,3,1,1,3,-1,-1,1,-3,1,3,-3,1,-1,-3,-1,3,1,3,1,-1,-3,-3,-1,-1,-3,-3,-3,-1,-1,-3,3,-1,-1,-1,-1,1,1,-3,3,1,3,3,1,-1,1,-3,1,-3,1,1,-3,-1,1,3,-1,3,3,-1,-3,1,-1,-3,3,3,3,-1,1,1,3,-1,-3,-1,3,-1,-1,-1,1,1,1,1,1,-1,3,-1,-3,1,1,3,-3,1,-3,-1,1,1,-3,-3,3,1,1,-3,1,3,3,1,-1,-3,3,-1,3,3,3,-3,1,-1,1,-1,-3,-1,1,3,-1,3,-3,-3,-1,-3,3,-3,-3,-3,-1,-1,-3,-1,-3,3,1,3,-3,-1,3,-1,1,-1,3,-3,1,-1,-3,-3,1,1,-1,1,-1,1,-1,3,1,-3,-1,1,-1,1,-1,-1,3,3,-3,-1,1,-3,-3,-1,-3,3,1,-1,-3,-1,-3,-3,3,-3,3,-3,-1,1,3,1,-3,1,3,3,-1,-3,-1,-1,-1,-1,3,3,3,1,3,3,-3,1,3,-1,3,-1,3,3,-3,3,1,-1,3,3,1,-1,3,3,-1,-3,3,-3,-1,-1,3,-1,3,-1,-1,1,1,1,1,-1,-1,-3,-1,3,1,-1,1,-1,3,-1,3,1,1,-1,-1,-3,1,1,-3,1,3,-3,1,1,-3,-3,-1,-1,-3,-1,1,3,1,1,-3,-1,-1,-3,3,-3,3,1,-3,3,-3,1,-1,1,-3,1,1,1,-1,-3,3,3,1,1,3,-1,-3,-1,-1,-1,3,1,-3,-3,-1,3,-3,-1,-3,-1,-3,-1,-1,-3,-1,-1,1,-3,-1,-1,1,-1,-3,1,1,-3,1,-3,-3,3,1,1,-1,3,-1,-1,1,1,-1,-1,-3,-1,3,-1,3,-1,1,3,1,-1,3,1,3,-3,-3,1,-1,-1,1,3
  };
  char ref12[360] = {-1,1,3,-3,3,3,1,1,3,1,-3,3,1,1,3,3,3,-1,1,-3,-3,1,-3,3,1,1,-3,-3,-3,-1,-3,-3,1,-3,1,-1,-1,1,1,1,1,-1,-3,-3,1,-3,3,-1,-1,3,1,-1,1,-1,-3,-1,1,-1,1,3,1,-3,3,-1,-1,1,1,-1,-1,3,-3,1,-1,3,-3,-3,-3,3,1,-1,3,3,-3,1,-3,-1,-1,-1,1,-3,3,-1,1,-3,3,1,1,-3,3,1,-1,-1,-1,1,1,3,-1,1,1,-3,-1,3,3,-1,-3,1,1,1,1,1,-1,3,-1,1,1,-3,-3,-1,-3,-3,3,-1,3,1,-1,-1,3,3,-3,1,3,1,3,3,1,-3,1,1,-3,1,1,1,-3,-3,-3,1,3,3,-3,3,-3,1,1,3,-1,-3,3,3,-3,1,-1,-3,-1,3,1,3,3,3,-1,1,3,-1,1,-3,-1,-1,1,1,3,1,-1,-3,1,3,1,-1,1,3,3,3,-1,-1,3,-1,-3,1,1,3,-3,3,-3,-3,3,1,3,-1,-3,3,1,1,-3,1,-3,-3,-1,-1,1,-3,-1,3,1,3,1,-1,-1,3,-3,-1,-3,-1,-1,-3,1,1,1,1,3,1,-1,1,-3,-1,-1,3,-1,1,-3,-3,-3,-3,-3,1,-1,-3,1,1,-3,-3,-3,-3,-1,3,-3,1,-3,3,1,1,-1,-3,-1,-3,1,-1,1,3,-1,1,1,1,3,1,3,3,-1,1,-1,-3,-3,1,1,-3,3,3,1,3,3,1,-3,-1,-1,3,1,3,-3,-3,3,-3,1,-1,-1,3,-1,-3,-3,-1,-3,-1,-3,3,1,-1,1,3,-3,-3,-1,3,-3,3,-1,3,3,-3,3,3,-1,-1,3,-3,-3,-1,-1,-3,-1,3,-3,3,1,-1};

  unsigned int q,m,n;
  if( Msc_RS >= 2 )
  {
	qbar = ref_primes[Msc_RS] * (u+1)/(double)31;
	if ((((int)floor(2*qbar))&1) == 0)
      q = (int)(floor(qbar+.5)) - v;
    else
      q = (int)(floor(qbar+.5)) + v;
    for (n=0; n<dftsizes[Msc_RS]; n++) 
    {
      m=n%ref_primes[Msc_RS];
      phase = (float)q*m*(m+1)/ref_primes[Msc_RS];
      x[n].x = cosf(M_PI*phase);
      x[n].y =-sinf(M_PI*phase);
    }
  }
  else if ( Msc_RS == 1 )
  {
    for (n=0; n<dftsizes[1]; n++) {
      x[n].x   = cosf(M_PI*((float)ref24[(u*24) + n])/4);
      x[n].y   = sinf(M_PI*((float)ref24[(u*24) + n])/4);
    }
  }
  else if ( Msc_RS == 0 )
  {
	for (n=0; n<dftsizes[0]; n++) {
      x[n].x = cosf(M_PI*ref12[(u*12) + n]/4);
      x[n].y = sinf(M_PI*ref12[(u*12) + n]/4);
    }
  }
}


void init_cuda(PHY_VARS_eNB *phy_vars_eNB, LTE_DL_FRAME_PARMS frame_parms )
{
  unsigned short dftsizes[33] = { 12,24,36,48,60,72,96,108,120,144,180,192,216,240,288,300,324,360,384,432,480,540,576,600,648,720,864,900,960,972,1080,1152,1200 }; 
  int i,j,k;
  int u,v,Msc_RS;
  cudaGetDeviceCount(&device_count);
  printf("[CUDA] now we have %d device\n",device_count);
  LTE_DL_FRAME_PARMS* const frame_parm = &phy_vars_eNB->lte_frame_parms;
  LTE_eNB_COMMON* const eNB_common_vars = &phy_vars_eNB->lte_eNB_common_vars;
  LTE_eNB_PUSCH** const eNB_pusch_vars  = phy_vars_eNB->lte_eNB_pusch_vars;
  LTE_eNB_SRS* const eNB_srs_vars       = phy_vars_eNB->lte_eNB_srs_vars;
  LTE_eNB_PRACH* const eNB_prach_vars   = &phy_vars_eNB->lte_eNB_prach_vars;
  for ( i = 0; i < device_count; i++ )
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    printf("[CUDA] device number= %d, device name= %s\n",i, deviceProp.name);
  }
  
  for ( Msc_RS = 0; Msc_RS < 2; Msc_RS++ )
  {
    for ( u = 0; u < 30; u++ )
	{
	  for ( v = 0; v < 1; v++ )
	  {
	    cudaMalloc( ( void **)&esti_const.d_ul_ref_sigs_rx[u][v][Msc_RS], 2*sizeof( float2 )*dftsizes[Msc_RS] );
		generate_ul_ref_sigs_rx_cu<<< 1, 1>>>( esti_const.d_ul_ref_sigs_rx[u][v][Msc_RS], Msc_RS, u, v );
	  }
	}
  }
  
  for ( Msc_RS = 2; Msc_RS < 33; Msc_RS++ )
  {
    for ( u = 0; u < 30; u++ )
	{
	  for ( v = 0; v < 2; v++ )
	  {
	    cudaMalloc( ( void **)&esti_const.d_ul_ref_sigs_rx[u][v][Msc_RS], 2*sizeof( float2 )*dftsizes[Msc_RS] );
		generate_ul_ref_sigs_rx_cu<<< 1, 1>>>( esti_const.d_ul_ref_sigs_rx[u][v][Msc_RS], Msc_RS, u, v );
	  }
	}
  }
  
  //host mem alloc
/*
  int eNB_id, UE_id;
  for ( eNB_id = 0; eNB_id < 3; eNB_id++ )
  {
    printf("Initial host port to device port\n");
	printf("Initial RX port\n");
    cudaMallocHost((void **) &eNB_common_vars->rxdata_7_5kHz[eNB_id],frame_parm->nb_antennas_rx*sizeof(int*));
    cudaMallocHost((void **) &eNB_common_vars->rxdataF[eNB_id], frame_parm->nb_antennas_rx*sizeof(int*));
	for ( i = 0; i < frame_parms.nb_antennas_rx; i++ )
	{
      cudaMallocHost((void **)&eNB_common_vars->rxdata_7_5kHz[eNB_id][i], frame_parm->samples_per_tti*sizeof(int));      
      cudaMallocHost((void **)&eNB_common_vars->rxdataF[eNB_id][i], 2*sizeof(int)*(frame_parm->ofdm_symbol_size*frame_parm->symbols_per_tti)  );      
    }
	printf("Initial TX port\n");
	cudaMallocHost((void **)eNB_common_vars->txdataF[eNB_id], frame_parm->nb_antennas_tx*sizeof(int*));
	for ( i = 0; i < frame_parms.nb_antennas_rx; i++ )
	{
	  cudaMallocHost((void **)&eNB_common_vars->txdataF[eNB_id][i], 2*(frame_parm->ofdm_symbol_size*frame_parm->symbols_per_tti)*sizeof(int) );
	}
  }

  for ( UE_id = 0; UE_id < NUMBER_OF_UE_MAX; UE_id++ )
  {
	for ( eNB_id = 0; eNB_id < 3; eNB_id++ )
    {
	  cudaMallocHost((void **) &eNB_pusch_vars[UE_id]->rxdataF_comp[eNB_id], frame_parm->nb_antennas_rx*sizeof(int*));
	  for ( i = 0; i < frame_parms.nb_antennas_rx; i++ )
	  {
	    cudaMallocHost((void **)&eNB_pusch_vars[UE_id]->rxdataF_comp[eNB_id][i], sizeof(int)*frame_parm->N_RB_UL*12*frame_parm->symbols_per_tti  );
	  }
	}
  }
*/
  for ( i = 0; i < 10; i++ )
  {
    ul_cu[i].CP = frame_parms.nb_prefix_samples;
    ul_cu[i].CP0= frame_parms.nb_prefix_samples0;
	
	ul_cu[i].fftsize = frame_parms.ofdm_symbol_size;
	ul_cu[i].Ncp = frame_parms.Ncp;
	ul_cu[i].symbols_per_tti         = frame_parms.symbols_per_tti;
	ul_cu[i].samples_per_tti         = frame_parms.samples_per_tti;
	ul_cu[i].nb_antennas_rx          = frame_parms.nb_antennas_rx;
	ul_cu[i].N_RB_UL                 = frame_parms.N_RB_UL;
	
	ul_cu[i].d_rxdata                = ( int **)malloc( frame_parms.nb_antennas_rx * sizeof( int *) );
	ul_cu[i].d_rxdata_fft            = ( float2 **)malloc( frame_parms.nb_antennas_rx * sizeof( float2 *) );
	ul_cu[i].d_rxdataF               = ( int **)malloc( frame_parms.nb_antennas_rx * sizeof( int *) );
	ul_cu[i].d_rxdata_ext            = ( float2 **)malloc( frame_parms.nb_antennas_rx * sizeof( float2 *) );
	ul_cu[i].d_rxdata_ext_int        = ( int **)malloc( frame_parms.nb_antennas_rx * sizeof( int *) );
	ul_cu[i].d_rxdata_comp           = ( float2 **)malloc( frame_parms.nb_antennas_rx * sizeof( float2 *) );
	ul_cu[i].d_rxdata_comp_int       = ( int **)malloc( frame_parms.nb_antennas_rx * sizeof( int *) );
	ul_cu[i].d_drs_ch                = ( float2 **)malloc( frame_parms.nb_antennas_rx * sizeof( float2 *) );
	ul_cu[i].d_drs_ch_int            = ( int **)malloc( frame_parms.nb_antennas_rx * sizeof( int *) );
	ul_cu[i].d_ulsch_power           = ( int **)malloc( frame_parms.nb_antennas_rx * sizeof( int *) );
	for ( k = 0; k < ul_cu[i].nb_antennas_rx; k++ )
	{
	  if(cudaMalloc(( void **)&ul_cu[i].d_rxdata[k]         , sizeof( int )* 15* 512))
            printf("error\n");
	  cudaMalloc(( void **)&ul_cu[i].d_rxdata_fft[k]     , sizeof( float2 )* ul_cu[i].symbols_per_tti* ul_cu[i].fftsize);
	  cudaMalloc(( void **)&ul_cu[i].d_rxdataF[k]        , 2* sizeof( int )* ul_cu[i].symbols_per_tti* ul_cu[i].fftsize );
	  cudaMalloc(( void **)&ul_cu[i].d_rxdata_ext[k]     , sizeof( float2 )* frame_parms.N_RB_UL* 12* frame_parms.symbols_per_tti );
	  cudaMalloc(( void **)&ul_cu[i].d_rxdata_ext_int[k] , sizeof( int )* frame_parms.N_RB_UL* 12* frame_parms.symbols_per_tti );
	  cudaMalloc(( void **)&ul_cu[i].d_rxdata_comp[k]    , sizeof( float2 )* frame_parms.N_RB_UL* 12* frame_parms.symbols_per_tti );
	  cudaMalloc(( void **)&ul_cu[i].d_rxdata_comp_int[k], sizeof( int )* frame_parms.N_RB_UL* 12* frame_parms.symbols_per_tti );
	  cudaMalloc(( void **)&ul_cu[i].d_drs_ch[k]         , sizeof( float2 )* frame_parms.N_RB_UL* 12* frame_parms.symbols_per_tti );
	  cudaMalloc(( void **)&ul_cu[i].d_drs_ch_int[k]     , sizeof( int )* frame_parms.N_RB_UL* 12* frame_parms.symbols_per_tti + 1 );
	  cudaMalloc(( void **)&ul_cu[i].d_ulsch_power[k]    , sizeof( int ) );
	}
	
    dl_cu[i].CP = frame_parms.nb_prefix_samples;
    dl_cu[i].CP0= frame_parms.nb_prefix_samples0;
	
	dl_cu[i].ifftsize = frame_parms.ofdm_symbol_size;
	dl_cu[i].Ncp = frame_parms.Ncp;
	dl_cu[i].symbols_per_tti = frame_parms.symbols_per_tti;
	dl_cu[i].samples_per_tti = frame_parms.samples_per_tti;
  }
  printf("[CUDA] CP0=%d, CP=%d, fftsize=%d, symbols_per_tti=%d, samples_per_tti=%d\n",ul_cu[i].CP0,ul_cu[i].CP,frame_parms.ofdm_symbol_size,frame_parms.symbols_per_tti,frame_parms.samples_per_tti);

  init_cufft( );
  
}

void init_cufft( void )
{
	//initial cufft plan fft128, fft256, fft512, fft1024, fft1536, fft2048
  int i,j;
  short fftsize = ul_cu[i].fftsize; 
  short Ncp = ul_cu[i].Ncp; 
  short symbols_per_tti = ul_cu[i].symbols_per_tti; 
  short samples_per_tti = ul_cu[i].samples_per_tti;
  for ( i = 0; i < 10; i++ )
  {
  //for ul cuda
    cudaStreamCreateWithFlags( &( ul_cu[i].stream_ul ), cudaStreamNonBlocking );
	cudaStreamCreateWithFlags( &( ul_cu[i].tempstrm ), cudaStreamNonBlocking );
	cufftPlan1d( &( ul_cu[i].fft ) , fftsize ,CUFFT_C2C, symbols_per_tti);
	cufftSetStream( ul_cu[i].fft , ul_cu[i].stream_ul );
	cudaStreamCreateWithFlags( &( ul_cu[i].timing_advance ), cudaStreamNonBlocking );
	
	cufftPlan1d( &( ul_cu[i].ifft_timing_advance ) , fftsize ,CUFFT_C2C, symbols_per_tti);
	cufftSetStream( ul_cu[i].ifft_timing_advance , ul_cu[i].timing_advance );
  //for dl cuda
    cudaStreamCreateWithFlags( &( dl_cu[i].stream_dl ), cudaStreamNonBlocking );
	cufftPlan1d( &( dl_cu[i].ifft ) , fftsize ,CUFFT_C2C, symbols_per_tti);
	cudaMalloc((void **)&(dl_cu[i].d_txdata)     , sizeof( short )*(symbols_per_tti+1)* 2* symbols_per_tti*fftsize);
    cudaMalloc((void **)&(dl_cu[i].d_txdata_o)   , sizeof( short )* samples_per_tti* 2 );
    cudaMalloc((void **)&(dl_cu[i].d_txdata_ifft), sizeof( float2 )* symbols_per_tti* fftsize);
	cudaMallocHost((void **)&(dl_cu[i].h_txdata) , sizeof( short )* symbols_per_tti* 2* fftsize);  
    cufftSetStream( dl_cu[i].ifft , dl_cu[i].stream_dl );	
  }
  for ( i = 0; i < 10; i++ )
  {
    cufftPlan1d( &( ul_cu[i].idft.fft12  ) , 12  ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft24  ) , 24  ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft36  ) , 36  ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft48  ) , 48  ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft60  ) , 60  ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft72  ) , 72  ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft84  ) , 84  ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft96  ) , 96  ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft108 ) , 108 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft120 ) , 120 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft132 ) , 132 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft144 ) , 144 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft156 ) , 156 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft168 ) , 168 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft180 ) , 180 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft192 ) , 192 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft204 ) , 204 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft216 ) , 216 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft228 ) , 228 ,CUFFT_C2C, 14 );
	cufftPlan1d( &( ul_cu[i].idft.fft240 ) , 240 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft252 ) , 252 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft264 ) , 264 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft276 ) , 276 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft288 ) , 288 ,CUFFT_C2C, 14 );
    cufftPlan1d( &( ul_cu[i].idft.fft300 ) , 300 ,CUFFT_C2C, 14 );
	
	cufftSetStream( ul_cu[i].idft.fft12  , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft24  , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft36  , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft48  , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft60  , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft72  , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft84  , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft96  , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft108 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft120 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft132 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft144 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft156 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft168 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft180 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft192 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft204 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft216 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft228 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft240 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft252 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft264 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft276 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft288 , ul_cu[i].stream_ul );
	cufftSetStream( ul_cu[i].idft.fft300 , ul_cu[i].stream_ul );
  }
}

void free_cufft(void)
{
  int i, j;
  for ( i = 0; i < 10; i++ )
  {
  //for ul cuda
    cudaFree(ul_cu[i].d_rxdata);
	cudaFree(ul_cu[i].d_rxdata_fft);
	cufftDestroy(ul_cu[i].fft);
	cudaStreamDestroy(ul_cu[i].stream_ul);
  //for dl cuda
    cudaFree(dl_cu[i].d_txdata);
    cudaFree(dl_cu[i].d_txdata_o);
    cudaFree(dl_cu[i].d_txdata_ifft);
    cudaFreeHost(dl_cu[i].h_txdata);
	cufftDestroy(dl_cu[i].ifft);
	cudaStreamDestroy(dl_cu[i].stream_dl);
  }

  cudaDeviceReset();
  printf("end cuda\n");
}












