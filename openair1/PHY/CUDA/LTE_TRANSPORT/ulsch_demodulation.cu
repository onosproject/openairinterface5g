#include "defs.h"
#include "PHY/CUDA/extern.h"
#ifndef CUFFT_H
#define CUFFT_H
#include "cufft.h"
#endif

#define ccmax(a,b)  ((a>b) ? (a) : (b))
#define ccmin(a,b)  ((a<b) ? (a) : (b))

__global__ void k_short_12( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_24( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_36( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_48( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_60( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_72( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_96( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_108( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_120( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_144( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_180( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_192( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_216( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_240( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_288( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);
__global__ void k_short_300( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb);

__global__ void exrb_compen_esti( float2 *x,
                                  float2 *ul_ref1,
							      float2 *ul_ref2,
							      float2 *out,
								  int  *sig_engery,
								  const unsigned int first_rb,
							      short cyclic_shift1,
							      short cyclic_shift2,
							      short Msc_RS)
{
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  const int tidy = threadIdx.y;
  float2 rxdataF_ext;
  float2 pilot_data1;
  float2 pilot_data2;
  float2 drs_ch;
  __shared__ float power[600];
  __shared__ int power1[600];
  int cs,k, channel_level;
  float phase, current_phase1, current_phase2;
  float const_value = 22.627417;
  float2 out_temp;
  float cs_re[12] = { 1, 0.866025,      0.5, 0,     -0.5, -0.866025, -1, -0.866025,      -0.5,  0,       0.5, 0.866025};
  float cs_im[12] = { 0,      0.5, 0.866025, 1, 0.866025,       0.5,  0,      -0.5, -0.866025, -1, -0.866025,     -0.5};
  
  int mag;
  int temp_re, temp_im;
  short inv_ch[257] = {512,256,170,128,102,85,73,64,56,51,46,42,39,36,34,32,30,28,26,25,24,23,22,21,20,19,18,18,17,17,16,16,15,15,14,14,13,13,13,12,12,12,11,11,11,11,10,10,10,10,10,
                       9,9,9,9,9,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,
                       3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                       2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1};
  int const_shift = 0;
  int i;
  unsigned int xcl;
  unsigned char l2;
  
  
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  rxdataF_ext = x[ symbol_id * 512 + pos ];
  pilot_data1 = x[ 1536 + pos ];
  pilot_data2 = x[ 5120 + pos ];
  out_temp.x = (pilot_data1.x * ul_ref1[re_id].x + pilot_data1.y * ul_ref1[re_id].y)*0.0441941738;
  out_temp.y = (pilot_data1.y * ul_ref1[re_id].x - pilot_data1.x * ul_ref1[re_id].y)*0.0441941738;
  cs = ( re_id * cyclic_shift1 )%12;
  pilot_data1.x = out_temp.x * cs_re[cs] + out_temp.y * cs_im[cs];
  pilot_data1.y = out_temp.y * cs_re[cs] - out_temp.x * cs_im[cs];
  out_temp.x = (pilot_data2.x * ul_ref2[re_id].x + pilot_data2.y * ul_ref2[re_id].y)*0.0441941738;
  out_temp.y = (pilot_data2.y * ul_ref2[re_id].x - pilot_data2.x * ul_ref2[re_id].y)*0.0441941738;
  cs = ( re_id * cyclic_shift2 )%12;
  pilot_data2.x = out_temp.x * cs_re[cs] + out_temp.y * cs_im[cs];
  pilot_data2.y = out_temp.y * cs_re[cs] - out_temp.x * cs_im[cs];
  switch ( tidy )
  {
    case 0: power[re_id<<1]     = pilot_data2.x * pilot_data1.x + pilot_data2.y * pilot_data1.y; break;
	case 1: power[(re_id<<1)+1] = pilot_data2.y * pilot_data1.x - pilot_data2.x * pilot_data1.y; break;
  }
  __syncthreads();
  for ( k = Msc_RS>>1; k > 0; k=k>>1 )
  {
	if ( re_id < k )
	  power[( re_id<<1)+ tidy] = power[(re_id<<1)+tidy] + power[((k+re_id)<<1)+tidy];
	__syncthreads();
	if ( k % 2 && re_id == 0 )
	  power[tidy] = power[tidy] + power[((k-1)<<1)+tidy];
	__syncthreads();
  }
   phase = atanf( power[1]/power[0] );
  
  if ( symbol_id != 10 && symbol_id != 3 )
  {
    current_phase1 = (phase/7)*(symbol_id- 3);
	current_phase2 = (phase/7)*(symbol_id- 10);
	drs_ch.x = ((pilot_data1.x * cosf(current_phase1) - pilot_data1.y * sinf(current_phase1)) +
	            (pilot_data2.x * cosf(current_phase2) - pilot_data2.y * sinf(current_phase2)))/2;
	drs_ch.y = ((pilot_data1.y * cosf(current_phase1) + pilot_data1.x * sinf(current_phase1)) +
	            (pilot_data2.y * cosf(current_phase2) + pilot_data2.x * sinf(current_phase2)))/2;
	switch(tidy)
	{
	  case 0: power1[re_id<<1] = ((short)drs_ch.x * (short)drs_ch.x + (short)drs_ch.y * (short)drs_ch.y); break;
	  case 1: power1[(re_id<<1)+1] = ((short)drs_ch.x * (short)drs_ch.x + (short)drs_ch.y * (short)drs_ch.y)>>4; break;
	}
	__syncthreads();
	for ( k = Msc_RS>>1; k > 0; k=k>>1 )	
	{
	  if ( re_id < k )
		power1[(re_id<<1)+tidy] = power1[(re_id<<1)+tidy] + power1[((k+re_id)<<1)+tidy];
      __syncthreads();
	  if ( k % 2 && re_id == 0  )
		power1[tidy] = power1[tidy] + power1[((k-1)<<1)+tidy];
	  __syncthreads();
	}
	xcl = (unsigned int)(power1[0]/(Msc_RS<<1));
	l2=0;

    for (i=0; i<31; i++)
      if ((xcl&(1<<i)) != 0)
        l2 = i+1;
	channel_level = (short)(l2>>1) + 4;
	mag = ((int)(drs_ch.x * drs_ch.x + drs_ch.y * drs_ch.y))>>channel_level;
	mag = ( mag >= 255 )? 255: mag;
	switch ( tidy )
	{
	  case 0: 
	    out[symbol_id*Msc_RS+re_id].x = (float)((((int)(((rxdataF_ext.x * drs_ch.x) + ( rxdataF_ext.y * drs_ch.y ))*0.0441941738))>>channel_level)*inv_ch[mag]);
		break;
	  case 1: 
	    out[symbol_id*Msc_RS+re_id].y = (float)((((int)(((rxdataF_ext.y * drs_ch.x) - ( rxdataF_ext.x * drs_ch.y ))*0.0441941738))>>channel_level)*inv_ch[mag]); 
		break;
	}
	if(tidy == 0 && re_id == 0 && symbol_id == 0)
	  sig_engery[0] = (int)power1[1]*8/Msc_RS;
  }
  else if ( symbol_id == 3 )
  {
	out[symbol_id * 300 + re_id] = pilot_data1;
  }
  else if ( symbol_id == 10 )
  {
	out[symbol_id * 300 + re_id] = pilot_data2;
  }
}
						         


__global__ void exrb( float2 *x,
					  float2 *out,
					  short  *out2,
					  const unsigned int first_rb,
					  const short Msc_RS
                            )
{
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  const int tidy = threadIdx.y;
  float2 rxdataF_ext;
  float2 pilot_data1;
  float2 pilot_data2;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  rxdataF_ext.x = x[ symbol_id * 512 + pos ].x*0.0441941738;
  rxdataF_ext.y = x[ symbol_id * 512 + pos ].y*0.0441941738;
  pilot_data1.x = x[ 1536 + pos ].x*0.0441941738;
  pilot_data1.y = x[ 1536 + pos ].y*0.0441941738;
  pilot_data2.x = x[ 5120 + pos ].x*0.0441941738;
  pilot_data2.y = x[ 5120 + pos ].y*0.0441941738;
  out[symbol_id * 300 + re_id] = rxdataF_ext;
  out2[((symbol_id*300+re_id)<<1)]   = (short)rxdataF_ext.x;
  out2[((symbol_id*300+re_id)<<1)+1] = (short)rxdataF_ext.y;
}

__global__ void estimation( float2 *x,
                            float2 *ul_ref1,
							float2 *ul_ref2,
							float2 *out,
							short  *out2,
							short cyclic_shift1,
							short cyclic_shift2,
							short Msc_RS
						   )
{
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  const int tidy = threadIdx.y;
  int cs,k, channel_level;
  float phase, current_phase1, current_phase2;
  float const_value = 22.627417;
  float2 out_temp;
  float cs_re[12] = { 1, 0.866025,      0.5, 0,     -0.5, -0.866025, -1, -0.866025,      -0.5,  0,       0.5, 0.866025};
  float cs_im[12] = { 0,      0.5, 0.866025, 1, 0.866025,       0.5,  0,      -0.5, -0.866025, -1, -0.866025,     -0.5};
  float2 drs_ch;
  float2 pilot_data1;
  float2 pilot_data2;
  __shared__ float power[600];
  
  pilot_data1 = x[900 + re_id];
  pilot_data2 = x[3000+ re_id];
  out_temp.x = pilot_data1.x * ul_ref1[re_id].x + pilot_data1.y * ul_ref1[re_id].y;
  out_temp.y = pilot_data1.y * ul_ref1[re_id].x - pilot_data1.x * ul_ref1[re_id].y;
  cs = ( re_id * cyclic_shift1 )%12;
  pilot_data1.x = out_temp.x * cs_re[cs] + out_temp.y * cs_im[cs];
  pilot_data1.y = out_temp.y * cs_re[cs] - out_temp.x * cs_im[cs];
  out_temp.x = pilot_data2.x * ul_ref2[re_id].x + pilot_data2.y * ul_ref2[re_id].y;
  out_temp.y = pilot_data2.y * ul_ref2[re_id].x - pilot_data2.x * ul_ref2[re_id].y;
  cs = ( re_id * cyclic_shift2 )%12;
  pilot_data2.x = out_temp.x * cs_re[cs] + out_temp.y * cs_im[cs];
  pilot_data2.y = out_temp.y * cs_re[cs] - out_temp.x * cs_im[cs];
  
  if ( tidy == 0 )
    power[re_id<<1]     = pilot_data2.x * pilot_data1.x + pilot_data2.y * pilot_data1.y;
  else
	power[(re_id<<1)+1] = pilot_data2.y * pilot_data1.x - pilot_data2.x * pilot_data1.y;
  __syncthreads();
  for ( k = Msc_RS>>1; k > 0; k=k>>1 )
  {
	if ( re_id < k )
	  power[( re_id<<1)+ tidy] = power[(re_id<<1)+tidy] + power[((k+re_id)<<1)+tidy];
	__syncthreads();
	if ( k % 2 && re_id == 0 )
	  power[tidy] = power[tidy] + power[((k-1)<<1)+tidy];
	__syncthreads();
  }
   phase = atanf( power[1]/power[0] );
  
  if ( symbol_id != 10 && symbol_id != 3 )
  {
    current_phase1 = (phase/7)*(symbol_id- 3);
	current_phase2 = (phase/7)*(symbol_id- 10);
	drs_ch.x = ((pilot_data1.x * cosf(current_phase1) - pilot_data1.y * sinf(current_phase1)) +
	            (pilot_data2.x * cosf(current_phase2) - pilot_data2.y * sinf(current_phase2)))/2;
	drs_ch.y = ((pilot_data1.y * cosf(current_phase1) + pilot_data1.x * sinf(current_phase1)) +
	            (pilot_data2.y * cosf(current_phase2) + pilot_data2.x * sinf(current_phase2)))/2;
	out[symbol_id*300+re_id] = drs_ch;
	out2[((symbol_id*300+re_id)<<1)]   = (short)drs_ch.x;
	out2[((symbol_id*300+re_id)<<1)+1] = (short)drs_ch.y;
	
  }
  else if ( symbol_id == 3 )
  {
	out[symbol_id * 300 + re_id] = pilot_data1;
	out2[((symbol_id*300+re_id)<<1)]   = (short)pilot_data1.x;
	out2[((symbol_id*300+re_id)<<1)+1] = (short)pilot_data1.y;
  }
  else if ( symbol_id == 10 )
  {
	out[symbol_id * 300 + re_id] = pilot_data2;
	out2[((symbol_id*300+re_id)<<1)]   = (short)pilot_data2.x;
	out2[((symbol_id*300+re_id)<<1)+1] = (short)pilot_data2.y;
  }
}

/*__global__ void add_value( int *x, int *y )
{
  x[14*300] = y[0];
}*/

__global__ void compensation( float2 *x,
                              short  *xt,
                              float2 *drs,
							  short  *drst,
							  float  Qm_const,
							  float2 *out,
							  short  *out2,
							  int  *sig_engery,
							  short  Msc_RS
                            )
{
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  const int tidy = threadIdx.y;
  int k, channel_level,mag;
  float2 out_temp;
  int temp_re, temp_im;
  __shared__ int power[600];
  /*short inv_ch[257] = {512,256,170,128,102,85,73,64,56,51,46,42,39,36,34,32,30,28,26,25,24,23,22,21,20,19,18,18,17,17,16,16,15,15,14,14,13,13,13,12,12,12,11,11,11,11,10,10,10,10,10,
                       9,9,9,9,9,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,
                       3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                       2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1};*/
  float sig_eng,inv_ch;
  float2 rxdataF_ext;
  int const_shift = 0;
  float2 drs_ch;
  int i;
  unsigned int xcl;
  unsigned char l2;
  short clr;
  short cli;
  /*rxdataF_ext.x = xt[(symbol_id * 300 + re_id)<<1];
  rxdataF_ext.y = xt[((symbol_id * 300 + re_id)<<1)+1];
  drs_ch.x = drst[(symbol_id * 300 + re_id)<<1];
  drs_ch.y = drst[((symbol_id * 300 + re_id)<<1)+1];*/
  clr = drst[(symbol_id * 300 + re_id)<<1];
  cli = drst[((symbol_id * 300 + re_id)<<1)+1];
  drs_ch = drs[symbol_id * 300 + re_id];
  rxdataF_ext = x[symbol_id * 300 + re_id];
  
  if ( symbol_id != 10 && symbol_id != 3 )
  {
	  
	switch(tidy)
	{
	  case 0: power[re_id<<1] = (clr * clr + cli * cli); break;
	  case 1: power[(re_id<<1)+1] = (clr * clr + cli * cli)>>4; break;
	}
	//power[re_id] = (clr * clr + cli * cli);
	__syncthreads();
	for ( k = Msc_RS>>1; k > 0; k=k>>1 )	
	{
	  if ( re_id < k )
		power[(re_id<<1)+tidy] = power[(re_id<<1)+tidy] + power[((k+re_id)<<1)+tidy];
      __syncthreads();
	  if ( k % 2 && re_id == 0  )
		power[tidy] = power[tidy] + power[((k-1)<<1)+tidy];
	  __syncthreads();
	}
	//xcl = (unsigned int)(power[0]/(Msc_RS<<1));
	//l2=0;

    //for (i=0; i<31; i++)
      //if ((xcl&(1<<i)) != 0)
        //l2 = i+1;
	//channel_level = (short)(l2>>1) + 4;
	sig_eng = power[1]*8/Msc_RS;
	mag = drs_ch.x * drs_ch.x + drs_ch.y * drs_ch.y;
        inv_ch = 512/(sqrtf(mag)*Qm_const);
	inv_ch = ( inv_ch > 512 )? 512:inv_ch;
        inv_ch = ( inv_ch < 1 )? 1:inv_ch;
	switch ( tidy )
	{
	  case 0: 
	    out[symbol_id*Msc_RS+re_id].x = (((rxdataF_ext.x * drs_ch.x) + ( rxdataF_ext.y * drs_ch.y ))*inv_ch)/sqrtf(mag);
		out2[(symbol_id*300+re_id)<<1] = (short)out[symbol_id*Msc_RS+re_id].x;
		break;
	  case 1: 
	    out[symbol_id*Msc_RS+re_id].y = (((rxdataF_ext.y * drs_ch.x) - ( rxdataF_ext.x * drs_ch.y ))*inv_ch)/sqrtf(mag); 
		out2[((symbol_id*300+re_id)<<1)+1] = (short)out[symbol_id*Msc_RS+re_id].y;
		break;
	}
	if(tidy == 0 && re_id == 0 && symbol_id == 0)
	  sig_engery[0] = (int)sig_eng;
  }
  
}

void exrb_compen_esti_cu( unsigned int first_rb,
                          unsigned int nb_rb,
					      unsigned short number_symbols,
				          unsigned short sf)
{
  dim3 block( number_symbols, 1, 1 );
  dim3 thread( ulsch_para[sf].Msc_RS, 2, 1 );
  //printf("[TEST]using RB = %d\n",nb_rb);
  exrb_compen_esti<<< block, thread, 0, ul_cu[sf].stream_ul>>>
    ( ul_cu[sf].d_rxdata_fft[0],
      esti_const.d_ul_ref_sigs_rx[ulsch_para[sf].u1][ulsch_para[sf].v1][ulsch_para[sf].Msc_RS_idx],
	  esti_const.d_ul_ref_sigs_rx[ulsch_para[sf].u2][ulsch_para[sf].v2][ulsch_para[sf].Msc_RS_idx],
	  ul_cu[sf].d_rxdata_comp[0],
	  ul_cu[sf].d_ulsch_power[0],
	  first_rb,
	  ulsch_para[sf].cyclic_shift1,
	  ulsch_para[sf].cyclic_shift2,
	  ulsch_para[sf].Msc_RS);
int aarx = 0;
switch ( ulsch_para[sf].Msc_RS )
{
  case 12:
    cufftExecC2C( ul_cu[sf].idft.fft12,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_12<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 24:
    cufftExecC2C( ul_cu[sf].idft.fft24,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_24<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 36:
    cufftExecC2C( ul_cu[sf].idft.fft36,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_36<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 48:
    cufftExecC2C( ul_cu[sf].idft.fft48,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_48<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 60:
    cufftExecC2C( ul_cu[sf].idft.fft60,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_60<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 72:
    cufftExecC2C( ul_cu[sf].idft.fft72,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_72<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 96:
    cufftExecC2C( ul_cu[sf].idft.fft96,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_96<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 108:
    cufftExecC2C( ul_cu[sf].idft.fft108,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_108<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 120:
    cufftExecC2C( ul_cu[sf].idft.fft120,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_120<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 144:
    cufftExecC2C( ul_cu[sf].idft.fft144,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_144<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 180:
    cufftExecC2C( ul_cu[sf].idft.fft180,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_180<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 192:
    cufftExecC2C( ul_cu[sf].idft.fft192,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_192<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 216:
    cufftExecC2C( ul_cu[sf].idft.fft216,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_216<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 240:
    cufftExecC2C( ul_cu[sf].idft.fft240,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_240<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 288:
    cufftExecC2C( ul_cu[sf].idft.fft288,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_288<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
    case 300:
      cufftExecC2C( ul_cu[sf].idft.fft300,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
	          (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		      CUFFT_INVERSE);
	k_short_300<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  }  
}



void ulsch_extract_rb_cu( unsigned int first_rb,
                          unsigned int nb_rb,
						  unsigned short number_symbols,
				          unsigned short sf)
{ 
  dim3 thread( nb_rb*12, 2, 1);
  dim3 block( number_symbols, 1, 1);
  exrb<<< block, thread, 0, ul_cu[sf].stream_ul>>>
    ( ul_cu[sf].d_rxdata_fft[0],
	  ul_cu[sf].d_rxdata_ext[0],
	  (short*)ul_cu[sf].d_rxdata_comp_int[0],
	  first_rb,
	  nb_rb*12
	);
	
}

void estimation_cu( unsigned int first_rb,
                    unsigned int nb_rb,
					unsigned short number_symbols,
				    unsigned short sf)
{
  dim3 block( number_symbols, 1, 1 );
  dim3 thread( ulsch_para[sf].Msc_RS, 2, 1 );
  estimation<<< block, thread, 0, ul_cu[sf].stream_ul>>>
	  ( ul_cu[sf].d_rxdata_ext[0], 
		esti_const.d_ul_ref_sigs_rx[ulsch_para[sf].u1][ulsch_para[sf].v1][ulsch_para[sf].Msc_RS_idx],
		esti_const.d_ul_ref_sigs_rx[ulsch_para[sf].u2][ulsch_para[sf].v2][ulsch_para[sf].Msc_RS_idx],
		ul_cu[sf].d_drs_ch[0],
	  (short*)ul_cu[sf].d_drs_ch_int[0],
		ulsch_para[sf].cyclic_shift1,
		ulsch_para[sf].cyclic_shift2,
		ulsch_para[sf].Msc_RS
	  );
}

void compensation_cu( unsigned int first_rb,
                    unsigned int nb_rb,
					unsigned short number_symbols,
					short Qm,
				    unsigned short sf)
{
  float Qm_const;
  dim3 block( number_symbols, 1, 1 );
  dim3 thread( ulsch_para[sf].Msc_RS, 2, 1 );
  //printf("in compensation\n");
  switch(Qm)
  {
    case 2: Qm_const = 1.0; break;
    case 4: Qm_const = 0.632456; break;
  }
  compensation<<< block, thread, 0, ul_cu[sf].stream_ul>>>
  ( ul_cu[sf].d_rxdata_ext[0],
    (short*)ul_cu[sf].d_rxdata_ext_int[0],
    ul_cu[sf].d_drs_ch[0],
	(short*)ul_cu[sf].d_drs_ch_int[0],
	Qm_const,
	ul_cu[sf].d_rxdata_comp[0],
	(short*)ul_cu[sf].d_rxdata_comp_int[0],
	ul_cu[sf].d_ulsch_power[0],
	ulsch_para[sf].Msc_RS
  );
}

void idft_cu(unsigned int first_rb,
             unsigned int nb_rb,
			 unsigned short number_symbols,
			 short cl,
			 unsigned short sf
            )
{
  int aarx = 0;
switch ( ulsch_para[sf].Msc_RS )
{
  case 12:
    cufftExecC2C( ul_cu[sf].idft.fft12,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_12<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 24:
    cufftExecC2C( ul_cu[sf].idft.fft24,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_24<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 36:
    cufftExecC2C( ul_cu[sf].idft.fft36,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_36<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 48:
    cufftExecC2C( ul_cu[sf].idft.fft48,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_48<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 60:
    cufftExecC2C( ul_cu[sf].idft.fft60,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_60<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 72:
    cufftExecC2C( ul_cu[sf].idft.fft72,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_72<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 96:
    cufftExecC2C( ul_cu[sf].idft.fft96,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_96<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 108:
    cufftExecC2C( ul_cu[sf].idft.fft108,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_108<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 120:
    cufftExecC2C( ul_cu[sf].idft.fft120,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_120<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 144:
    cufftExecC2C( ul_cu[sf].idft.fft144,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_144<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 180:
    cufftExecC2C( ul_cu[sf].idft.fft180,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_180<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 192:
    cufftExecC2C( ul_cu[sf].idft.fft192,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_192<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 216:
    cufftExecC2C( ul_cu[sf].idft.fft216,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_216<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 240:
    cufftExecC2C( ul_cu[sf].idft.fft240,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_240<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  case 288:
    cufftExecC2C( ul_cu[sf].idft.fft288,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
		   (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		   CUFFT_INVERSE);
	k_short_288<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
    case 300:
      cufftExecC2C( ul_cu[sf].idft.fft300,
              (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
	          (cufftComplex *) ul_cu[sf].d_rxdata_comp[aarx],
 		      CUFFT_INVERSE);
	k_short_300<<< number_symbols, 300, 0, ul_cu[sf].stream_ul >>>( ul_cu[sf].d_rxdata_comp[aarx], (short*)ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0],ul_cu[sf].d_rxdata_fft[0],first_rb);
	break;
  }
  //add_value<<< 1,1,0,ul_cu[sf].stream_ul >>>(ul_cu[sf].d_rxdata_comp_int[aarx],ul_cu[sf].d_ulsch_power[0]);
}

__global__ void k_short_12( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart12 =  0.28867513459;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 12 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart12);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart12);
  }
}

__global__ void k_short_24( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart24 =  0.2041241452;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 24 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart24);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart24);
  }
}

__global__ void k_short_36( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart36 =  0.16666666667;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 36 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart36);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart36);
  }
}

__global__ void k_short_48( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart48 =  0.144337567297;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 48 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart48);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart48);
  }
}

__global__ void k_short_60( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart60 =  0.12909944487;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 60 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart60);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart60);
  }
}

__global__ void k_short_72( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart72 =  0.117851130198;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 72 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart72);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart72);
  }
}

__global__ void k_short_96( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart96 =  0.102062072616;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 96 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart96);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart96);
  }
}

__global__ void k_short_108( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart108 =  0.096225044865;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 108 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart108);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart108);
  }
}

__global__ void k_short_120( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart120 =  0.0912870929175;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 120 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart120);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart120);
  }
}

__global__ void k_short_144( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart144 =  0.083333333333333;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 144 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart144);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart144);
  }
}

__global__ void k_short_180( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart180 =  0.07453559925;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 180 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart180);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart180);
  }
}

__global__ void k_short_192( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart192 =  0.072168783649;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 192 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart192);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart192);
  }
}

__global__ void k_short_216( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart216 =  0.068041381744;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 216 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart216);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart216);
  }
}

__global__ void k_short_240( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart240 =  0.0645497224368;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 240 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart240);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart240);
  }
}

__global__ void k_short_288( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart288 =  0.0589255651;
  int outi= 300 * blockIdx.x+ threadIdx.x;
  int ini = 288 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000+(re_id<<1)] = sig_eng[0];
        y[6001+(re_id<<1)] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart288);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart288);
  }
}

__global__ void k_short_300( float2 *x, short *y, int *sig_eng, float2 *rxF, unsigned int first_rb)
{
  const float one_per_sqart300 =  0.057735026919;
  const int outi= 300 * blockIdx.x + threadIdx.x;
  const int ini = 300 * blockIdx.x + threadIdx.x;
  const int symbol_id = blockIdx.x;
  const int re_id = threadIdx.x;
  int pos = (362 + first_rb * 12 + re_id)%512;
  pos = ( pos >= 150 && pos < 362)? pos+212: pos;
  switch ( symbol_id )
  {
	case 3:
	  y[1800+(re_id<<1)] = rxF[ 3584 + pos ].x*0.0441941738;
	  y[1801+(re_id<<1)] = rxF[ 3584 + pos ].y*0.0441941738;
	  break;
	case 10:
	  if ( re_id == 0 )
	  {
	    y[6000] = sig_eng[0];
        y[6001] = sig_eng[1];
	  }
	  break;
	default:
	  y[outi<<1] =   ( short )(x[ini].x*one_per_sqart300);
      y[(outi<<1)+1]=( short )(x[ini].y*one_per_sqart300);
  }
}




