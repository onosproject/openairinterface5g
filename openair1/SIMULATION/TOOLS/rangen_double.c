/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 *-------------------------------------------------------------------------------
 * Optimization using SIMD instructions
 * Frecuency Domain Analysis
 * Luis Felipe Ariza Vesga, email:lfarizav@unal.edu.co
 * Functions: SHR3, UNI, NOR, nfix(), table_nor(), Ziggurat() -->"The Ziggurat-
 * Method for Generating Random Variables", G. Marsaglia, W. W. Tsang.
 * Functions: SHR3_SSE, UNI_SSE, NOR_SSE, nfix_SSE() --> SSE versions of Ziggurat Method.
 * Functions: boxmuller_SSE_float () --> Box-Muller pseudo-random normal-
 * number generation modificated version from Miloyip and Cephes sources.
 * More info https://github.com/miloyip/normaldist-benchmark.
 *-------------------------------------------------------------------------------
 */

#ifdef USER_MODE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#endif

#include  "defs.h"
#include "PHY/sse_intrin.h"

static unsigned int seed, iy, ir[98];

/*
@defgroup _uniformdouble
@ingroup numerical Uniform linear congruential random number generator.
*/

/*!\brief Initialization routine for Uniform/Gaussian random number generators. */

#define a 1664525lu
#define mod 4294967296.0                /* is 2**32 */

#if 1
void randominit(unsigned seed_init)
{
  int i;
  // this need to be integrated with the existing rng, like taus: navid
  msg("Initializing random number generator, seed %x\n",seed_init);

  if (seed_init == 0) {
    srand((unsigned)time(NULL));

    seed = (unsigned int) rand();
  } else {
    seed = seed_init;
  }

  if (seed % 2 == 0) seed += 1; /* seed and mod are relative prime */

  for (i=1; i<=97; i++) {
    seed = a*seed;                 /* mod 2**32  */
    ir[i]= seed;                   /* initialize the shuffle table    */
  }
 iy=1;
}
#endif

/*!\brief Uniform linear congruential random number generator on \f$[0,1)\f$.  Returns a double-precision floating-point number.*/

double uniformrandom(void)
{
  int j;
  j=1 + 97.0*iy/mod;
  iy=ir[j];
  seed = a*seed;                          /* mod 2**32 */
  ir[j] = seed;
  return( (double) iy/mod );
}

/*!\brief Ziggurat random number generator based on rejection sampling. It returns a pseudo-random normally distributed number with x=0 and variance=1*/
static double wn[128],fn[128];
static uint32_t iz,jz,jsr=123456789,kn[128];
static int32_t hz;

#define SHR3 (jz=jsr, jsr^=(jsr<<13),jsr^=(jsr>>17),jsr^=(jsr<<5),jz+jsr)
#define UNI (0.5+(signed) SHR3 * 0.2328306e-9)
#define NOR (hz=SHR3,iz=(hz&127),(abs(hz)<kn[iz])? hz*wn[iz] : nfix())

double nfix(void)
{
  const double r = 3.442620; 
  static double x, y;
  for (;;)
  {
      x=hz *  wn[iz];
      if (iz==0)
      {   
        do
        {
          x = - 0.2904764 * log (UNI);
          y = - log (UNI);
	} 
        while (y+y < x*x);
        return (hz>0)? r+x : -r-x;
      }
      if (fn[iz]+UNI*(fn[iz-1]-fn[iz])<exp(-0.5*x*x)){
        return x;
      }
      hz = SHR3;
      iz = hz&127;
      if (abs(hz) < kn[iz]){
        return ((hz)*wn[iz]);
      }
  }
}

static uint32_t jsr4[4] __attribute__((aligned(16))) = {123456789,112548569,985584512,452236879};//This initialization depends on the seed for nor_table function in oaisim_functions.c file.
static uint32_t iz4[4] __attribute__((aligned(16)));
static uint32_t iz1[4] __attribute__((aligned(16)));
static uint32_t iz2[4] __attribute__((aligned(16)));
//static float out[4] __attribute__((aligned(16)));
//static int32_t ifabs4[4] __attribute__((aligned(16)));
static int32_t hz4[4] __attribute__((aligned(16)));
static int32_t hz1[4] __attribute__((aligned(16)));
static int32_t hz2[4] __attribute__((aligned(16)));
static int32_t abshz[4] __attribute__((aligned(16)));
static int32_t abshz1[4] __attribute__((aligned(16)));
static int32_t abshz2[4] __attribute__((aligned(16)));
static __m128i jsr_128 __attribute__((aligned(16)));
static __m128i jz_128 __attribute__((aligned(16)));
static __m128i hz_128 __attribute__((aligned(16)));
static __m128i hz1_128 __attribute__((aligned(16)));
static __m128i hz2_128 __attribute__((aligned(16)));
static __m128i abs_hz_128 __attribute__((aligned(16)));
static __m128i abs_hz1_128 __attribute__((aligned(16)));
static __m128i abs_hz2_128 __attribute__((aligned(16)));
static __m128i iz_128 __attribute__((aligned(16)));
static __m128i iz1_128 __attribute__((aligned(16)));
static __m128i iz2_128 __attribute__((aligned(16)));
static __m128i cmplt_option0_128 __attribute__((aligned(16)));
static int count99=0;
static int count0=0;
static int nfix_first_run=0;
static __m128 x __attribute__((aligned(16)));

#define SHR3_SSE (jsr_128=_mm_loadu_si128((__m128i *)jsr4),jz_128=jsr_128, jsr_128=_mm_xor_si128(_mm_slli_epi32(jsr_128,13),jsr_128),jsr_128=_mm_xor_si128(_mm_srli_epi32(jsr_128,17),jsr_128),jsr_128=_mm_xor_si128(_mm_slli_epi32(jsr_128,5),jsr_128),_mm_storeu_si128((__m128i *)jsr4,jsr_128),_mm_add_epi32(jz_128,jsr_128))
#define UNI_SSE (_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.2328306e-9),_mm_cvtepi32_ps(SHR3_SSE)),_mm_set1_ps(0.5)))

#define NOR_SSE (hz_128=SHR3_SSE,_mm_storeu_si128((__m128i *)hz4,hz_128),iz_128=_mm_and_si128(hz_128,_mm_set1_epi32(127)),_mm_storeu_si128((__m128i *)iz4,iz_128),abs_hz_128=_mm_and_si128(hz_128, _mm_set1_epi32(~0x80000000)),cmplt_option0_128 = _mm_cmplt_epi32(abs_hz_128,_mm_setr_epi32(kn[iz4[0]],kn[iz4[1]],kn[iz4[2]],kn[iz4[3]])),count99=(count99>99)?0:count99+4,nfix_first_run=(count99>99)?0:1,(_mm_testc_si128(cmplt_option0_128,_mm_setr_epi32(0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF)))?_mm_mul_ps(_mm_cvtepi32_ps(hz_128),_mm_setr_ps(wn[iz4[0]],wn[iz4[1]],wn[iz4[2]],wn[iz4[3]])):nfix_SSE(iz_128))

//#define NOR1_SSE (hz1_128=SHR3_SSE,_mm_storeu_si128((__m128i *)hz1,hz1_128),iz1_128=_mm_and_si128(hz1_128,_mm_set1_epi32(127)),_mm_storeu_si128((__m128i *)iz1,iz1_128),abs_hz1_128=_mm_and_si128(hz1_128, _mm_set1_epi32(~0x80000000)),_mm_storeu_si128((__m128i *)abshz1,abs_hz1_128))

//#define NOR2_SSE (hz2_128=SHR3_SSE,_mm_storeu_si128((__m128i *)hz2,hz2_128),iz2_128=_mm_and_si128(hz2_128,_mm_set1_epi32(127)),_mm_storeu_si128((__m128i *)iz2,iz2_128),abs_hz2_128=_mm_and_si128(hz2_128, _mm_set1_epi32(~0x80000000)),_mm_storeu_si128((__m128i *)abshz2,abs_hz2_128))

//#define NOR_SSE (hz_128=SHR3_SSE,_mm_storeu_si128((__m128i *)hz4,hz_128),iz_128=_mm_and_si128(hz_128,_mm_set1_epi32(127)),_mm_storeu_si128((__m128i *)iz4,iz_128),abs_hz_128=_mm_and_si128(hz_128, _mm_set1_epi32(~0x80000000)),_mm_storeu_si128((__m128i *)abshz4,abs_hz_128),cmplt_option0_128 = _mm_cmplt_epi32(abs_hz_128,_mm_setr_epi32(kn[iz4[0]],kn[iz4[1]],kn[iz4[2]],kn[iz4[3]])),_mm_storeu_si128((__m128i *)cmplt_option0,cmplt_option0_128),count0=0,(cmplt_option0[0]==0xFFFFFFFF)?count99+=count0++:count0,(cmplt_option0[1]==0xFFFFFFFF)?count99+=count0++:count0,(cmplt_option0[2]==0xFFFFFFFF)?count99+=count0++:count0,(cmplt_option0[3]==0xFFFFFFFF)?count99+=count0++:count0,(cmplt_option0[0]==0xFFFFFFFF && cmplt_option0[1]==0xFFFFFFFF && cmplt_option0[2]==0xFFFFFFFF && cmplt_option0[3]==0xFFFFFFFF && count99<95 && count0==4)?_mm_mul_ps(_mm_cvtepi32_ps(hz_128),_mm_setr_ps(wn[iz4[0]],wn[iz4[1]],wn[iz4[2]],wn[iz4[3]])):nfix_SSE())

//,ifabs=_mm_cmplt_epi32(_mm_max_epi32(_mm_sub_epi32(_mm_setzero_si128(),hz_128),hz_128),_mm_setr_epi32(kn[iz4[0]],kn[iz4[1]],kn[iz4[2]],kn[iz4[3]])),_mm_storeu_si128((__m128i *)ifabs4,ifabs),abs_hz_128=_mm_and_si128(hz_128, _mm_set1_epi32(~0x80000000)),_mm_storeu_si128((__m128i *)abshz4,abs_hz_128),printf("abs_hz_128 %d,%d,%d,%d\n",abshz4[0],abshz4[1],abshz4[2],abshz4[3]),printf("kn %d,%d,%d,%d\n",kn[iz4[0]],kn[iz4[1]],kn[iz4[2]],kn[iz4[3]]),printf("ifabs %x,%x,%x,%x\n",ifabs4[0],ifabs4[1],ifabs4[2],ifabs4[3]),x128=_mm_and_ps(_mm_cvtepi32_ps(_mm_cmplt_epi32(_mm_max_epi32(_mm_sub_epi32(_mm_setzero_si128(),hz_128),hz_128),_mm_setr_epi32(kn[iz4[0]],kn[iz4[1]],kn[iz4[2]],kn[iz4[3]]))),_mm_mul_ps(_mm_cvtepi32_ps(hz_128),_mm_setr_ps(wn[iz4[0]],wn[iz4[1]],wn[iz4[2]],wn[iz4[3]]))),printf("x128 %e,%e,%e,%e\n",x128[0],x128[1],x128[2],x128[3]),printf("iz %d,%d,%d,%d\n",iz4[0],iz4[1],iz4[2],iz4[3]),printf("wn*hz %e,%e,%e,%e\n",hz4[0]*wn[iz4[0]],hz4[1]*wn[iz4[1]],hz4[2]*wn[iz4[2]],hz4[3]*wn[iz4[3]]))

//,_mm_storeu_si128(ssh3_sse4,hz_128),printf("ssh3_sse4 %lu,%lu,%lu,%lu\n",ssh3_sse4[0],ssh3_sse4[1],ssh3_sse4[2],ssh3_sse4[3])
//#define NOR (hz=SHR3, printf("hz %d\n",hz),sign=(hz&128)>>7,printf("sign %s\n",(sign)?"-":"+"),iz=hz&127,printf("iz %d\n",iz), (abs(hz)<kn[iz])? (sign)?(-1)*hz*wn[iz]:hz*wn[iz] : (sign)?(-1)*nfix():nfix())

__m128 nfix_SSE(__m128i iz)
{
  __m128 y __attribute__((aligned(16)));
  __m128i cmplt_option1_128 __attribute__((aligned(16)));
  __m128i cmplt_option2_128 __attribute__((aligned(16)));
  int32_t cmplt_option0[4] __attribute__((aligned(16)));
  int32_t cmplt_option1[4] __attribute__((aligned(16)));
  int32_t cmplt_option2[4] __attribute__((aligned(16)));
  float output[12] __attribute__((aligned(16)));
  float x4_option0[4] __attribute__((aligned(16)));
  float x4[4] __attribute__((aligned(16)));
  int i;
  static float r = 3.442620; 
  uint32_t iz4_i[4] __attribute__((aligned(16))) ;

    //x=hz *  wn[iz];
    _mm_storeu_si128((__m128i *)iz4_i,iz_128);
    _mm_storeu_si128((__m128i *)cmplt_option0,cmplt_option0_128);
    _mm_storeu_ps(x4_option0,_mm_mul_ps(_mm_cvtepi32_ps(hz_128),_mm_setr_ps(wn[iz4[0]],wn[iz4[1]],wn[iz4[2]],wn[iz4[3]])));
    count0=0;
    for (i=0;i<4;i++)
    {
	    if (cmplt_option0[i]==0xFFFFFFFF)
	    {
		output[count0]=hz4[i]*wn[iz4_i[i]];
		count0++;
	    }  
    }
    if ((iz4_i[0]==0||iz4_i[1]==0||iz4_i[2]==0||iz4_i[3]==0)&&nfix_first_run==0&&count0>0)
    {
		nfix_first_run=1;
		do
		{
		    //x = - 0.2904764 * log (UNI);
		    x = _mm_mul_ps(_mm_set1_ps(-0.2904764f), log_ps(UNI_SSE));
		    _mm_storeu_ps(x4,x);
		    //y = - log (UNI);
		    y = _mm_mul_ps(_mm_set1_ps(-1.0f), log_ps(UNI_SSE));
		   //(y+y < x*x)?
		    cmplt_option1_128 = _mm_cvtps_epi32(_mm_cmplt_ps(_mm_add_ps(y,y),_mm_mul_ps(x,x)));
		    _mm_storeu_si128((__m128i *)cmplt_option1,cmplt_option1_128);
		    for (i=0;i<4;i++)
		    {
			    if (cmplt_option1[i]==0x80000000)
			    {
				output[3]=(hz4[i]>0)? x4[i]+r:-x4[i]-r;
			        break;
			    }  
		    }
		}
		while (cmplt_option1[0]!=0x80000000 && cmplt_option1[1]!=0x80000000 && cmplt_option1[2]!=0x80000000 && cmplt_option1[3]!=0x80000000);
		//return _mm_setr_ps(output[0],output[1],output[2],output[3]);	   
    }
    else if (iz4_i[0]>0&&iz4_i[1]>0&&iz4_i[2]>0&&iz4_i[3]>0&&nfix_first_run==0&&count0>0)
    {
        nfix_first_run=1;
	cmplt_option2_128 = _mm_cvtps_epi32(_mm_cmplt_ps(_mm_add_ps(_mm_setr_ps(fn[iz4_i[0]],fn[iz4_i[1]],fn[iz4_i[2]],fn[iz4_i[3]]),_mm_mul_ps(UNI_SSE,_mm_sub_ps(_mm_setr_ps(fn[iz4_i[0]-1],fn[iz4_i[1]-1],fn[iz4_i[2]-1],fn[iz4_i[3]-1]),_mm_setr_ps(fn[iz4_i[0]],fn[iz4_i[1]],fn[iz4_i[2]],fn[iz4_i[3]])))),exp_ps(_mm_mul_ps(_mm_mul_ps(x,x),_mm_set1_ps(-0.5f)))));
	_mm_storeu_si128((__m128i *)cmplt_option2,cmplt_option2_128);
	for (i=0;i<4;i++)
	{
		if (cmplt_option2[i]==0x80000000)
		{
			output[3]=x4_option0[i];
			break; 
		} 
	}
	//return _mm_setr_ps(output[0],output[1],output[2],output[3]);
    }
    if (count0==3)
    {
	return _mm_setr_ps(output[0],output[1],output[2],output[3]);	
    }
    else
    {
	    hz_128=SHR3_SSE;
    	    _mm_storeu_si128((__m128i *)hz4,hz_128);
    	    iz_128=_mm_and_si128(hz_128,_mm_set1_epi32(127));
    	    _mm_storeu_si128((__m128i *)iz4,iz_128);
	    abs_hz_128=_mm_and_si128(hz_128, _mm_set1_epi32(~0x80000000));
	    _mm_storeu_si128((__m128i *)iz4_i,iz_128);
    	    _mm_storeu_si128((__m128i *)cmplt_option0,_mm_cmplt_epi32(abs_hz_128,_mm_setr_epi32(kn[iz4_i[0]],kn[iz4_i[1]],kn[iz4_i[2]],kn[iz4_i[3]])));
	    for (i=count0;i<3;i++)
	    {
		    if (cmplt_option0[i]==0xFFFFFFFF)
		    {
			output[count0]=hz4[i]*wn[iz4_i[i]];
			count0++;
		    }  
	    }  
	    return _mm_setr_ps(output[0],output[1],output[2],output[3]);
    }
}

/*!\Procedure to create tables for normal distribution kn,wn and fn. */
void table_nor(unsigned long seed)
{
  jsr=seed;
  double dn = 3.442619855899;
  int i;
  const double m1 = 2147483648.0;
  double q;
  double tn = 3.442619855899;
  const double vn = 9.91256303526217E-03;

  q = vn/exp(-0.5*dn*dn);

  kn[0] = ((dn/q)*m1);
  kn[1] = 0;

  wn[0] =  ( q / m1 );
  wn[127] = ( dn / m1 );

  fn[0] = 1.0;
  fn[127] = ( exp ( - 0.5 * dn * dn ) );
  for ( i = 126; 1 <= i; i-- )
  {
    dn = sqrt (-2.0 * log ( vn/dn + exp(-0.5*dn*dn)));
    kn[i+1] = ((dn / tn)*m1);
    tn = dn;
    fn[i] = (exp (-0.5*dn*dn));
    wn[i] = (dn / m1);
  }

  return;
}
double ziggurat(double mean, double variance)
{
  return NOR;
}
__m128 ziggurat_SSE_float(void)
{
  return   NOR_SSE;
}

void boxmuller_SSE_float(__m128 *data1, __m128 *data2) {
	__m128 twopi = _mm_set1_ps(2.0f * 3.14159265358979323846f);
	__m128 minustwo = _mm_set1_ps(-2.0f);
	__m128 u1_ps,u2_ps;
	__m128 radius,theta,sintheta,costheta;

	u1_ps = UNI_SSE;
	u2_ps = UNI_SSE;
	radius = _mm_sqrt_ps(_mm_mul_ps(minustwo, log_ps(u1_ps)));
	theta = _mm_mul_ps(twopi, u2_ps);
        sincos_ps(theta, &sintheta, &costheta);
	*data1=_mm_mul_ps(radius, costheta);
	*data2=_mm_mul_ps(radius, sintheta);
}
/*
@defgroup _gaussdouble Gaussian random number generator based on modified Box-Muller transformation.
@ingroup numerical
*/

/*!\brief Gaussian random number generator based on modified Box-Muller transformation.Returns a double-precision floating-point number. */

double gaussdouble(double mean, double variance)
{
  static int iset=0;
  static double gset;
  double fac,r,v1,v2;
  static double max=-1000000;
  static double min=1000000;

  if (iset == 0) {
    do {
      v1 = 2.0*UNI-1.0;
      v2 = 2.0*UNI-1.0;
      r = v1*v1+v2*v2;
    }  while (r >= 1.0);
    fac = sqrt(-2.0*log(r)/r);
    gset= v1*fac;
    iset=1;
    return(sqrt(variance)*v2*fac + mean);
  } else {
    iset=0;
    if (max<sqrt(variance)*gset + mean)
	max=sqrt(variance)*gset + mean;
    if (min>sqrt(variance)*gset + mean)
	min=sqrt(variance)*gset + mean;

    return(sqrt(variance)*gset + mean);
  }
}


#ifdef MAIN
main(int argc,char **argv)
{

  int i;

  randominit();

  for (i=0; i<10; i++) {
    printf("%f\n",gaussdouble(0.0,1.0));
  }
}
#endif

/*void uniformrandomSSE(__m128d *d1,__m128d *d2)
{
  int i,j;
  __m128d u_rand128;
  __m128i j128;
  int j_array[]={(int) (1 + 9.103678167e-08*iy_array[0]),(int) (1 + 9.103678167e-08*iy_array[1]),(int) (1 + 9.103678167e-08*iy_array[2]),(int) (1 + 9.103678167e-08*iy_array[3])};
  //j128=_mm_setr_epi32(1 + 391.0*iy_array[0]/mod,1 + 391.0*iy_array[1]/mod,1 + 391.0*iy_array[2]/mod,1 + 391.0*iy_array[3]/mod);
  //j128=_mm_setr_epi32(391,391,391,391);
  //j128=_mm_mul_epu32(j128,_mm_setr_epi32(iy_array[0],iy_array[1],iy_array[2],iy_array[3]));
  //j128=_mm_slli_epi32(j128,32);
  //j=1 + 97.0*iy/mod;
  iy_array[0]=ir[j_array[0]];
  iy_array[1]=ir[j_array[1]];
  iy_array[2]=ir[j_array[2]];
  iy_array[3]=ir[j_array[3]];

  seed_array[0] = a*seed_array[0];                          
  seed_array[1] = a*seed_array[1];                          
  seed_array[2] = a*seed_array[2];                          
  seed_array[3] = a*seed_array[3];                          
  ir[j_array[0]] = seed_array[0];
  ir[j_array[1]] = seed_array[1];
  ir[j_array[2]] = seed_array[2];
  ir[j_array[3]] = seed_array[3];
  *d1=_mm_setr_pd (2*((double) iy_array[0]/mod)-1, 2*((double) iy_array[1])-1);
  *d2=_mm_setr_pd (2*((double) iy_array[2]/mod)-1, 2*((double) iy_array[3])-1);
  //return ((double) iy/mod );
  return;

}*/

/*static void gaussfloat_sse2(float* data, size_t count) {
	//assert(count % 8 == 0);
	//LCG<__m128> r;
	for (int i = 0; i < count; i += 8) {
        __m128 u1 = _mm_sub_ps(_mm_set1_ps(1.0f), r()); // [0, 1) -> (0, 1]
        __m128 u2 = r();
		__m128 radius = _mm_sqrt_ps(_mm_mul_ps(_mm_set1_ps(-2.0f), log_ps(u1)));
		__m128 theta = _mm_mul_ps(_mm_set1_ps(2.0f * 3.14159265358979323846f), u2);
        __m128 sintheta, costheta;
        sincos_ps(theta, &sintheta, &costheta);
		_mm_store_ps(&data[i    ], _mm_mul_ps(radius, costheta));
		_mm_store_ps(&data[i + 4], _mm_mul_ps(radius, sintheta));
	}
}*/
/*#define randominit_SSE
#ifdef randominit_SSE
void randominit(unsigned seed_init)
{
  int i;
  // this need to be integrated with the existing rng, like taus: navid
  msg("Initializing random number generator, seed %x\n",seed_init);

  if (seed_init == 0) {
    srand((unsigned)time(NULL));
    seed_array[0] = (unsigned int) rand();
    seed_array[1] = (unsigned int) rand();
    seed_array[2] = (unsigned int) rand();
    seed_array[3] = (unsigned int) rand();
    seed = (unsigned int) rand();
  } else {
    seed = seed_init;
    seed_array[0] = seed_init;
    seed_array[1] = log(seed_init);
    seed_array[2] = pow(seed_init,3);
    seed_array[3] = sqrt(seed_init);
  }
  if (seed % 2 == 0) seed += 1; // seed and mod are relative prime 
  for (i=0;i<4;i++)
  	if (seed_array[i] % 2 == 0) seed_array[i] += 1; // seed and mod are relative prime 

  for (i=1; i<4*98; i++) { //4 times 98 to use in SSE implementations
    seed_array[i%4] = a*seed_array[i%4];                 // mod 2**32  

    ir[i]= seed_array[i%4];                   // initialize the shuffle table    
  }
  for ( i = 0; i < 4*98; i++ ){
	printf("ir[%d]: %d\n",i,ir[i]);
  }
  iy=1;
}*/

//#define jsr4 (jz=jsr, jsr^=(jsr<<13),jsr^=(jsr>>17),jsr^=(jsr<<5),jz+jsr,printf("seed %d, next seed %d\n",jz,jsr))
//#define SHR3_SSE (jsr_128=_mm_loadu_si128(jsr4),jz_128=jsr_128, printf("jsr4 is %lu,%lu,%lu,%lu\n",jsr4[0],jsr4[1],jsr4[2],jsr4[3]), jsr_128=_mm_xor_si128(_mm_slli_epi32(jsr_128,13),jsr_128),_mm_storeu_si128((__m128i *)jsr4,jsr_128),printf("jsr128<<13 is %lu,%lu,%lu,%lu\n",jsr4[0],jsr4[1],jsr4[2],jsr4[3]),      jsr_128=_mm_xor_si128(_mm_srli_epi32(jsr_128,17),jsr_128),_mm_storeu_si128((__m128i *)jsr4,jsr_128),printf("jsr128>>17 is %lu,%lu,%lu,%lu\n",jsr4[0],jsr4[1],jsr4[2],jsr4[3]),      jsr_128=_mm_xor_si128(_mm_slli_epi32(jsr_128,5),jsr_128),_mm_storeu_si128((__m128i *)jsr4,jsr_128), printf("jsr128<<5  is %lu,%lu,%lu,%lu\n",jsr4[0],jsr4[1],jsr4[2],jsr4[3]),      _mm_storeu_si128(out,_mm_add_epi32(jz_128,jsr_128)),printf("out is %lu,%lu,%lu,%lu\n",out[0],out[1],out[2],out[3]),_mm_add_epi32(jz_128,jsr_128))

//#define UNI_SSE (_mm_storeu_ps(out,_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.2328306e-9),_mm_cvtepi32_ps(SHR3_SSE)),_mm_set1_ps(0.5))),printf("out is %e,%e,%e,%e\n",out[0],out[1],out[2],out[3]),_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.2328306e-9),_mm_cvtepi32_ps(SHR3_SSE)),_mm_set1_ps(0.5)))

/*double gaussdouble(double mean, double variance)
{
  static int iset=0;
  static float gset;
  float fac,rn,r[2],v1[2],v2[2],ones[]={1.0,1.0,1.0,1.0};
  static double max=-1000000;
  static double min=1000000;
  __m128 v1_128, v2_128, r128, ones128,compge128_mask;
  ones128 = _mm_load_ps(ones);
  if (iset == 0) {
    do {
      //v2 = 2.0*uniformrandom()-1.0;
      v1_128 = _mm_set1_ps(2*UNI-1);
      v2_128 = _mm_set1_ps(2*UNI-1);
      //r = v1*v1+v2*v2;
      r128= _mm_add_ps(_mm_mul_ps(v1_128,v1_128),_mm_mul_ps(v2_128,v2_128));
      compge128_mask=_mm_cmpge_ps(r128,ones128);
      //_mm_storeu_ps(&r[0],r128);
      //printf("Inside do: r[0] %e, r[1] %e\n",r[0],r[1]);
    }  while (compge128_mask[0] != 4294967295 && compge128_mask[1]!=4294967295 && compge128_mask[2]!=4294967295 && compge128_mask[3]!=4294967295);
    //printf("outside do: r[0] %e, r[1] %e\n",r[0],r[1]);
    if (r[0]<r[1]){
        fac = sqrt(-2.0*log(r[0])/r[0]);
        gset= v1[0]*fac;
        iset=1;
        return(sqrt(variance)*v1[1]*fac + mean);
    }
    else{
        fac = sqrt(-2.0*log(r[1])/r[1]);
        gset= v2[0]*fac;
        iset=1;
        return(sqrt(variance)*v2[1]*fac + mean);
    }
  } else {
    iset=0;
    //printf("normal random number %e, max %e, min %e\n",sqrt(variance)*gset + mean, max,min);
    if (max<sqrt(variance)*gset + mean)
	max=sqrt(variance)*gset + mean;
    if (min>sqrt(variance)*gset + mean)
	min=sqrt(variance)*gset + mean;

    return(sqrt(variance)*gset + mean);
  }
}*/

/*__m128 nfix1_SSE(void)
{
  __m128 x1 __attribute__((aligned(16)));
  __m128 y1 __attribute__((aligned(16)));
  __m128i cmplt_option0_128 __attribute__((aligned(16)));
  __m128i cmplt_option1_128 __attribute__((aligned(16)));
  __m128i cmplt_option2_128 __attribute__((aligned(16)));
  int32_t cmplt_option0[4] __attribute__((aligned(16)));
  int32_t cmplt_option1[4] __attribute__((aligned(16)));
  int32_t cmplt_option2[4] __attribute__((aligned(16)));
  float output1[12] __attribute__((aligned(16)));
  float x1_option0[4] __attribute__((aligned(16)));
  float x4[4] __attribute__((aligned(16)));

  int count0=0;
  int count1=0;
  int count2=0;

  int i;
  static float r = 3.442620; 
  static int nfix_first_run=0;
  for (;;)
  {
    NOR1_SSE;
    //(abs(hz)<kn[iz])? hz*wn[iz]
    cmplt_option0_128 = _mm_cmplt_epi32(abs_hz1_128,_mm_setr_epi32(kn[iz1[0]],kn[iz1[1]],kn[iz1[2]],kn[iz1[3]]));
    _mm_storeu_si128((__m128i *)cmplt_option0,cmplt_option0_128);
    //x=hz *  wn[iz];
    for (i=0;i<4;i++)
    {
	    if (cmplt_option0[i]==0xFFFFFFFF)
	    {
		//printf("count0 %d\n",count0);
		output1[count0]=hz1[i]*wn[iz1[i]];
		count0++;
	    }  
    }

    if (count0>3)
    {
    	count99+=4;
	if (count99>99)
	{
		count99=0;
		nfix_first_run=0;
	}
	return _mm_setr_ps(output1[0],output1[1],output1[2],output1[3]);
    }
    //x=hz *  wn[iz];
    x1=_mm_mul_ps(_mm_cvtepi32_ps(hz1_128),_mm_setr_ps(wn[iz1[0]],wn[iz1[1]],wn[iz1[2]],wn[iz1[3]]));
    _mm_storeu_ps(x1_option0,x1);
    //printf("count0 is %d, count1 is %d, count2 is %d,count99 is %d\n",count0,count1,count2,count99);
    if ((iz1[0]==0||iz1[1]==0||iz1[2]==0||iz1[3]==0)&&nfix_first_run==0&&count0>0)
    {
		//printf("\niz == 0 [%d,%d,%d,%d]\n\n",iz4[0],iz4[1],iz4[2],iz4[3]);
		nfix_first_run=1;
		do
		{
		    //x = - 0.2904764 * log (UNI);
		    x1 = _mm_mul_ps(_mm_set1_ps(-0.2904764f), log_ps(UNI_SSE));
		    _mm_storeu_ps(x4,x1);
		    //y = - log (UNI);
		    y1 = _mm_mul_ps(_mm_set1_ps(-1.0f), log_ps(UNI_SSE));
		   //(y+y < x*x)?
		    cmplt_option1_128 = _mm_cvtps_epi32(_mm_cmplt_ps(_mm_add_ps(y1,y1),_mm_mul_ps(x1,x1)));
		    _mm_storeu_si128((__m128i *)cmplt_option1,cmplt_option1_128);
		    for (i=0;i<4;i++)
		    {
			    if (cmplt_option1[i]==0x80000000)
			    {
				//printf("count22 %d\n",count2);
				output1[3]=(hz1[i]>0)? x4[i]+r:-x4[i]-r;
				count2++;
				break;
			    }  
		    }
		}
		while (cmplt_option1[0]!=0x80000000 || cmplt_option1[1]!=0x80000000 || cmplt_option1[2]!=0x80000000 || cmplt_option1[3]!=0x80000000);
		if (count0+count2>3)
		{
        		count99+=4;
			if (count99>99)
			{
				count99=0;
				nfix_first_run=0;
			}
			return _mm_setr_ps(output1[0],output1[1],output1[2],output1[3]);
	        }
    }
    if (iz1[0]>0&&iz1[1]>0&&iz1[2]>0&&iz1[3]>0&&nfix_first_run==0&&count0>0)
    {
	//printf("\niz > 0 [%d,%d,%d,%d]\n\n",iz4[0],iz4[1],iz4[2],iz4[3]);
	nfix_first_run=1;
	printf("\niz1 > 0 [%d,%d,%d,%d].\nfn [%e,%e,%e,%e].\n\n",iz1[0],iz1[1],iz1[2],iz1[3],fn[iz1[0]],fn[iz1[1]],fn[iz1[2]],fn[iz1[3]]);
	printf("fn1 - 1 [%e,%e,%e,%e]\n",fn[iz1[0]-1],fn[iz1[1]-1],fn[iz1[2]-1],fn[iz1[3]-1]);
	//if (iz==0)
	printf("\niz [%d,%d,%d,%d]\n",iz4[0],iz4[1],iz4[2],iz4[3]);
	printf("iz==0 [%d,%d,%d,%d]\n",iz4[0]==0,iz4[1]==0,iz4[2]==0,iz4[3]==0);
	printf("iz>0 [%d,%d,%d,%d]\n\n",iz4[0]>0,iz4[1]>0,iz4[2]>0,iz4[3]>0);//
        // if (fn[iz]+UNI*(fn[iz-1]-fn[iz])<exp(-0.5*x*x))
	//printf("iz [%d,%d,%d,%d] is ok? %d\n",iz4[0],iz4[1],iz4[2],iz4[3],iz4[0]==0&&iz4[1]==0&&iz4[2]==0&&iz4[3]==0);
	//printf("iz>0 inside [%d,%d,%d,%d]\n",iz4[0]>0,iz4[1]>0,iz4[2]>0,iz4[3]>0);
	//printf("iz-1 [%d,%d,%d,%d]\n",iz4[0]-1,iz4[1]-1,iz4[2]-1,iz4[3]-1);
	//printf("x [%e,%e,%e,%e]\n",x[0],x[1],x[2],x[3]);
	//printf("exp [%e,%e,%e,%e]\n",exp(-0.5*x[0]*x[0]),exp(-0.5*x[1]*x[1]),exp(-0.5*x[2]*x[2]),exp(-0.5*x[3]*x[3]));//
	cmplt_option2_128 = _mm_cvtps_epi32(_mm_cmplt_ps(_mm_add_ps(_mm_setr_ps(fn[iz1[0]],fn[iz1[1]],fn[iz1[2]],fn[iz1[3]]),_mm_mul_ps(UNI_SSE,_mm_sub_ps(_mm_setr_ps(fn[iz1[0]-1],fn[iz1[1]-1],fn[iz1[2]-1],fn[iz1[3]-1]),_mm_setr_ps(fn[iz1[0]],fn[iz1[1]],fn[iz1[2]],fn[iz1[3]])))),exp_ps(_mm_mul_ps(_mm_mul_ps(x1,x1),_mm_set1_ps(-0.5f)))));
	//cmplt_option2_128 = _mm_cvtps_epi32(_mm_cmplt_ps(_mm_setr_ps(fn[iz4[0]],fn[iz4[1]],fn[iz4[2]],fn[iz4[3]]),exp_ps(_mm_mul_ps(_mm_mul_ps(x,x),_mm_set1_ps(-0.5f)))));
	_mm_storeu_si128((__m128i *)cmplt_option2,cmplt_option2_128);
	for (i=0;i<4;i++)
	{
		if (cmplt_option2[i]==0x80000000)
		{
			//printf("count1 %d\n",count1);
			output1[3]=x1_option0[i];
			count1++;
			break; 
		} 
	}
	if (count0+count1>3)
	{
        	count99+=4;
		if (count99>109)
		{
			count99=0;
			nfix_first_run=0;
		}
		return _mm_setr_ps(output1[0],output1[1],output1[2],output1[3]);
	}
    }

    NOR1_SSE;
    //(abs(hz)<kn[iz])? hz*wn[iz]
    cmplt_option0_128 = _mm_cmplt_epi32(abs_hz1_128,_mm_setr_epi32(kn[iz1[0]],kn[iz1[1]],kn[iz1[2]],kn[iz1[3]]));
    _mm_storeu_si128((__m128i *)cmplt_option0,cmplt_option0_128);
    for (i=count0;i<3;i++)
    {
	    if (cmplt_option0[i-count0]==0xFFFFFFFF)
	    {
		//printf("count0 %d\n",count0);
		output1[count0]=hz1[i-count0]*wn[iz1[i-count0]];
		count0++;
	    }  
    }
    count99+=4;
    if (count99>109)
    {
		count99=0;
		nfix_first_run=0;
    }
    return _mm_setr_ps(output1[0],output1[1],output1[2],output1[3]);
  }
}
__m128 nfix2_SSE(void)
{
  __m128 x2 __attribute__((aligned(16)));
  __m128 y2 __attribute__((aligned(16)));
  __m128i cmplt_option0_128 __attribute__((aligned(16)));
  __m128i cmplt_option1_128 __attribute__((aligned(16)));
  __m128i cmplt_option2_128 __attribute__((aligned(16)));
  int32_t cmplt_option0[4] __attribute__((aligned(16)));
  int32_t cmplt_option1[4] __attribute__((aligned(16)));
  int32_t cmplt_option2[4] __attribute__((aligned(16)));
  float output2[12] __attribute__((aligned(16)));
  float x2_option0[4] __attribute__((aligned(16)));
  float x4[4] __attribute__((aligned(16)));

  static int count0=0;
  static int count1=0;
  static int count2=0;
  static int count99=0;
  int i;
  static float r = 3.442620; 
  static int nfix_first_run=0;
  for (;;)
  {
    NOR2_SSE;
    //(abs(hz)<kn[iz])? hz*wn[iz]
    cmplt_option0_128 = _mm_cmplt_epi32(abs_hz2_128,_mm_setr_epi32(kn[iz2[0]],kn[iz2[1]],kn[iz2[2]],kn[iz2[3]]));
    _mm_storeu_si128((__m128i *)cmplt_option0,cmplt_option0_128);
    //x=hz *  wn[iz];
    for (i=0;i<4;i++)
    {
	    if (cmplt_option0[i]==0xFFFFFFFF)
	    {
		//printf("count0 %d\n",count0);
		output2[count0]=hz2[i]*wn[iz2[i]];
		count0++;
	    }  
    }

    if (count0>3)
    {
    	count99+=4;
	if (count99>99)
	{
		count99=0;
		nfix_first_run=0;
	}
	return _mm_setr_ps(output2[0],output2[1],output2[2],output2[3]);
    }
    //x=hz *  wn[iz];
    x2=_mm_mul_ps(_mm_cvtepi32_ps(hz2_128),_mm_setr_ps(wn[iz2[0]],wn[iz2[1]],wn[iz2[2]],wn[iz2[3]]));
    _mm_storeu_ps(x2_option0,x2);
    //printf("count0 is %d, count1 is %d, count2 is %d,count99 is %d\n",count0,count1,count2,count99);
    if ((iz2[0]==0||iz2[1]==0||iz2[2]==0||iz2[3]==0)&&nfix_first_run==0&&count0>0)
    {
		//printf("\niz == 0 [%d,%d,%d,%d]\n\n",iz4[0],iz4[1],iz4[2],iz4[3]);
		nfix_first_run=1;
		do
		{
		    //x = - 0.2904764 * log (UNI);
		    x2 = _mm_mul_ps(_mm_set1_ps(-0.2904764f), log_ps(UNI_SSE));
		    _mm_storeu_ps(x4,x2);
		    //y = - log (UNI);
		    y2 = _mm_mul_ps(_mm_set1_ps(-1.0f), log_ps(UNI_SSE));
		   //(y+y < x*x)?
		    cmplt_option1_128 = _mm_cvtps_epi32(_mm_cmplt_ps(_mm_add_ps(y2,y2),_mm_mul_ps(x2,x2)));
		    _mm_storeu_si128((__m128i *)cmplt_option1,cmplt_option1_128);
		    for (i=0;i<4;i++)
		    {
			    if (cmplt_option1[i]==0x80000000)
			    {
				//printf("count22 %d\n",count2);
				output2[3]=(hz2[i]>0)? x4[i]+r:-x4[i]-r;
				count2++;
				break;
			    }  
		    }
		}
		while (cmplt_option1[0]!=0x80000000 || cmplt_option1[1]!=0x80000000 || cmplt_option1[2]!=0x80000000 || cmplt_option1[3]!=0x80000000);
		if (count0+count2>3)
		{
        		count99+=4;
			if (count99>109)
			{
				count99=0;
				nfix_first_run=0;
			}
			return _mm_setr_ps(output2[0],output2[1],output2[2],output2[3]);
	        }
    }
    if (iz2[0]>0&&iz2[1]>0&&iz2[2]>0&&iz2[3]>0&&nfix_first_run==0&&count0>0)
    {
	//printf("\niz > 0 [%d,%d,%d,%d]\n\n",iz4[0],iz4[1],iz4[2],iz4[3]);
	nfix_first_run=1;
	printf("\niz2 > 0 [%d,%d,%d,%d].\nfn [%e,%e,%e,%e].\n\n",iz2[0],iz2[1],iz2[2],iz2[3],fn[iz2[0]],fn[iz2[1]],fn[iz2[2]],fn[iz2[3]]);
	printf("fn2 - 1 [%e,%e,%e,%e]\n",fn[iz2[0]-1],fn[iz2[1]-1],fn[iz2[2]-1],fn[iz2[3]-1]);
	//if (iz==0)
	printf("\niz [%d,%d,%d,%d]\n",iz4[0],iz4[1],iz4[2],iz4[3]);
	printf("iz==0 [%d,%d,%d,%d]\n",iz4[0]==0,iz4[1]==0,iz4[2]==0,iz4[3]==0);
	printf("iz>0 [%d,%d,%d,%d]\n\n",iz4[0]>0,iz4[1]>0,iz4[2]>0,iz4[3]>0);//
        // if (fn[iz]+UNI*(fn[iz-1]-fn[iz])<exp(-0.5*x*x))
	//printf("iz [%d,%d,%d,%d] is ok? %d\n",iz4[0],iz4[1],iz4[2],iz4[3],iz4[0]==0&&iz4[1]==0&&iz4[2]==0&&iz4[3]==0);
	printf("iz>0 inside [%d,%d,%d,%d]\n",iz4[0]>0,iz4[1]>0,iz4[2]>0,iz4[3]>0);
	printf("iz-1 [%d,%d,%d,%d]\n",iz4[0]-1,iz4[1]-1,iz4[2]-1,iz4[3]-1);
	printf("x [%e,%e,%e,%e]\n",x[0],x[1],x[2],x[3]);
	printf("exp [%e,%e,%e,%e]\n",exp(-0.5*x[0]*x[0]),exp(-0.5*x[1]*x[1]),exp(-0.5*x[2]*x[2]),exp(-0.5*x[3]*x[3]));//
	cmplt_option2_128 = _mm_cvtps_epi32(_mm_cmplt_ps(_mm_add_ps(_mm_setr_ps(fn[iz2[0]],fn[iz2[1]],fn[iz2[2]],fn[iz2[3]]),_mm_mul_ps(UNI_SSE,_mm_sub_ps(_mm_setr_ps(fn[iz2[0]-1],fn[iz2[1]-1],fn[iz2[2]-1],fn[iz2[3]-1]),_mm_setr_ps(fn[iz2[0]],fn[iz2[1]],fn[iz2[2]],fn[iz2[3]])))),exp_ps(_mm_mul_ps(_mm_mul_ps(x2,x2),_mm_set1_ps(-0.5f)))));
	//cmplt_option2_128 = _mm_cvtps_epi32(_mm_cmplt_ps(_mm_setr_ps(fn[iz4[0]],fn[iz4[1]],fn[iz4[2]],fn[iz4[3]]),exp_ps(_mm_mul_ps(_mm_mul_ps(x,x),_mm_set1_ps(-0.5f)))));
	_mm_storeu_si128((__m128i *)cmplt_option2,cmplt_option2_128);
	for (i=0;i<4;i++)
	{
		if (cmplt_option2[i]==0x80000000)
		{
			//printf("count1 %d\n",count1);
			output2[3]=x2_option0[i];
			count1++;
			break; 
		} 
	}
	if (count0+count1>3)
	{
        	count99+=4;
		if (count99>109)
		{
			count99=0;
			nfix_first_run=0;
		}
		return _mm_setr_ps(output2[0],output2[1],output2[2],output2[3]);
	}
    }

    NOR2_SSE;
    //(abs(hz)<kn[iz])? hz*wn[iz]
    cmplt_option0_128 = _mm_cmplt_epi32(abs_hz2_128,_mm_setr_epi32(kn[iz2[0]],kn[iz2[1]],kn[iz2[2]],kn[iz2[3]]));
    _mm_storeu_si128((__m128i *)cmplt_option0,cmplt_option0_128);
    for (i=count0;i<3;i++)
    {
	    if (cmplt_option0[i-count0]==0xFFFFFFFF)
	    {
		//printf("count0 %d\n",count0);
		output2[count0]=hz2[i-count0]*wn[iz2[i-count0]];
		count0++;
	    }  
    }
    count99+=4;
    if (count99>109)
    {
		count99=0;
		nfix_first_run=0;
    }
    return _mm_setr_ps(output2[0],output2[1],output2[2],output2[3]);
  }
}*/

