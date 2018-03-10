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
 */

#ifdef USER_MODE
#include <stdio.h>
#include <stdlib.h>
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
//#define randominit_SSE
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
  if (seed % 2 == 0) seed += 1; /* seed and mod are relative prime */
  for (i=0;i<4;i++)
  	if (seed_array[i] % 2 == 0) seed_array[i] += 1; /* seed and mod are relative prime */

  for (i=1; i<4*98; i++) { /*4 times 98 to use in SSE implementations*/
    seed_array[i%4] = a*seed_array[i%4];                 /* mod 2**32  */

    ir[i]= seed_array[i%4];                   /* initialize the shuffle table    */
  }
  for ( i = 0; i < 4*98; i++ ){
	printf("ir[%d]: %d\n",i,ir[i]);
  }
  iy=1;
}
#else
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
/*!\brief Ziggurat random number generator based on rejection sampling. It returns a pseudorandom normally distributed double number with x=0 and variance=1*/
//Procedure to create tables for normal distribution kn,wn and fn


//#define SHR3 (jz=jsr,printf("jsr %lu ",jsr), jsr^=(jsr<<13),printf("jsr<<13 %lu ",jsr),jsr^=(jsr>>17),printf("jsr>>17 %lu ",jsr),jsr^=(jsr<<5),printf("jsr<<5 %lu\n",jsr),printf("SHR3	jsr %lu, jz %lu, jsr+jz %lu\n",jsr,jz,jz+jsr),jz+jsr)
#define SHR3 (jz=jsr, jsr^=(jsr<<13),jsr^=(jsr>>17),jsr^=(jsr<<5),jz+jsr)

#define UNI (0.5+(signed) SHR3 * 0.2328306e-9)

#define NOR (hz=SHR3,iz=(hz&127),(abs(hz)<kn[iz])? hz*wn[iz] : nfix())
static double wn[128],fn[128];
static uint32_t iz,jz,jsr=123456789,kn[128];
static int32_t hz;

static uint32_t jsr4[4] __attribute__((aligned(16))) = {123456789,112548569,985584512,452236879};//This initialization depends on the seed for nor_table function in oaisim_functions.c file.
static uint32_t iz4[4] __attribute__((aligned(16)));
static float out[4] __attribute__((aligned(16)));
static uint32_t ssh3_sse4[4] __attribute__((aligned(16)));
static uint32_t ifabs4[4] __attribute__((aligned(16)));
static uint32_t hz4[4] __attribute__((aligned(16)));
#if defined(__x86_64__) || defined(__i386__)
static __m128i jsr_128 __attribute__((aligned(16)));
static __m128i jz_128 __attribute__((aligned(16)));
static __m128i hz_128 __attribute__((aligned(16)));
static __m128i iz_128 __attribute__((aligned(16)));
static __m128 x128 __attribute__((aligned(16)));
static __m128i ifabs __attribute__((aligned(16)));

//#define jsr4 (jz=jsr, jsr^=(jsr<<13),jsr^=(jsr>>17),jsr^=(jsr<<5),jz+jsr,printf("seed %d, next seed %d\n",jz,jsr))
//#define SHR3_SSE (jsr_128=_mm_loadu_si128(jsr4),jz_128=jsr_128, printf("jsr4 is %lu,%lu,%lu,%lu\n",jsr4[0],jsr4[1],jsr4[2],jsr4[3]), jsr_128=_mm_xor_si128(_mm_slli_epi32(jsr_128,13),jsr_128),_mm_storeu_si128((__m128i *)jsr4,jsr_128),printf("jsr128<<13 is %lu,%lu,%lu,%lu\n",jsr4[0],jsr4[1],jsr4[2],jsr4[3]),      jsr_128=_mm_xor_si128(_mm_srli_epi32(jsr_128,17),jsr_128),_mm_storeu_si128((__m128i *)jsr4,jsr_128),printf("jsr128>>17 is %lu,%lu,%lu,%lu\n",jsr4[0],jsr4[1],jsr4[2],jsr4[3]),      jsr_128=_mm_xor_si128(_mm_slli_epi32(jsr_128,5),jsr_128),_mm_storeu_si128((__m128i *)jsr4,jsr_128), printf("jsr128<<5  is %lu,%lu,%lu,%lu\n",jsr4[0],jsr4[1],jsr4[2],jsr4[3]),      _mm_storeu_si128(out,_mm_add_epi32(jz_128,jsr_128)),printf("out is %lu,%lu,%lu,%lu\n",out[0],out[1],out[2],out[3]),_mm_add_epi32(jz_128,jsr_128))
#define SHR3_SSE (jsr_128=_mm_loadu_si128((__m128i *)jsr4),jz_128=jsr_128, jsr_128=_mm_xor_si128(_mm_slli_epi32(jsr_128,13),jsr_128),jsr_128=_mm_xor_si128(_mm_srli_epi32(jsr_128,17),jsr_128),jsr_128=_mm_xor_si128(_mm_slli_epi32(jsr_128,5),jsr_128),_mm_storeu_si128((__m128i *)jsr4,jsr_128),_mm_add_epi32(jz_128,jsr_128))

//#define UNI_SSE (_mm_storeu_ps(out,_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.2328306e-9),_mm_cvtepi32_ps(SHR3_SSE)),_mm_set1_ps(0.5))),printf("out is %e,%e,%e,%e\n",out[0],out[1],out[2],out[3]),_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.2328306e-9),_mm_cvtepi32_ps(SHR3_SSE)),_mm_set1_ps(0.5)))
#define UNI_SSE (_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.2328306e-9),_mm_cvtepi32_ps(SHR3_SSE)),_mm_set1_ps(0.5)))
//,_mm_storeu_si128(ssh3_sse4,hz_128),printf("ssh3_sse4 %lu,%lu,%lu,%lu\n",ssh3_sse4[0],ssh3_sse4[1],ssh3_sse4[2],ssh3_sse4[3])
#define NOR_SSE (hz_128=SHR3_SSE,_mm_storeu_si128(hz4,hz_128),iz_128=_mm_and_si128(hz_128,_mm_set1_epi32(127)),_mm_storeu_si128(iz4,iz_128),ifabs=_mm_cmplt_epi32(_mm_max_epi32(_mm_sub_epi32(_mm_setzero_si128(),hz_128),hz_128),_mm_setr_epi32(kn[iz4[0]],kn[iz4[1]],kn[iz4[2]],kn[iz4[3]])),_mm_storeu_si128(ifabs4,ifabs),printf("ifabs %x,%x,%x,%x\n",ifabs4[0],ifabs4[1],ifabs4[2],ifabs4[3]),x128=_mm_and_ps(_mm_cvtepi32_ps(_mm_cmplt_epi32(_mm_max_epi32(_mm_sub_epi32(_mm_setzero_si128(),hz_128),hz_128),_mm_setr_epi32(kn[iz4[0]],kn[iz4[1]],kn[iz4[2]],kn[iz4[3]]))),_mm_mul_ps(_mm_cvtepi32_ps(hz_128),_mm_setr_ps(wn[iz4[0]],wn[iz4[1]],wn[iz4[2]],wn[iz4[3]]))),printf("x128 %e,%e,%e,%e\n",x128[0],x128[1],x128[2],x128[3]),printf("iz %d,%d,%d,%d\n",iz4[0],iz4[1],iz4[2],iz4[3]),printf("wn*hz %e,%e,%e,%e\n",hz4[0]*wn[iz4[0]],hz4[1]*wn[iz4[1]],hz4[2]*wn[iz4[2]],hz4[3]*wn[iz4[3]]),_mm_set1_ps(-1,2,3,4))
#endif
//
//
//
//
//

//#define NOR (hz=SHR3, printf("hz %d\n",hz),sign=(hz&128)>>7,printf("sign %s\n",(sign)?"-":"+"),iz=hz&127,printf("iz %d\n",iz), (abs(hz)<kn[iz])? (sign)?(-1)*hz*wn[iz]:hz*wn[iz] : (sign)?(-1)*nfix():nfix())

/*__m128 nfix_SSE(void)
{
  __m128 r = _mm_set1_ps(3.442620); 
  static __m128 x, y;
  for (;;)
  {
      x= x128;
      //printf("iz %d,x %e,l %d, hz %d,d %e wn %e\n",iz, x, l, hz, d, wn[iz]);
      if (iz==0)
      {   
        do
        {
          x = - 0.2904764 * log (UNI);
          y = - log (UNI);
	} 
        while (y+y < x*x);

        return (hz>0)? _mm_set1_ps(r+x) : _mm_set1_ps(-r-x);
      }
      if (fn[iz]+UNI*(fn[iz-1]-fn[iz])<exp(-0.5*x*x)){

        return _mm_set1_ps(x);
      }
      hz = SHR3;
      iz = hz&127;
      if (abs(hz) < kn[iz]){

        return (_mm_set1_ps((hz)*wn[iz]));
      }
  }
}*/
double nfix(void)
{
  const double r = 3.442620; 
  static double x, y;
  for (;;)
  {
      x=hz *  wn[iz];
      //printf("iz %d,x %e,l %d, hz %d,d %e wn %e\n",iz, x, l, hz, d, wn[iz]);
      if (iz==0)
      {   
        do
        {
          x = - 0.2904764 * log (UNI);
          y = - log (UNI);
	} 
        while (y+y < x*x);
	/*printf("x*x %e,y+y %e\n",x*x,y+y);
	printf("return 1: %e\n",(hz>0)? r+x : -r-x);*/
        return (hz>0)? r+x : -r-x;
      }
      if (fn[iz]+UNI*(fn[iz-1]-fn[iz])<exp(-0.5*x*x)){
        //printf("return 2: %e\n",x);
        return x;
      }
      hz = SHR3;
      iz = hz&127;
      if (abs(hz) < kn[iz]){
	/*printf("return 3: %e\n",(double)hz*wn[iz]);
        printf("return 3: iz %d, hz %d, wn[%d] %e, hz*wz[%d] %e\n",iz,hz,iz,wn[iz],iz,wn[iz]*(double)hz);*/
        return ((hz)*wn[iz]);
      }
  }
}
void table_nor(unsigned long seed)
{
  jsr=seed;
  //printf("Seed for Ziggurat random number generator is %d\n",seed);
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
  /*for ( i = 0; i <= 127; i++ ){
	printf("i %d: kn %d, fn %e, wn %e\n",i,kn[i],fn[i],wn[i]);
  }*/

  return;
}
double ziggurat(double mean, double variance)
{


  /*__m128  n __attribute__((aligned(16)));
  __m128  m __attribute__((aligned(16)));
  //n=NOR_SSE;
  m=log_ps(_mm_setr_ps(1230,100,340,-1000));
  printf("log using functions (1230,100,340,-1000) %e,%e,%e,%e\n",log(1230),log(100),log(340),log(-1000));
  printf("log sse (1230,100,340,-1000) %e,%e,%e,%e\n",m[0],m[1],m[2],m[3]);*/
  return NOR;
}

void boxmuller_SSE_float(__m128 *data1, __m128 *data2) {
	//assert(count % 8 == 0);
	__m128 twopi = _mm_set1_ps(2.0f * 3.14159265358979323846f);
	//__m128 one = _mm_set1_ps(1.0f);
	__m128 minustwo = _mm_set1_ps(-2.0f);
	__m128 u1_ps,u2_ps;
	__m128 radius,theta,sintheta,costheta;

	u1_ps = UNI_SSE;//_mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2),UNI_SSE),_mm_set1_ps(-1));
	u2_ps = UNI_SSE;//_mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2),UNI_SSE),_mm_set1_ps(-1));
	//_mm_storeu_ps(out,u1_ps);
	//_mm_storeu_ps(out1,u2_ps);
	//printf("NOR SSE is %e,%e,%e,%e\n",out[0],out[1],out[2],out[3]);
	//printf("NOR SSE1 is %e,%e,%e,%e\n",out1[0],out1[1],out1[2],out1[3]);
	radius = _mm_sqrt_ps(_mm_mul_ps(minustwo, log_ps(u1_ps)));
	theta = _mm_mul_ps(twopi, u2_ps);
        sincos_ps(theta, &sintheta, &costheta);
	*data1=_mm_mul_ps(radius, costheta);
	*data2=_mm_mul_ps(radius, sintheta);
	//}
}
/*
@defgroup _gaussdouble Gaussian random number generator based on modified Box-Muller transformation.
@ingroup numerical
*/

/*!\brief Gaussian random number generator based on modified Box-Muller transformation.Returns a double-precision floating-point number. */
//#define random_SSE
#ifdef random_SSE
double gaussdouble(double mean, double variance)
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
}
#else
double gaussdouble(double mean, double variance)
{
  /*static int first_run;
  static double sum;
  static int count;
  if (!first_run)
  {
     first_run=1;
     sum=0;
     count=0;
  } */

  static int iset=0;
  static double gset;
  double fac,r,v1,v2;
  static double max=-1000000;
  static double min=1000000;

  if (iset == 0) {
    do {
      /*count++;
      clock_t start=clock();*/
      v1 = 2.0*UNI-1.0;
      /*clock_t stop=clock();
      printf("UE_freq_channel time is %f s, AVERAGE time is %f s, count %d, sum %e\n",(float) (stop-start)/CLOCKS_PER_SEC,(float) (sum+stop-start)/(count*CLOCKS_PER_SEC),count,sum+stop-start);
      sum=(sum+stop-start);*/
      v2 = 2.0*UNI-1.0;
      r = v1*v1+v2*v2;
      //printf("Inside do: r %e\n",r);
    }  while (r >= 1.0);
    //printf("outside do: r %e\n",r);
    fac = sqrt(-2.0*log(r)/r);
    gset= v1*fac;
    iset=1;
    return(sqrt(variance)*v2*fac + mean);
  } else {
    iset=0;
    //printf("normal random number %e, max %e, min %e\n",sqrt(variance)*gset + mean, max,min);
    if (max<sqrt(variance)*gset + mean)
	max=sqrt(variance)*gset + mean;
    if (min>sqrt(variance)*gset + mean)
	min=sqrt(variance)*gset + mean;

    return(sqrt(variance)*gset + mean);
  }
}
#endif

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
