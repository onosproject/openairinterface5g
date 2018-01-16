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

static unsigned int seed, iy, ir[98];
static uint32_t kn[128];static double wn[128],fn[128];
static unsigned long iz,jz,jsr=123456789;
static long hz;

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

//Procedure to create tables for normal distribution kn,wn and fn
#define SHR3 (jz=jsr, jsr^=(jsr<<13),jsr^=(jsr>>17),jsr^=(jsr<<5),jz+jsr)
#define UNI (0.5+(signed) SHR3 * 0.2328306e-9)
#define NOR (hz=SHR3, iz=hz&127, (abs(hz)<kn[iz])? hz*wn[iz] : nfix())
double nfix()
{
  const double r = 3.442620; static double x, y;
  for (;;)
  {
      x=hz*wn[iz];
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
      if (fn[iz]+UNI*(fn[iz-1]-fn[iz])<exp(-0.5*x*x))
        return x;
      hz = SHR3;
      iz = hz&127;
      if (abs(hz) < kn[iz])
        return (hz*wn[iz]);
  }
}

void setup_nor()
{
  double dn = 3.442619855899;
  int i;
  const double m1 = 2147483648.0;
  double q;
  double tn = 3.442619855899;
  const double vn = 9.91256303526217E-03;

  q = vn/exp(-0.5*dn*dn);

  kn[0] = (int)((dn/q)*m1);
  kn[1] = 0;

  wn[0] = (double) ( q / m1 );
  wn[127] = (double) ( dn / m1 );

  fn[0] = 1.0;
  fn[127] = (double) ( exp ( - 0.5 * dn * dn ) );
  for ( i = 126; 1 <= i; i-- )
  {
    dn = sqrt (-2.0 * log ( vn/dn + exp(-0.5*dn*dn)));
    kn[i+1] = (int) ((dn / tn)*m1);
    tn = dn;
    fn[i] = (double) (exp (-0.5*dn*dn));
    wn[i] = (double) (dn / m1);
  }
  for ( i = 0; i <= 127; i++ ){
	printf("i %d: kn %d, fn %e, wn %e\n",i,kn[i],fn[i],wn[i]);
  }

  return;
}

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
/*!\brief Ziggurat random number generator based on rejection sampling. It returns a pseudorandom normally distributed double number between -4.5 and 4.5*/
double ziggurat()
{
  return NOR;
}

/*
@defgroup _gaussdouble Gaussian random number generator based on modified Box-Muller transformation.
@ingroup numerical
*/

/*!\brief Gaussian random number generator based on modified Box-Muller transformation.Returns a double-precision floating-point number. */
#define random_SSE
#ifdef random_SSE
double gaussdouble(double mean, double variance)//It is necessary to improve the function. However if we enable SSE the gain in time it is not too much.
{
  static int iset=0;
  static double gset;
  double fac,rn,r[2],v1[2],v2[2],ones[]={1.0,1.0};
  static double max=-1000000;
  static double min=1000000;
  __m128d v1_128, v2_128, r128, ones128, compge128;
  ones128 = _mm_load_pd(ones);
  if (iset == 0) {
    do {
      //v2 = 2.0*uniformrandom()-1.0;
      v1_128 = _mm_set_pd(2*UNI-1,2*UNI-1);
      v2_128 = _mm_set_pd(2*UNI-1,2*UNI-1);
      //r = v1*v1+v2*v2;
      v1_128= _mm_mul_pd(v1_128,v1_128);
      v2_128= _mm_mul_pd(v2_128,v2_128);
      r128= _mm_add_pd(v1_128,v2_128);
      //compge128=_mm_cmpge_pd(r128,ones128);
      _mm_storeu_pd(&r[0],r128);
      //printf("Inside do: r[0] %e, r[1] %e\n",r[0],r[1]);
    }  while (r[0] >= 1.0 && r[1]>=1.0);
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
      v1 = 2.0*uniformrandom()-1.0;
      /*clock_t stop=clock();
      printf("UE_freq_channel time is %f s, AVERAGE time is %f s, count %d, sum %e\n",(float) (stop-start)/CLOCKS_PER_SEC,(float) (sum+stop-start)/(count*CLOCKS_PER_SEC),count,sum+stop-start);
      sum=(sum+stop-start);*/
      v2 = 2.0*uniformrandom()-1.0;
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

