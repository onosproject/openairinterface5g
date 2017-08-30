/***********************************************************************

**********************************************************************/
/*! \file PHY/LTE_REFSIG/lte_gold_NB_IoT.c
* \function called by lte_dl_cell_spec_NB_IoT.c ,	 TS 36-211, V13.4.0 2017-02
* \author M. KANJ
* \date 2017
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/

//#include "defs.h"
#include "PHY/LTE_REFSIG/defs_NB_IoT.h"

void lte_gold_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms,uint32_t lte_gold_table_NB_IoT[20][2][14],uint16_t Nid_cell)  //  Nid_cell = Nid_cell_NB_IoT
{
  unsigned char ns,l,Ncp=1;
  unsigned int  n,x1,x2;

  for (ns=0; ns<20; ns++) {

    for (l=0; l<2; l++) {

      x2 = Ncp + (Nid_cell<<1) + (((1+(Nid_cell<<1))*(1 + (l+5) + (7*(1+ns))))<<10); //cinit
     
      x1 = 1+ (1<<31);
      x2 = x2 ^ ((x2 ^ (x2>>1) ^ (x2>>2) ^ (x2>>3))<<31);

      // skip first 50 double words (1600 bits)
      for (n=1; n<50; n++) {
        x1 = (x1>>1) ^ (x1>>4);
        x1 = x1 ^ (x1<<31) ^ (x1<<28);
        x2 = (x2>>1) ^ (x2>>2) ^ (x2>>3) ^ (x2>>4);
        x2 = x2 ^ (x2<<31) ^ (x2<<30) ^ (x2<<29) ^ (x2<<28);
      }

      for (n=0; n<14; n++) {
        x1 = (x1>>1) ^ (x1>>4);
        x1 = x1 ^ (x1<<31) ^ (x1<<28);
        x2 = (x2>>1) ^ (x2>>2) ^ (x2>>3) ^ (x2>>4);
        x2 = x2 ^ (x2<<31) ^ (x2<<30) ^ (x2<<29) ^ (x2<<28);
        lte_gold_table_NB_IoT[ns][l][n] = x1^x2;
      }

    }

  }
}


// \brief gold sequenquence generator
//\param x1
//\param x2 this should be set to c_init if reset=1
//\param reset resets the generator
//\return 32 bits of the gold sequence

unsigned int lte_gold_generic_NB_IoT(unsigned int *x1, unsigned int *x2, unsigned char reset)
{
  int n;
  if (reset) {
    *x1 = 1+ (1<<31);
    *x2=*x2 ^ ((*x2 ^ (*x2>>1) ^ (*x2>>2) ^ (*x2>>3))<<31);

    for (n=1; n<50; n++) {
      *x1 = (*x1>>1) ^ (*x1>>4);
      *x1 = *x1 ^ (*x1<<31) ^ (*x1<<28);
      *x2 = (*x2>>1) ^ (*x2>>2) ^ (*x2>>3) ^ (*x2>>4);
      *x2 = *x2 ^ (*x2<<31) ^ (*x2<<30) ^ (*x2<<29) ^ (*x2<<28);
    }
  }

  *x1 = (*x1>>1) ^ (*x1>>4);
  *x1 = *x1 ^ (*x1<<31) ^ (*x1<<28);
  *x2 = (*x2>>1) ^ (*x2>>2) ^ (*x2>>3) ^ (*x2>>4);
  *x2 = *x2 ^ (*x2<<31) ^ (*x2<<30) ^ (*x2<<29) ^ (*x2<<28);
  return(*x1^*x2);
}


