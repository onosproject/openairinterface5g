/***********************************************************************

**********************************************************************/
/*! \file PHY/LTE_REFSIG/lte_dl_cell_spec_NB_IoT.c
* \function called by pilots_NB_IoT.c ,	 TS 36-211, V13.4.0 2017-02
* \author M. KANJ
* \date 2017
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/

/* check if this is required for NB-IoT
#ifdef USER_MODE
#include <stdio.h>
#include <stdlib.h>
#endif
*/
#include "PHY/LTE_REFSIG/defs_NB_IoT.h"
#include "PHY/defs_NB_IoT.h"

int lte_dl_cell_spec_NB_IoT(PHY_VARS_eNB          *phy_vars_eNB,
                            int32_t               *output,
                            short                 amp,
                            unsigned char         Ns,
                            unsigned char         l,
                            unsigned char         p,
					                  unsigned short        RB_IoT_ID) 			// the ID of the RB dedicated for NB_IoT
{
  unsigned char   nu,m;
  unsigned short  k,a;
  unsigned short  NB_IoT_start,bandwidth_even_odd;
  int32_t         qpsk[4];

  a = (amp*ONE_OVER_SQRT2_Q15_NB_IoT)>>15;
  ((short *)&qpsk[0])[0] = a;
  ((short *)&qpsk[0])[1] = a;
  ((short *)&qpsk[1])[0] = -a;
  ((short *)&qpsk[1])[1] = a;
  ((short *)&qpsk[2])[0] = a;
  ((short *)&qpsk[2])[1] = -a;
  ((short *)&qpsk[3])[0] = -a;
  ((short *)&qpsk[3])[1] = -a;

  if ((p==0) && (l==0) )
    nu = 0;
  else if ((p==0) && (l>0))
    nu = 3;
  else if ((p==1) && (l==0))
    nu = 3;
  else if ((p==1) && (l>0))
    nu = 0;
  else {
    printf("lte_dl_cell_spec_NB_IoT: p %d, l %d -> ERROR\n",p,l);
    return(-1);
  }

  // testing if the total number of RBs is even or odd 
  bandwidth_even_odd = phy_vars_eNB->frame_parms.N_RB_DL % 2;    // 0 even, 1 odd
  
  //mprime = 0; 										// mprime = 0,1 for NB_IoT //  for LTE , maximum number of resources blocks (110) - the total number of RB in the selected bandwidth (.... 15 , 25 , 50, 100)
  k = (nu + phy_vars_eNB->frame_parms.nushift)%6;

  if(RB_IoT_ID < (phy_vars_eNB->frame_parms.N_RB_DL/2))
  {																																//XXX this mod operation is not valid since the second member is not an integer but double (for the moment i put a cast)
		NB_IoT_start = phy_vars_eNB->frame_parms.ofdm_symbol_size - 12*(phy_vars_eNB->frame_parms.N_RB_DL/2) - (bandwidth_even_odd*6) + 12*(RB_IoT_ID%((int)(ceil(phy_vars_eNB->frame_parms.N_RB_DL/(float)2))));
  } else {
	  	  	  	  	  	  	  	  	  	  	  	  	 //XXX invalid mod operation (put a cast for the moment)
		NB_IoT_start = (bandwidth_even_odd*6) + 12*(RB_IoT_ID%((int)(ceil(phy_vars_eNB->frame_parms.N_RB_DL/(float)2))));
  }
   
  k+=NB_IoT_start;
   
  DevAssert( Ns < 20 );
  DevAssert( l < 2 );

  for (m=0; m<2; m++) {
    output[k] = qpsk[(phy_vars_eNB->lte_gold_table[Ns][l][0]) & 3]; //TODO should be defined one for NB-IoT
    k+=6;
  }

  return(0);
}


/////////////////////////////////////////////////////////////////////////
/*
int lte_dl_cell_spec_NB_IoT(PHY_VARS_eNB_NB_IoT   *phy_vars_eNB,
                            int32_t               *output,
                            short                 amp,
                            unsigned char         Ns,
                            unsigned char         l,
                            unsigned char         p,
                            unsigned short        RB_IoT_ID)      // the ID of the RB dedicated for NB_IoT
{
  unsigned char   nu,m;
  unsigned short  k,a;
  unsigned short  NB_IoT_start,bandwidth_even_odd;
  int32_t         qpsk[4];

  a = (amp*ONE_OVER_SQRT2_Q15_NB_IoT)>>15;
  ((short *)&qpsk[0])[0] = a;
  ((short *)&qpsk[0])[1] = a;
  ((short *)&qpsk[1])[0] = -a;
  ((short *)&qpsk[1])[1] = a;
  ((short *)&qpsk[2])[0] = a;
  ((short *)&qpsk[2])[1] = -a;
  ((short *)&qpsk[3])[0] = -a;
  ((short *)&qpsk[3])[1] = -a;

  if ((p==0) && (l==0) )
    nu = 0;
  else if ((p==0) && (l>0))
    nu = 3;
  else if ((p==1) && (l==0))
    nu = 3;
  else if ((p==1) && (l>0))
    nu = 0;
  else {
    printf("lte_dl_cell_spec_NB_IoT: p %d, l %d -> ERROR\n",p,l);
    return(-1);
  }

  // testing if the total number of RBs is even or odd 
  bandwidth_even_odd = phy_vars_eNB->frame_parms.N_RB_DL % 2;    // 0 even, 1 odd
  
  //mprime = 0;                     // mprime = 0,1 for NB_IoT //  for LTE , maximum number of resources blocks (110) - the total number of RB in the selected bandwidth (.... 15 , 25 , 50, 100)
  k = (nu + phy_vars_eNB->frame_parms.nushift)%6;

  if(RB_IoT_ID < (phy_vars_eNB->frame_parms.N_RB_DL/2))
  {                                                               //XXX this mod operation is not valid since the second member is not an integer but double (for the moment i put a cast)
    NB_IoT_start = phy_vars_eNB->frame_parms.ofdm_symbol_size - 12*(phy_vars_eNB->frame_parms.N_RB_DL/2) - (bandwidth_even_odd*6) + 12*(RB_IoT_ID%((int)(ceil(phy_vars_eNB->frame_parms.N_RB_DL/(float)2))));
  } else {
                                                   //XXX invalid mod operation (put a cast for the moment)
    NB_IoT_start = (bandwidth_even_odd*6) + 12*(RB_IoT_ID%((int)(ceil(phy_vars_eNB->frame_parms.N_RB_DL/(float)2))));
  }
   
  k+=NB_IoT_start;
   
  DevAssert( Ns < 20 );
  DevAssert( l < 2 );

  for (m=0; m<2; m++) {
    output[k] = qpsk[(phy_vars_eNB->lte_gold_table_NB_IoT[Ns][l][0]) & 3]; //TODO should be defined one for NB-IoT
    k+=6;
  }

  return(0);
}

*/