/***********************************************************************

**********************************************************************/
/*! \file PHY/LTE_TRANSPORT/npss_NB_IoT.c
* \Generation of Narrowband Primary Synchronisation Signal(NPSS) for NB-IoT,	 TS 36-212, V13.4.0 2017-02
* \author M. KANJ
* \date 2017
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/

#include "PHY/defs.h"
///////////////#include "PHY/defs_nb_iot.h"
#include "PHY/extern.h"
#include <math.h>
//#include "PHY/impl_defs_lte_NB_IoT.h"
//#include "PHY/impl_defs_top_NB_IoT.h"
//or #include "PHY/defs_nb_iot.h"
#include "PHY/LTE_REFSIG/primary_synch_NB_IoT.h"

int generate_npss_NB_IoT(int32_t 				**txdataF,
						short 					amp,
						LTE_DL_FRAME_PARMS   *frame_parms,
						unsigned short 			symbol_offset,				// symbol_offset should equal to 3 for NB-IoT 
						unsigned short 			slot_offset,
						unsigned short 			RB_IoT_ID)					// new attribute (values are between 0.. Max_RB_number-1), it does not exist for LTE
{
   unsigned short  c,aa,a,s;
   unsigned short  slot_id;
   short 		   *primary_sync;
   unsigned short  NB_IoT_start; 			// Index of the first RE in the RB dedicated for NB-IoT
   unsigned short  bandwidth_even_odd;

   slot_id 		= slot_offset;  					// The id(0..19) of the slot including the NPSS signal // For NB-IoT, slod_id should be 10 (SF5)
   primary_sync = primary_synch_NB_IoT;     		// primary_synch_NB_IoT[264] of primary_synch_NB_IoT.h

   // Signal amplitude
   a = (frame_parms->nb_antennas_tx == 1) ? amp: (amp*ONE_OVER_SQRT2_Q15_NB_IoT)>>15;

   // Testing if the total number of RBs is even or odd (i.e. Identification of the bandwidth: 1.4, 3, 5, 10, ... MHz)
   bandwidth_even_odd = frame_parms->N_RB_DL % 2;  		// 0 for even, 1 for odd

   for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
	
		if(RB_IoT_ID < (frame_parms->N_RB_DL/2))
		{
			NB_IoT_start = frame_parms->ofdm_symbol_size - 12*(frame_parms->N_RB_DL/2) - (bandwidth_even_odd*6) + 12*(RB_IoT_ID%(int)(ceil(frame_parms->N_RB_DL/(float)2)));
		} else {
			NB_IoT_start = (bandwidth_even_odd*6) + 12*(RB_IoT_ID%(int)(ceil(frame_parms->N_RB_DL/(float)2)));
		}
		// For the In-band or Stand-alone case the REs of NPSS signal have the same positions
		for (s=0; s<11; s++ )   				// loop on OFDM symbols
		{	
			for (c=0; c<12; c++) {   			// loop on NB-IoT carriers
			
				((short*)txdataF[aa])[2*( (slot_id*7*frame_parms->ofdm_symbol_size) + ((symbol_offset+s)*frame_parms->ofdm_symbol_size) + NB_IoT_start + c )] =
			
										(a * primary_sync[2*c + (2*12*s)]) >> 15;
										
				((short*)txdataF[aa])[2*( (slot_id*7*frame_parms->ofdm_symbol_size) + ((symbol_offset+s)*frame_parms->ofdm_symbol_size) + NB_IoT_start + c )+1] =
			
										(a * primary_sync[2*c + (2*12*s) + 1]) >> 15;
			}
		}
	}

  return(0);
}

// int generate_npss_NB_IoT(int32_t 				**txdataF,
// 						short 					amp,
// 						LTE_DL_FRAME_PARMS   *frame_parms,
// 						unsigned short 			symbol_offset,				// symbol_offset should equal to 3 for NB-IoT 
// 						unsigned short 			slot_offset,
// 						unsigned short 			RB_IoT_ID)					// new attribute (values are between 0.. Max_RB_number-1), it does not exist for LTE
// {
//    unsigned short  c,aa,a,s;
//    unsigned short  slot_id;
//    short 		   *primary_sync;
//    unsigned short  NB_IoT_start; 			// Index of the first RE in the RB dedicated for NB-IoT
//    unsigned short  bandwidth_even_odd; 
//    unsigned short  UL_RB_ID_NB_IoT; // index of the NB-IoT RB
//    unsigned char   poffset=0, pilot=0; // poffset: base frequency offset of pilots; pilot: LTE pilot flag


//    UL_RB_ID_NB_IoT = frame_parms->NB_IoT_RB_ID; // index of RB dedicated to NB-IoT

//    slot_id 		= slot_offset;  					// The id(0..19) of the slot including the NPSS signal // For NB-IoT, slod_id should be 10 (SF5)
//    primary_sync = primary_synch_NB_IoT;     		// primary_synch_NB_IoT[264] of primary_synch_NB_IoT.h

//    // Signal amplitude
//    a = (frame_parms->nb_antennas_tx == 1) ? amp: (amp*ONE_OVER_SQRT2_Q15_NB_IoT)>>15; 

//    // Testing if the total number of RBs is even or odd (i.e. Identification of the bandwidth: 1.4, 3, 5, 10, ... MHz)
//    bandwidth_even_odd = frame_parms->N_RB_DL % 2;  		// 0 for even, 1 for odd

//    for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
	
// 		if(RB_IoT_ID < (frame_parms->N_RB_DL/2))
// 		{ // RB in first half (below DC)
// 			// NB_IoT_start = frame_parms->ofdm_symbol_size - 12*(frame_parms->N_RB_DL/2) - (bandwidth_even_odd*6) + 12*(RB_IoT_ID%(int)(ceil(frame_parms->N_RB_DL/(float)2)));
// 			NB_IoT_start = UL_RB_ID_NB_IoT*12 + frame_parms->first_carrier_offset;
// 		} else { // RB in the second half (above DC): DC is taken into account
// 			// NB_IoT_start = 1+ (bandwidth_even_odd*6) + 12*(RB_IoT_ID%(int)(ceil(frame_parms->N_RB_DL/(float)2))); 
// 			NB_IoT_start = 1 + bandwidth_even_odd*6 + 6*(2*UL_RB_ID_NB_IoT - (frame_parms->N_RB_DL+bandwidth_even_odd));  
// 		}
// 		// For the In-band or Stand-alone case the REs of NPSS signal have the same positions
// 		for (s=0; s<11; s++ )   				// loop on OFDM symbols
// 		{				
// 		   // CRS (LTE pilot) position within subframe in time
// 		   // Note that pilot position takes into account symbol_offset value
// 		   if (frame_parms->mode1_flag==1){ // SISO mode
// 		   		if (s==1 || s==4 || s==8){
// 		   			pilot = 1; 
// 		   			if (s==1 || s==8){
// 		   				poffset = 3; 
// 		   			}
// 		   		}

// 		   }
// 		   if (pilot == 0){
// 				for (c=0; c<12; c++) {   			// loop on NB-IoT carriers
				
// 					((short*)txdataF[aa])[2*( (slot_id*7*frame_parms->ofdm_symbol_size) + ((symbol_offset+s)*frame_parms->ofdm_symbol_size) + NB_IoT_start + c )] =
				
// 											(a * primary_sync[2*c + (2*12*s)]) >> 15;
											
// 					((short*)txdataF[aa])[2*( (slot_id*7*frame_parms->ofdm_symbol_size) + ((symbol_offset+s)*frame_parms->ofdm_symbol_size) + NB_IoT_start + c )+1] =
				
// 											(a * primary_sync[2*c + (2*12*s) + 1]) >> 15;
// 				}
// 			}
// 			else{
// 				for (c=0; c<12; c++) {   			// loop on NB-IoT carriers
// 					if ((c!=(frame_parms->nushift+poffset)) &&
//                   		(c!=((frame_parms->nushift+poffset+6)%12)))
// 					{
// 						((short*)txdataF[aa])[2*( (slot_id*7*frame_parms->ofdm_symbol_size) + ((symbol_offset+s)*frame_parms->ofdm_symbol_size) + NB_IoT_start + c )] =
					
// 												(a * primary_sync[2*c + (2*12*s)]) >> 15;
												
// 						((short*)txdataF[aa])[2*( (slot_id*7*frame_parms->ofdm_symbol_size) + ((symbol_offset+s)*frame_parms->ofdm_symbol_size) + NB_IoT_start + c )+1] =
					
// 												(a * primary_sync[2*c + (2*12*s) + 1]) >> 15;
// 					}
// 				}
// 			}

// 			pilot = 0; 
// 			poffset = 0;
// 		}
// 	}

//   return(0);
// }

/* (for LTE)
int generate_pss_emul(PHY_VARS_eNB_NB_IoT *phy_vars_eNb,uint8_t sect_id)
{

  msg("[PHY] EMUL eNB generate_pss_emul eNB %d, sect_id %d\n",phy_vars_eNb->Mod_id,sect_id);
  eNB_transport_info[phy_vars_eNb->Mod_id][phy_vars_eNb->CC_id].cntl.pss=sect_id;
  return(0);
}
*/
