/***********************************************************************

**********************************************************************/
/*! \file PHY/LTE_TRANSPORT/nsss_NB_IoT.c
* \Generation of Narrowband Secondary Synchronisation Signal(NSSS) for NB-IoT,	 TS 36-212, V13.4.0 2017-02
* \author M. KANJ
* \date 2017
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/

//#include <math.h>
#include "PHY/defs.h"
//#include "PHY/defs_NB_IoT.h" // not can be replaced by impl_defs_lte_NB_IoT & impl_defs_top_NB_IoT if "msg" function is not used
//#include "defs.h"
//#include "PHY/extern_NB_IoT.h"
#include "PHY/extern.h"
//#include "PHY/impl_defs_lte_NB_IoT.h"
//#include "PHY/impl_defs_top_NB_IoT.h"
#include "nsss_NB_IoT.h"

int generate_sss_NB_IoT(int32_t 				**txdataF,
						int16_t 				amp,
						LTE_DL_FRAME_PARMS   *frame_parms, 
						uint16_t 				symbol_offset, 	// symbol_offset = 3 for NB-IoT 
						uint16_t 				slot_offset, 
						unsigned short 			frame_number, 	// new attribute (Get value from higher layer), it does not exist for LTE
						unsigned short 			RB_IoT_ID)		// new attribute (values are between 0.. Max_RB_number-1), it does not exist for LTE
{
	uint8_t  	    aa,Nid_NB_IoT,Nid2,f,q,s,c,u;
	int16_t 		*d;
	uint16_t 		n_f;
	unsigned short  a;
	uint16_t 		slot_id;  						// slot_id = 17 in NB_IoT
	unsigned short  bandwidth_even_odd;
	unsigned short  NB_IoT_start;
  
	n_f 	   = frame_number;
	Nid_NB_IoT = frame_parms->Nid_cell;     // supposing Cell_Id of LTE = Cell_Id of NB-IoT  // if different , NB_IOT_DL_FRAME_PARMS should be includes as attribute
  
	f = (n_f/2) % 4;   						// f = 0, 1, 2, 3
	q = Nid_NB_IoT/126; 					// q = 0, 1, 2, 3
	u = (Nid_NB_IoT % 126); 
  
	Nid2 = q*4 + f; 						// Nid2 = 0..15
  
	switch (Nid2) {
	case 0:
		d = d0f0;
		break;
	case 1:
		d = d0f1;
		break;
	case 2:
		d = d0f2;
		break;
	case 3:
		d = d0f3;
		break;
	case 4:
		d = d1f0;
		break;
	case 5:
		d = d1f1;
		break;
	case 6:
		d = d1f2;
		break;
	case 7:
		d = d1f3;
		break;
	case 8:
		d = d2f0;
		break;
	case 9:
		d = d2f1;
		break;
	case 10:
		d = d2f2;
		break;
	case 11:
		d = d2f3;
	case 12:
		d = d3f0;
		break;
	case 13:
		d = d3f1;
		break;
	case 14:
		d = d3f2;
		break;
	case 15:
		d = d3f3;
		break;

	default:
		msg("[NSSS] ERROR\n");
		return(-1);
	}

	slot_id = slot_offset;
   
	//  Signal amplitude
	a = (frame_parms->nb_antennas_tx == 1) ? amp: (amp*ONE_OVER_SQRT2_Q15_NB_IoT)>>15;
   
	// Testing if the total number of RBs is even or odd (i.e. Identification of the bandwidth: 1.4, 3, 5, 10, ... MHz)
	bandwidth_even_odd = frame_parms->N_RB_DL % 2;  		// 0 even, 1 odd

	for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {

		if(RB_IoT_ID < (frame_parms->N_RB_DL/2))
		{
			NB_IoT_start = frame_parms->ofdm_symbol_size - 12*(frame_parms->N_RB_DL/2) - (bandwidth_even_odd*6) + 12*(RB_IoT_ID%(int)(ceil(frame_parms->N_RB_DL/(float)2)));
		} else {
			NB_IoT_start = (bandwidth_even_odd*6) + 12*(RB_IoT_ID%(int)(ceil(frame_parms->N_RB_DL/(float)2)));
		}
		// For the In-band or Stand-alone case the REs of NPSS signal have the same positions
		for (s=0; s<11; s++ ) 								// loop on OFDM symbols
		{		
			for (c=0; c<12; c++) {							// loop on NB-IoT carriers
			
				((short*)txdataF[aa])[2*( (slot_id*7*frame_parms->ofdm_symbol_size) + ((symbol_offset+s)*frame_parms->ofdm_symbol_size) + NB_IoT_start + c )] =
			
										(a * d[(2*u*132) + (2*c) + (2*s*12) ]) >> 15;
										
				((short*)txdataF[aa])[2*( (slot_id*7*frame_parms->ofdm_symbol_size) + ((symbol_offset+s)*frame_parms->ofdm_symbol_size) + NB_IoT_start + c )+1] =
			
										(a * d[(2*u*132) + (2*c) + (2*s*12) + 1]) >> 15;

			}
		}
	}
  return(0);
}

int rx_nsss_NB_IoT(PHY_VARS_UE_NB_IoT *ue,int32_t *tot_metric)
{

	uint8_t Nid2,q_est,u_est;
	NB_IoT_DL_FRAME_PARMS *frame_parms = &ue->frame_parms; 
	int l,k,m; 
	int toto=0;
	int16_t *d, *nsss_sf;
	int32_t nsss_ext[2][132]; // up to 2 rx antennas ? 
	int32_t metric; // correlation metric

	// we suppose we are in NSSS subframe, after DFT
	// this could be changed in further version
	for (l=0;l<11;l++){
		toto = nsss_extract_NB_IoT(ue,frame_parms,nsss_ext,l);  
	} 

	// now do the Cell ID estimation based on the precomputed sequences in PHY/LTE_TRANSPORT/nsss_NB_IoT.h

	*tot_metric = -99999999;
	nsss_sf = (int16_t*)&nsss_ext[0][0];

	for (Nid2=0;Nid2<16;Nid2++){

		switch (Nid2) {
		case 0:
			d = d0f0;
			break;
		case 1:
			d = d0f1;
			break;
		case 2:
			d = d0f2;
			break;
		case 3:
			d = d0f3;
			break;
		case 4:
			d = d1f0;
			break;
		case 5:
			d = d1f1;
			break;
		case 6:
			d = d1f2;
			break;
		case 7:
			d = d1f3;
			break;
		case 8:
			d = d2f0;
			break;
		case 9:
			d = d2f1;
			break;
		case 10:
			d = d2f2;
			break;
		case 11:
			d = d2f3;
		case 12:
			d = d3f0;
			break;
		case 13:
			d = d3f1;
			break;
		case 14:
			d = d3f2;
			break;
		case 15:
			d = d3f3;
			break;

		default:
			msg("[NSSS] ERROR\n");
			return(-1);
		}

		for (k=0;k<126;k++){ // corresponds to u in standard
			metric = 0;  

			for (m=0;m<132;m++){ // 132 resource elements in NSSS subframe

				metric += (int32_t)d[(k*126+m)<<1] * (int32_t)nsss_sf[(k*126+m)<<1] + 
						(int32_t)d[((k*126+m)<<1)+1] * (int32_t)nsss_sf[((k*126+m)<<1)+1]; // real part of correlation

			}

			if (metric > *tot_metric){
				q_est = Nid2/4; 
				u_est = k; 
				ue->frame_parms.Nid_cell = q_est*126 + u_est; 
			}

		}

	}

	return(0); 
}

int nsss_extract_NB_IoT(PHY_VARS_UE_NB_IoT *ue,
						NB_IoT_DL_FRAME_PARMS *frame_parms,
						int32_t **nsss_ext,
						int l)
{
	uint8_t i,aarx; 
	unsigned short  UL_RB_ID_NB_IoT; // index of RB dedicated to NB-IoT
	int32_t *nsss_rxF,*nsssF_ext;
	int32_t **rxdataF; 
	int first_symbol_offset = 3; // NSSS starts at third symbol in subframe
	
	UL_RB_ID_NB_IoT = frame_parms->NB_IoT_RB_ID; 

	for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {

		rxdataF  =  ue->common_vars.common_vars_rx_data_per_thread[0].rxdataF; // Note that 
		nsssF_ext   = &nsss_ext[aarx][l*12];

		if (UL_RB_ID_NB_IoT <= (frame_parms->N_RB_DL>>1)) { // NB-IoT RB is in the first half 
          nsss_rxF       = &rxdataF[aarx][(UL_RB_ID_NB_IoT*12 + frame_parms->first_carrier_offset + ((l+first_symbol_offset)*(frame_parms->ofdm_symbol_size)))];
        }// For second half of RBs skip DC carrier
        else{ // NB-IoT RB is in the second half 
          nsss_rxF       = &rxdataF[aarx][(1 + 6*(2*UL_RB_ID_NB_IoT - frame_parms->N_RB_DL) + ((l+first_symbol_offset)*(frame_parms->ofdm_symbol_size)))];
          //dl_ch0++;
        } 

        for (i=0; i<12; i++) {
              nsssF_ext[i]=nsss_rxF[i];
        }

	}

	return(0); 

}

