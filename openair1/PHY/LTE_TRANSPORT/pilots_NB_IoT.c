/***********************************************************************

**********************************************************************/
/*! \file PHY/LTE_TRANSPORT/pilots_NB_IoT.c
* \Generation of Reference signal (RS) for NB-IoT,	 TS 36-211, V13.4.0 2017-02
* \author M. KANJ
* \date 2017
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/

#include "PHY/defs.h"
#include "PHY/defs_NB_IoT.h"
#include "PHY/LTE_REFSIG/defs_NB_IoT.h"

void generate_pilots_NB_IoT(PHY_VARS_eNB         *phy_vars_eNB,
                            int32_t              **txdataF,
                            int16_t              amp,
                            uint16_t             Ntti, 				// Ntti = 0..9
					        unsigned short       RB_ID,			// RB reserved for NB-IoT
					        unsigned short       With_NSSS) 		// With_NSSS = 1; if the frame include a sub-Frame with NSSS signal
{

  LTE_DL_FRAME_PARMS *frame_parms = &phy_vars_eNB->frame_parms;
  uint16_t subframe = Ntti;
  uint32_t tti,tti_offset,slot_offset,Nsymb,samples_per_symbol;
  uint8_t  first_pilot,second_pilot;
  unsigned short RB_IoT_ID = RB_ID;
  Nsymb        = 14;
  first_pilot  = 5;     // first pilot position
  second_pilot = 6;		// second pilot position

  slot_offset        = (Ntti*2)%20;

  if(subframe !=5 && ((With_NSSS*subframe)!= 9) )			//  condition to avoid NPSS and NSSS signals						
  {
    tti_offset         = subframe*frame_parms->ofdm_symbol_size*Nsymb; 				// begins with 0
    samples_per_symbol = frame_parms->ofdm_symbol_size;  				// ex. 512
	
		//Generate Pilots for slot 0 and 1
		
		//antenna 0 symbol 5 slot 0
		lte_dl_cell_spec_NB_IoT(phy_vars_eNB,
					            &txdataF[0][tti_offset + (first_pilot*samples_per_symbol)], 			// tti_offset 512 x 32 bits
                                amp,
                                slot_offset,
                                0,
                                0,
                                RB_IoT_ID);
					 
		//antenna 0 symbol 6 slot 0
		lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[0][tti_offset + (second_pilot*samples_per_symbol)],
                                amp,
                                slot_offset,
                                1,
                                0,
                                RB_IoT_ID);

		//antenna 0 symbol 5 slot 1
		lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[0][tti_offset + (7*samples_per_symbol) + (first_pilot*samples_per_symbol)],
                                amp,
                                1+slot_offset,
                                0,
                                0,
                                RB_IoT_ID);

		//antenna 0 symbol 6 slot 1
		lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[0][tti_offset + (7*samples_per_symbol) + (second_pilot*samples_per_symbol)],
                                amp,
                                1+slot_offset,
                                1,
                                0,
                                RB_IoT_ID);

		if (frame_parms->nb_antennas_tx > 1) {  																// Pilots generation with two antennas

			// antenna 1 symbol 5 slot 0
			lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[1][tti_offset + (first_pilot*samples_per_symbol)],
                                    amp,
                                    slot_offset,
                                    0,
                                    1,
                                    RB_IoT_ID);

			// antenna 1 symbol 6 slot 0
			lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[1][tti_offset + (second_pilot*samples_per_symbol)],
                                    amp,
                                    slot_offset,
                                    1,
                                    1,
                                    RB_IoT_ID);

			//antenna 1 symbol 5 slot 1
			lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[1][tti_offset + (7*samples_per_symbol) + (first_pilot*samples_per_symbol)],
                                    amp,
                                    1+slot_offset,
                                    0,
                                    1,
                                    RB_IoT_ID);

			// antenna 1 symbol 6 slot 1
			lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[1][tti_offset + (7*samples_per_symbol) + (second_pilot*samples_per_symbol)],
                                    amp,
                                    1+slot_offset,
                                    1,
                                    1,
                                    RB_IoT_ID);
		}
  }
}



//////////////////////////////////////////////////////
/*
void generate_pilots_NB_IoT(PHY_VARS_eNB_NB_IoT  *phy_vars_eNB,
                            int32_t              **txdataF,
                            int16_t              amp,
                            uint16_t             Ntti,              // Ntti = 10
                            unsigned short       RB_IoT_ID,         // RB reserved for NB-IoT
                            unsigned short       With_NSSS)         // With_NSSS = 1; if the frame include a sub-Frame with NSSS signal
{

  NB_IoT_DL_FRAME_PARMS *frame_parms = &phy_vars_eNB->frame_parms_NB_IoT;

  uint32_t tti,tti_offset,slot_offset,Nsymb,samples_per_symbol;
  uint8_t  first_pilot,second_pilot;

  Nsymb        = 14;
  first_pilot  = 5;     // first pilot position
  second_pilot = 6;     // second pilot position
  
  for (tti=0; tti<Ntti; tti++) {                                        // loop on sub-frames    

    tti_offset         = tti*frame_parms->ofdm_symbol_size*Nsymb;               // begins with 0
    samples_per_symbol = frame_parms->ofdm_symbol_size;                 // ex. 512
    slot_offset        = (tti*2)%20;                                            // 0, 2, 4, ....... 18 
    
    if((slot_offset != 10) && ((With_NSSS*slot_offset) != 18)) {        //  condition to avoid NPSS and NSSS signals
    
        //Generate Pilots for slot 0 and 1
        
        //antenna 0 symbol 5 slot 0
        lte_dl_cell_spec_NB_IoT(phy_vars_eNB,
                                &txdataF[0][tti_offset + (first_pilot*samples_per_symbol)],             // tti_offset 512 x 32 bits
                                amp,
                                RB_IoT_ID,
                                slot_offset,
                                0, //p
                                0);
                     
        //antenna 0 symbol 6 slot 0
        lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[0][tti_offset + (second_pilot*samples_per_symbol)],
                                amp,
                                RB_IoT_ID,
                                slot_offset,
                                1,
                                0);

        //antenna 0 symbol 5 slot 1
        lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[0][tti_offset + (7*samples_per_symbol) + (first_pilot*samples_per_symbol)],
                                amp,
                                RB_IoT_ID,
                                1+slot_offset,
                                0,
                                0);

        //antenna 0 symbol 6 slot 1
        lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[0][tti_offset + (7*samples_per_symbol) + (second_pilot*samples_per_symbol)],
                                amp,
                                RB_IoT_ID,
                                1+slot_offset,
                                1,
                                0);

        if (frame_parms->nb_antennas_tx > 1) {                                                                  // Pilots generation with two antennas

            // antenna 1 symbol 5 slot 0
            lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[1][tti_offset + (first_pilot*samples_per_symbol)],
                                    amp,
                                    RB_IoT_ID,
                                    slot_offset,
                                    0,
                                    1);

            // antenna 1 symbol 6 slot 0
            lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[1][tti_offset + (second_pilot*samples_per_symbol)],
                                    amp,
                                    RB_IoT_ID,
                                    slot_offset,
                                    1,
                                    1);

            //antenna 1 symbol 5 slot 1
            lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[1][tti_offset + (7*samples_per_symbol) + (first_pilot*samples_per_symbol)],
                                    amp,
                                    RB_IoT_ID,
                                    1+slot_offset,
                                    0,
                                    1);

            // antenna 1 symbol 6 slot 1
            lte_dl_cell_spec_NB_IoT(phy_vars_eNB,&txdataF[1][tti_offset + (7*samples_per_symbol) + (second_pilot*samples_per_symbol)],
                                    amp,
                                    RB_IoT_ID,
                                    1+slot_offset,
                                    1,
                                    1);
        }
    }
  }
}
*/