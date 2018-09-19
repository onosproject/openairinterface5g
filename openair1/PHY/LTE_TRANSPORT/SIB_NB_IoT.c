/* Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
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

/*! \file PHY/LTE_TRANSPORT/SIB_NB_IoT.c
* \Fucntions for the generation of SIB information for NB_IoT,	 TS 36-212, V13.4.0 2017-02
* \author M. KANJ
* \date 2018
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/


#include "PHY/LTE_TRANSPORT/defs_NB_IoT.h"
#include "PHY/LTE_TRANSPORT/proto_NB_IoT.h"
//#include "PHY/CODING/defs_NB_IoT.h"
//#include "PHY/LTE_REFSIG/defs_NB_IoT.h"
//#include "PHY/impl_defs_lte_NB_IoT.h"
//#include "PHY/impl_defs_top_NB_IoT.h"
#include "PHY/impl_defs_lte.h"

/////////////////////////////////////////////////////////////////////////////////////////

int generate_SIB1(NB_IoT_eNB_NDLSCH_t 		*sib1_struct,
                   int32_t 					**txdataF,
                   int16_t                  amp,
                   LTE_DL_FRAME_PARMS 	    *frame_parms,
                   uint32_t 				frame,
                   uint32_t 				subframe,
                   int                      RB_IoT_ID)
{
 	int done=0;
 	uint8_t *sib1_pdu  = sib1_struct->harq_process->pdu;

	uint8_t tmp =0;
    uint8_t rep_val = 0;
    uint8_t start_frame = get_start_frame_SIB1_NB_IoT(frame_parms, get_rep_num_SIB1_NB_IoT(sib1_struct->repetition_number_SIB1));

    switch( get_rep_num_SIB1_NB_IoT(sib1_struct->repetition_number_SIB1) )
    {
      case 4:
              rep_val = 64;
      break;

      case 8:
              rep_val = 32;
      break;
              
      case 16:
              rep_val = 16;
      break;

      default:
            printf("Error in SIB1");

    }

    uint8_t var = 0;

    if(start_frame == 1)
    {
      var =1;
    }

    if(start_frame>=16)
    {
        tmp = 1;
    }

    uint8_t born_inf = 0 + start_frame*tmp;
    uint8_t born_sup = 16 + start_frame*tmp;

    if((subframe == 4)  && (frame%2 == var) && (born_inf<= frame % rep_val) && (frame % rep_val < born_sup ))
    {

        if( frame % rep_val == var )
        {
            dlsch_encoding_NB_IoT(sib1_pdu,
                                  sib1_struct,
                                  8,             ///// number_of_subframes_required
                                  236);         //// this vallue is fixed, should take into account in future the case of stand-alone & guard-band 
        
             dlsch_scrambling_Gen_NB_IoT(frame_parms,
                                         sib1_struct,
                                         1888,
                                         frame, 
                                         subframe*2,
                                         sib1_struct->rnti);
        }

        dlsch_modulation_NB_IoT(txdataF,
                                amp,
                                frame_parms,
                                3,                          // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
                                sib1_struct,
                                236,                       // number of bits per subframe
                                ((frame%16)/2),
                                4,       
                                RB_IoT_ID);
        done =1;
        
    }

 return(done);
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/*int generate_SIB23(NB_IoT_eNB_NPBCH_t 		*eNB_npbch,
                   int32_t 					**txdataF,
                   int 						amp,
                   LTE_DL_FRAME_PARMS 	    *frame_parms,
                   uint8_t 					*npbch_pdu,
                   uint8_t 					frame_mod64,
				   unsigned short 			NB_IoT_RB_ID)
{
 
 return(0);
}
*/
////////////////////////////////////////////////////////////////////////

