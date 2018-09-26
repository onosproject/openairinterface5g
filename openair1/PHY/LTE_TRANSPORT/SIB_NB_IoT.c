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
                   int                      RB_IoT_ID,
                   uint8_t                  operation_mode)
{
 	int done=0;
 	uint8_t *sib1_pdu  = sib1_struct->harq_process->pdu;
 	uint8_t opr_mode = 3;
 	if(operation_mode>=2)
 	{
 		opr_mode =0;
 	}
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
    	int G = get_G_SIB1_NB_IoT(frame_parms,operation_mode);

        if( frame % rep_val == var )
        {
            dlsch_encoding_NB_IoT(sib1_pdu,
                                  sib1_struct,
                                  8,             ///// number_of_subframes_required
                                  G);         //// this vallue is fixed, should take into account in future the case of stand-alone & guard-band 
        
             dlsch_scrambling_Gen_NB_IoT(frame_parms,
                                         sib1_struct,
                                         8*G,
                                         frame, 
                                         subframe*2,
                                         sib1_struct->rnti);
        }

        dlsch_modulation_NB_IoT(txdataF,
                                amp,
                                frame_parms,
                                opr_mode,                          // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
                                sib1_struct,
                                G,                       // number of bits per subframe
                                ((frame%16)/2),
                                4,       
                                RB_IoT_ID);
        done =1;
        frame_parms->flag_free_sf =1;
        
    }

 return(done);
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

int generate_SIB23(NB_IoT_eNB_NDLSCH_t 	      *SIB23,
	                 int32_t 				  **txdataF,
	                 int16_t                  amp,
	                 LTE_DL_FRAME_PARMS 	  *frame_parms,
	                 uint32_t 			      frame,
	                 uint32_t 			      subframe,
	                 int                      RB_IoT_ID)
{
    int done=0;

    if( SIB23->active == 1 )
    {
    	uint8_t *SIB23_pdu  = SIB23->harq_process->pdu;
	 	uint32_t rep =  SIB23->resource_assignment;
	 	uint8_t eutro_control_region = 3;

	    uint32_t counter_rep    =  SIB23->counter_repetition_number;
	    uint32_t pointer_to_sf  =  SIB23->pointer_to_subframe;             /// to identify wich encoded subframe to transmit 

    	int G = get_G_NB_IoT(frame_parms);
    	uint8_t Nsf = SIB23->resource_assignment;   //value 2 or 8

        if(counter_rep == rep)
        {
            dlsch_encoding_NB_IoT(SIB23_pdu,
                                  SIB23,
                                  Nsf,             ///// number_of_subframes_required
                                  G);              //// this vallue is fixed, should take into account in future the case of stand-alone & guard-band 
        
             dlsch_scrambling_Gen_NB_IoT(frame_parms,
                                         SIB23,
                                         Nsf*G,
                                         frame, 
                                         subframe*2,
                                         SIB23->rnti);
        }

        dlsch_modulation_NB_IoT(txdataF,
                                amp,
                                frame_parms,
                                eutro_control_region,     //should be replace by start_symbole   // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
                                SIB23,
                                G,                  // number of bits per subframe
                                pointer_to_sf,
                                subframe,       
                                RB_IoT_ID);

        SIB23->counter_repetition_number--;
        SIB23->pointer_to_subframe++;
        
        frame_parms->flag_free_sf =1;

        if(SIB23->counter_repetition_number == 0)
        {
        	SIB23->active = 0;
        	done =1;
        }

    }

 return(done);

}

////////////////////////////////////////////////////////////////////////


int generate_NDLSCH_NB_IoT(NB_IoT_eNB_NDLSCH_t 	  *RAR,
		                   int32_t 				  **txdataF,
		                   int16_t                amp,
		                   LTE_DL_FRAME_PARMS 	  *frame_parms,
		                   uint32_t 			  frame,
		                   uint32_t 			  subframe,
		                   int                    RB_IoT_ID)
{
    int done = 0;

    if( (RAR->active == 1)  && (frame_parms->flag_free_sf == 0))
    {
    	uint8_t *RAR_pdu  = RAR->harq_process->pdu;
	 	uint32_t rep =  RAR->repetition_number;
	 	uint8_t eutro_control_region = 3;

	    uint32_t counter_rep    =  RAR->counter_repetition_number;
	    uint32_t counter_sf_rep =  RAR->counter_current_sf_repetition;   /// for identifiying when to trigger new scrambling
	    uint32_t pointer_to_sf  =  RAR->pointer_to_subframe;             /// to identify wich encoded subframe to transmit 

    	int G = get_G_NB_IoT(frame_parms);
    	uint8_t Nsf = RAR->number_of_subframes_for_resource_assignment;

        if( (counter_rep == rep) && (counter_sf_rep == 0) && (pointer_to_sf == 0) )
        {
        	

            dlsch_encoding_NB_IoT(RAR_pdu,
                                  RAR,
                                  Nsf,             ///// number_of_subframes_required
                                  G);              //// this vallue is fixed, should take into account in future the case of stand-alone & guard-band 
        
             dlsch_scrambling_Gen_NB_IoT(frame_parms,
                                         RAR,
                                         Nsf*G,
                                         frame, 
                                         subframe*2,
                                         RAR->rnti);
        }

		if( (counter_rep != rep) && (counter_sf_rep == 0) && (pointer_to_sf == 0) )
		{
			dlsch_scrambling_Gen_NB_IoT(frame_parms,
                                         RAR,
                                         Nsf*G,
                                         frame, 
                                         subframe*2,
                                         RAR->rnti);
		}

        if( rep > 4)
        {

		        RAR->counter_current_sf_repetition++;

		        dlsch_modulation_NB_IoT(txdataF,
		                                amp,
		                                frame_parms,
		                                eutro_control_region,     //should be replace by start_symbole   // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
		                                RAR,
		                                G,                  // number of bits per subframe
		                                pointer_to_sf,
		                                subframe,       
		                                RB_IoT_ID);

		        if(RAR->counter_current_sf_repetition == 4)
		        {
		        	RAR->pointer_to_subframe++;
		        	RAR->counter_current_sf_repetition =0;

		        	if (Nsf == RAR->pointer_to_subframe && (RAR->counter_repetition_number > 4))
		        	{
		        		RAR->counter_repetition_number = RAR->counter_repetition_number-4;
		        		RAR->pointer_to_subframe =0;
		        		RAR->counter_current_sf_repetition =0;

		        	} else {

		        		RAR->active = 0;
		        		done =1;
		        	}

		        }

        } else {

        		RAR->counter_current_sf_repetition++;

		        dlsch_modulation_NB_IoT(txdataF,
		                                amp,
		                                frame_parms,
		                                eutro_control_region,     //should be replace by start_symbole   // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
		                                RAR,
		                                G,                  // number of bits per subframe
		                                pointer_to_sf,
		                                subframe,       
		                                RB_IoT_ID);

		        if(RAR->counter_current_sf_repetition == rep)
		        {
		        	RAR->pointer_to_subframe++;

		        	if (Nsf == RAR->pointer_to_subframe)
		        	{
		        		RAR->active = 0;
		        		done =1;
		        	}

		        }
        }   
    }

	return(done);
}
/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
/*int generate_NDCCH_NB_IoT(NB_IoT_eNB_NPDCCH_t 	  *DCI,
		                   int32_t 				  **txdataF,
		                   int16_t                amp,
		                   LTE_DL_FRAME_PARMS 	  *frame_parms,
		                   uint32_t 			  frame,
		                   uint32_t 			  subframe,
		                   int                    RB_IoT_ID)
{
    int done=0;

 	uint8_t  *DCI  = RAR->harq_process->pdu;
 	uint32_t rep =  RAR->repetition_number;
 	uint8_t  eutro_control_region = 3;

    uint32_t counter_rep    =  RAR->counter_repetition_number;
    uint32_t counter_sf_rep =  RAR->counter_current_sf_repetition;   /// for identifiying when to trigger new scrambling
    uint32_t pointer_to_sf  =  RAR->pointer_to_subframe;             /// to identify wich encoded subframe to transmit 

    if( RAR->active == 1 )
    {
    	int G = get_G_NB_IoT(frame_parms);
    	uint8_t Nsf = RAR->number_of_subframes_for_resource_assignment;

        if( (counter_rep == rep) && (counter_sf_rep == 0) && (pointer_to_sf == 0) )
        {
        	

            dlsch_encoding_NB_IoT(RAR_pdu,
                                  RAR,
                                  Nsf,             ///// number_of_subframes_required
                                  G);              //// this vallue is fixed, should take into account in future the case of stand-alone & guard-band 
        
             dlsch_scrambling_Gen_NB_IoT(frame_parms,
                                         RAR,
                                         Nsf*G,
                                         frame, 
                                         subframe*2,
                                         RAR->rnti);
        }

		if( (counter_rep != rep) && (counter_sf_rep == 0) && (pointer_to_sf == 0) )
		{
			dlsch_scrambling_Gen_NB_IoT(frame_parms,
                                         RAR,
                                         Nsf*G,
                                         frame, 
                                         subframe*2,
                                         RAR->rnti);
		}

        if( rep > 4)
        {

		        RAR->counter_current_sf_repetition++;

		        dlsch_modulation_NB_IoT(txdataF,
		                                amp,
		                                frame_parms,
		                                eutro_control_region,     //should be replace by start_symbole   // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
		                                RAR,
		                                G,                  // number of bits per subframe
		                                pointer_to_sf,
		                                subframe,       
		                                RB_IoT_ID);

		        if(RAR->counter_current_sf_repetition == 4)
		        {
		        	RAR->pointer_to_subframe++;
		        	RAR->counter_current_sf_repetition =0;

		        	if (Nsf == RAR->pointer_to_subframe && (RAR->counter_repetition_number > 4))
		        	{
		        		RAR->counter_repetition_number = RAR->counter_repetition_number-4;
		        		RAR->pointer_to_subframe =0;
		        		RAR->counter_current_sf_repetition =0;

		        	} else {

		        		RAR->active = 0;
		        		done =1;
		        	}

		        }

        } else {

        		RAR->counter_current_sf_repetition++;

		        dlsch_modulation_NB_IoT(txdataF,
		                                amp,
		                                frame_parms,
		                                eutro_control_region,     //should be replace by start_symbole   // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
		                                RAR,
		                                G,                  // number of bits per subframe
		                                pointer_to_sf,
		                                subframe,       
		                                RB_IoT_ID);

		        if(RAR->counter_current_sf_repetition == rep)
		        {
		        	RAR->pointer_to_subframe++;

		        	if (Nsf == RAR->pointer_to_subframe)
		        	{
		        		RAR->active = 0;
		        		done =1;
		        	}

		        }
        }   
    }

	return(done);
}

*/


////////////////////////////////////////////////// backup ///////////////////////////
 //////////////////////////////////////////////////// SIB23 ////////////////////////////////////////////////////////////////////////
 /* if( (subframe >0) && (subframe !=5) && (With_NSSS == 0) && (frame%2==1) && (frame%64<16) )   ////if((subframe != 0)  && (subframe != 4) && (subframe != 9) ) 
  {
        

        if( subframe == 1 )
        {
            dlsch_encoding_NB_IoT(sib23_pdu,
                                  sib23,
                                  8,                      ///// number_of_subframes_required
                                  236);                   //////////// G*2
        
            dlsch_scrambling_Gen_NB_IoT(fp,                    // is called only in subframe 4
                                         sib23,
                                         1888,            //////   total_bits
                                         frame,
                                         subframe*2,
                                         eNB->ndlsch_SIB23->rnti);
        }

        if( subframe < 5 )
        {

        dlsch_modulation_NB_IoT(txdataF,
                                AMP,
                                fp,
                                3,                          // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
                                sib23,
                                236,                       // number of bits per subframe
                                (subframe-1),   ///npdsch_data_subframe, data per subframe//subframe index of the data table of npdsch channel (G*Nsf) ((frame%32)/2),values are between 0..Nsf        
                                subframe,
                                RB_IoT_ID);
       } else {

         dlsch_modulation_NB_IoT(txdataF,
                                AMP,
                                fp,
                                3,                          // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
                                sib23,
                                236,                       // number of bits per subframe
                                (subframe-2),///npdsch_data_subframe, data per subframe//subframe index of the data table of npdsch channel (G*Nsf) ((frame%32)/2),values are between 0..Nsf        
                                subframe,
                                RB_IoT_ID);

       }
        
  }
*/