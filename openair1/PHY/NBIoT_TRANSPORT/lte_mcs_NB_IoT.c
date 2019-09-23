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

/*! \file PHY/LTE_TRANSPORT/lte_mcs_NB_IoT.c
* \brief Some support routines for MCS computations
* \author  M. KANJ
* \date 2017
* \version 0.1
* \company b<>com
* \email: 
* \note
* \warning
*/

//#include "PHY/defs.h"
//#include "PHY/extern.h"

#include "PHY/NBIoT_TRANSPORT/proto_NB_IoT.h"
#include "PHY/NBIoT_TRANSPORT/extern_NB_IoT.h"


uint8_t get_Qm_UL_NB_IoT(unsigned char I_mcs, uint8_t N_sc_RU, uint8_t I_sc, uint8_t Msg3_flag)
{
	if (Msg3_flag == 1)    /////////////////////////// case of Msg3
    {
    	
        if(I_mcs > 0)
        {
            return  2;

        } else if (I_mcs == 0 && I_sc <12) {

            return 1;

        } else {           ////// I_mcs == 0 && I_sc >11

            return 2;
        }

    } else {                 /////////////////////// case of other NPUSCH config

        if(N_sc_RU == 1)
        {
            if(I_mcs <2)
            {
                return 1;
            } else {
                return 2;
            }

        } else {     /////////////// N_sc_RU > 1

            return 2;
        }

    }
	
}

int get_G_NB_IoT(LTE_DL_FRAME_PARMS *frame_parms)
{
  
	uint16_t num_ctrl_symbols = frame_parms->control_region_size;  // eutra_control_region_size values are 0,1,2

    uint8_t nb_antennas_tx_LTE = frame_parms->nb_antennas_tx;
    uint8_t nb_antennas_tx_NB_IoT = frame_parms->nb_antennas_tx_NB_IoT;

    int G_value=0;

    switch (nb_antennas_tx_NB_IoT + (2*nb_antennas_tx_LTE)) {

		case 10:
			G_value = G_tab[(1*3)-num_ctrl_symbols-1];	
		break;

		case 6:
			G_value = G_tab[(2*3)-num_ctrl_symbols-1];
		break;

		case 4 :
			G_value = G_tab[(3*3)-num_ctrl_symbols-1];
		break;

		case 9 :
			G_value = G_tab[(4*3)-num_ctrl_symbols-1];	
		break;

		case 5:
			G_value = G_tab[(5*3)-num_ctrl_symbols-1];
		break;

		case 3 :
			G_value = G_tab[(6*3)-num_ctrl_symbols-1];
		break;

		default: 

			printf("Error getting G");

	}
  
    return(G_value);
  
}

int get_G_SIB1_NB_IoT(LTE_DL_FRAME_PARMS *frame_parms, uint8_t operation_mode_info)
{
  
    uint16_t num_ctrl_symbols = 0;  // eutra_control_region_size values are 0,1,2
    if(operation_mode_info<2)        /// operation_mode_info, in-band (two value 0,1), stand-alone(3), guard band (2)
    {
        num_ctrl_symbols = 2;
    }
    uint8_t nb_antennas_tx_LTE = frame_parms->nb_antennas_tx;
    uint8_t nb_antennas_tx_NB_IoT = frame_parms->nb_antennas_tx_NB_IoT;

    int G_value=0;

    switch (nb_antennas_tx_NB_IoT + (2*nb_antennas_tx_LTE)) {

        case 10:
            G_value = G_tab[(1*3)-num_ctrl_symbols-1];  
        break;

        case 6:
            G_value = G_tab[(2*3)-num_ctrl_symbols-1];
        break;

        case 4 :
            G_value = G_tab[(3*3)-num_ctrl_symbols-1];
        break;

        case 9 :
            G_value = G_tab[(4*3)-num_ctrl_symbols-1];  
        break;

        case 5:
            G_value = G_tab[(5*3)-num_ctrl_symbols-1];
        break;

        case 3 :
            G_value = G_tab[(6*3)-num_ctrl_symbols-1];
        break;

        default: 

            printf("Error getting G");

    }
  
    return(G_value);
  
}

int get_rep_num_SIB1_NB_IoT(uint8_t scheduling_info_sib1)
{

    int value=0;

    if(scheduling_info_sib1 >11)
    {
        printf("value not allowed for schedulinginfo for sib1");

    } else {
   
        switch(scheduling_info_sib1 % 3)
        {
            case 0:
                value =4;
            break;

            case 1:
                value =8;
            break;

            case 2:
                value =16;
            break;
        }
   }
   
    return(value);
  
}

int get_start_frame_SIB1_NB_IoT(LTE_DL_FRAME_PARMS *frame_parms,uint8_t repetition)
{

    int value=0;

    uint16_t cell_id = frame_parms->Nid_cell ;

    if(repetition == 4)
    {
    	switch(cell_id %4)
    	{
    		case 0:
    			value =0;
    		break;

    		case 1:
    			value =16;
    		break;

    		case 2:
    			value =32;
    		break;

    		case 3:
    			value =48;
    		break;
    	}

    } else if(repetition == 8) {

    	switch(cell_id %2)
    	{
    		case 0:
    			value =0;
    		break;

    		case 1:
    			value =16;
    		break;
    	}

    } else if(repetition == 16) {

    	switch(cell_id %2)
    	{
    		case 0:
    			value =0;
    		break;

    		case 1:
    			value =1;
    		break;
    	}


    } else {
    	printf("Error in getting the starting frame of SIB1 ");
    }
  
    return(value);
  
}

