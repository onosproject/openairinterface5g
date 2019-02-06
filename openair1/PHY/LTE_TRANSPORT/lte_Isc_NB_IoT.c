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
* \brief Some support routines for subcarrier start into UL RB for ULSCH
* \author M. KANJ
* \date 2017
* \version 0.1
* \company b<>com
* \email: 
* \note
* \warning
*/

//#include "PHY/defs.h"
//#include "PHY/extern.h"
#include "PHY/LTE_TRANSPORT/proto_NB_IoT.h"

uint8_t tab_ack_15khz[16]= {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
uint8_t tab_ack_3_75khz[16]= {38,39,40,41,42,43,44,45,38,39,40,41,42,43,44,45};
uint8_t tab_I_ru_N_ru_UL[8]= {1,2,3,4,5,6,8,10};
uint8_t tab_I_rep_N_rep_UL[8]={1,2,4,8,16,32,64,128};

/*
// Section 16.5.1.1 in 36.213
uint16_t get_UL_sc_start_NB_IoT(uint16_t I_sc)
{

	if (0<=I_sc && I_sc<=11)
	{
		return I_sc;

	} else if (12<=I_sc && I_sc<=15) {

		return 3*(I_sc-12); 

	}  else if (16<=I_sc && I_sc<=17) {

		return 6*(I_sc-16);

	} else if (I_sc==18){

		return 0; 

	} else if (I_sc>18 || I_sc<0){

		return -1;   /// error msg is needed for this case

	} else {

		return -1;   /// error msg is needed for this case
	}

}
*/

uint16_t get_UL_N_rep_NB_IoT(uint8_t I_rep)
{
   return tab_I_rep_N_rep_UL[I_rep];
}

uint16_t get_UL_N_ru_NB_IoT(uint8_t I_mcs, uint8_t I_ru, uint8_t flag_msg3)
{

	if(flag_msg3 ==1)      // msg3
	{

		if(I_mcs == 0)
		{
			return 4;

		} else if(I_mcs == 1) {

			return 3;

		} else if(I_mcs == 2) {

			return 1;
		} else {
			printf("error in I_mcs value from nfapi");
			return 0;
		}

	} else {      // other NPUSCH

		return tab_I_ru_N_ru_UL[I_ru];

	}

}




uint16_t get_UL_sc_index_start_NB_IoT(uint8_t subcarrier_spacing, uint16_t I_sc, uint8_t npush_format)
{

	if(npush_format == 0)    // format 1
	{

		if(subcarrier_spacing == 1)  ////////// 15 KHz
		{

			if (0<=I_sc && I_sc<12)
			{
				return I_sc;

			} else if (12<=I_sc && I_sc<16) {

				return 3*(I_sc-12); 

			}  else if (16<=I_sc && I_sc<18) {

				return 6*(I_sc-16);

			} else if (I_sc==18){

				return 0; 

			} else {

				return -1;  
				printf("Error in passed nfapi parameters (I_sc)");

			} 

		} else {      //////////// 3.75 KHz

			return I_sc;       /// values 0-47
		}

	} else {       /////////////////////////////////////// format 2

		if(subcarrier_spacing == 1)  ////////// 15 KHz
		{
			
			return(tab_ack_15khz[I_sc]);   

		} else {      //////////// 3.75 KHz

            return(tab_ack_3_75khz[I_sc]);
		}

	}

}

///////////////////////////////////////////////
uint8_t get_numb_UL_sc_NB_IoT(uint8_t subcarrier_spacing, uint8_t I_sc, uint8_t npush_format) 
{
			
	if(npush_format == 0)    // format 1
	{
		if(subcarrier_spacing == 1)  // 15 KHz
		{

			if(I_sc >= 0 && I_sc < 12)
			{
				return 1;
			} else if (I_sc >= 12 && I_sc < 16) {
				return 3;
			} else if (I_sc >= 16 && I_sc < 18) {
				return 6;
			} else if (I_sc == 18) {
				return 12;
			} else {
				return 0;
			}
		} else {
			return 1;
		}

	} else {
		return 1;
	}

}

////////////////////////////////////////////////////
uint8_t get_UL_slots_per_RU_NB_IoT(uint8_t subcarrier_spacing, uint8_t subcarrier_indcation, uint8_t UL_format)
{
	uint8_t subcarrier_number = get_numb_UL_sc_NB_IoT(subcarrier_spacing, subcarrier_indcation, UL_format);

	if(UL_format == 0) // format 1
	{
		if(subcarrier_spacing == 1) // 15 KHz
		{
			if (subcarrier_number == 1 )
			{
				return 16;

			} else if (subcarrier_number == 3) {

				return 8;

			} else if (subcarrier_number == 6) {

				return 4;

			} else {

				return 2;
			}

		} else {         // 3.75 KHz

			return 16;
		}

	} else {   // format 2

		return 4;

	}

}

