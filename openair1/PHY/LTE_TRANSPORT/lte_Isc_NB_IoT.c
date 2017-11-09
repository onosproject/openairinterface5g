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
* \author V. Savaux
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

// Section 16.5.1.1 in 36.213
uint16_t get_UL_sc_start_NB_IoT(uint16_t I_sc)
{

	if (0<=I_sc && I_sc<=11){
		return I_sc; 
	}
	if (12<=I_sc && I_sc<=15){
		return 3*(I_sc-12); 
	}
	if (16<=I_sc && I_sc<=17){
		return 6*(I_sc-16); 
	}
	if (I_sc==18){
		return 0; 
	}
	if (I_sc>18){
		return -1; 
	}

}

