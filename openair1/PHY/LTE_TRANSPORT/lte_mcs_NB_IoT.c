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

// 36213 Section 16.5.1.2, Table 
unsigned char get_Qm_ul_NB_IoT(unsigned char I_MCS, uint8_t N_sc_RU)
{

	if (N_sc_RU)
		return(2);
	else
		if (I_MCS<2)
			return(1); 
		else
			return(2);
		
  // if (I_MCS < 11)
  //   return(2);
  // else if (I_MCS < 21)
  //   return(4);
  // else
  //   return(6);

}

