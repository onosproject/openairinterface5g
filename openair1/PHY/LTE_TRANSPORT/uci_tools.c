/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
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

/*! \file PHY/LTE_TRANSPORT/phich.c
* \brief Routines for generation of and computations regarding the uplink control information (UCI) for PUSCH. V8.6 2009-03
* \author R. Knopp, F. Kaltenberger, A. Bhamri
* \date 2011
* \version 0.1
* \company Eurecom
* \email: knopp@eurecom.fr, florian.kaltenberger@eurecom.fr, ankit.bhamri@eurecom.fr
* \note
* \warning
*/
#include "PHY/defs_eNB.h"
#ifdef DEBUG_UCI_TOOLS
#include "PHY/vars.h"
#endif

//#define DEBUG_UCI 1


int16_t find_uci(uint16_t rnti, int frame, int subframe, PHY_VARS_eNB *eNB,find_type_t type) {
  uint16_t i;
  int16_t first_free_index=-1;
  if (eNB==NULL) {
    LOG_E(PHY, "eNB is null\n");
    return (-1);
  }

  for (i=0; i<NUMBER_OF_UCI_VARS_MAX; i++) {
    if ((eNB->uci_vars[i].active >0) &&
	(eNB->uci_vars[i].rnti==rnti) &&
	(eNB->uci_vars[i].frame==frame) &&
	(eNB->uci_vars[i].subframe==subframe)) return(i); 
    else if ((eNB->uci_vars[i].active == 0) && (first_free_index==-1)) first_free_index=i;
  }
  if (type == SEARCH_EXIST){
	LOG_E(PHY, "find_uci find_type is SEARCH_EXIST , return -1\n");
	return(-1);
  }	
  else{
	if(first_free_index < 0){
	  LOG_E(PHY, "find_uci first_free_index = %d,rnti = %d,frame = %d,subframe = %d, last i = %d\n",first_free_index,rnti,frame,subframe, i);
	  for (i=0; i<NUMBER_OF_UCI_VARS_MAX; i++) {
	    LOG_T(PHY, "eNB->uci_vars[%d]\tactive = %d\trnti = %x\tframe = %d\tsubframe = %d\t type %d\n",i,eNB->uci_vars[i].active,eNB->uci_vars[i].rnti,eNB->uci_vars[i].frame,eNB->uci_vars[i].subframe,eNB->uci_vars[i].type);
		}
	}
	return(first_free_index);
  }
}



