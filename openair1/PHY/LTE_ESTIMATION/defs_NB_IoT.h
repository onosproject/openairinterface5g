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

#ifndef __LTE_ESTIMATION_DEFS_NB_IOT__H__
#define __LTE_ESTIMATION_DEFS_NB_IOT__H__

#include "PHY/defs_NB_IoT.h"

/*
int lte_est_timing_advance(NB_IoT_DL_FRAME_PARMS *frame_parms,
                           NB_IoT_eNB_SRS *lte_eNb_srs,
                           unsigned int *eNb_id,
                           unsigned char clear,
                           unsigned char number_of_cards,
                           short coef);
*/

int NB_IoT_est_timing_advance_pusch(PHY_VARS_eNB_NB_IoT* phy_vars_eNB,module_id_t UE_id);


/** @} */
#endif
