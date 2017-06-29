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

/* file: PHY/CODING/defs_nb_iot.h
   purpose: Top-level definitions, data types and function prototypes for openairinterface coding blocks for NB-IoT
   author: raymond.knopp@eurecom.fr, michele.paffetti@studio.unibo.it
   date: 29.06.2017
*/

#ifndef OPENAIR1_PHY_CODING_DEFS_NB_IOT_H_
#define OPENAIR1_PHY_CODING_DEFS_NB_IOT_H_

#include <stdint.h>
#include "PHY/defs.h"

uint32_t sub_block_interleaving_cc_NB_IoT(uint32_t D, uint8_t *d,uint8_t *w);

uint32_t lte_rate_matching_cc_NB_IoT(uint32_t RCC,      // RRC = 2
				     uint16_t E,        // E = 1600
				     uint8_t *w,	// length
				     uint8_t *e);	// length 1600

void ccode_encode_NB_IoT (int32_t numbits,
						  uint8_t add_crc,
						  uint8_t *inPtr,
						  uint8_t *outPtr,
						  uint16_t rnti);



#endif /* OPENAIR1_PHY_CODING_DEFS_NB_IOT_H_ */
