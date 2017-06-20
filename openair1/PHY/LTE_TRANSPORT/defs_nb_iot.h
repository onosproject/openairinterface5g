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

/*! \file PHY/LTE_TRANSPORT/defs.h
* \brief data structures for PDSCH/DLSCH/PUSCH/ULSCH physical and transport channel descriptors (TX/RX)
* \author R. Knopp
* \date 2011
* \version 0.1
* \company Eurecom
* \email: raymond.knopp@eurecom.fr, florian.kaltenberger@eurecom.fr, oscar.tonelli@yahoo.it
* \note
* \warning
*/
#ifndef __LTE_TRANSPORT_DEFS_NB_IOT__H__
#define __LTE_TRANSPORT_DEFS_NB_IOT__H__
#include "PHY/defs.h"
#include "dci_nb_iot.h"
#ifndef STANDALONE_COMPILE
#include "UTIL/LISTS/list.h"
#endif


typedef struct {
  /// Length of DCI in bits
  uint8_t dci_length;
  /// Aggregation level only 0,1 in NB-IoT
  uint8_t L;
  /// Position of first CCE of the dci
  int firstCCE;
  /// flag to indicate that this is a RA response
  boolean_t ra_flag;
  /// rnti
  rnti_t rnti;
  /// Format
  DCI_format_NB_t format;
  /// DCI pdu
  uint8_t dci_pdu[8];
} DCI_ALLOC_NB_t;

typedef struct {
  //delete the count for the DCI numbers,NUM_DCI_MAX should set to 1 
  uint32_t num_npdcch_symbols;
  uint8_t Num_dci;
  DCI_ALLOC_NB_t dci_alloc[2] ;
} DCI_PDU_NB;

// to be created LTE_eNB_DLSCH_t --> is duplicated for each number of UE and then indexed in the table


/**@}*/
#endif
