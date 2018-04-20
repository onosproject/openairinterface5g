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

/*! \file PHY/LTE_TRANSPORT/slss.c
 * \brief Functions to Generate and Receive PSDCH
 * \author R. Knopp
 * \date 2017
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr
 * \note
 * \warning
 */
#ifndef __LTE_TRANSPORT_SLSS__C__
#define __LTE_TRANSPORT_SLSS__C__
#include "PHY/defs.h"


void generate_sldch(PHY_VARS_UE *ue,SLDCH_t *sldch,int frame_tx,int subframe_tx) {

  UE_tport_t pdu;
  size_t sldch_header_len = sizeof(UE_tport_header_t);

  pdu.header.packet_type = SLDCH;
  pdu.header.absSF = (frame_tx*10)+subframe_tx;


  AssertFatal(sldch->payload_length <=1500-sldch_header_len - sizeof(SLDCH_t) + sizeof(uint8_t*),
                "SLDCH payload length > %lu\n",
                1500-sldch_header_len - sizeof(SLDCH_t) + sizeof(uint8_t*));
  memcpy((void*)&pdu.sldch,
         (void*)sldch,
         sizeof(SLDCH_t));

  LOG_I(PHY,"SLDCH configuration %lu bytes, TBS payload %d bytes => %lu bytes\n",
        sizeof(SLDCH_t)-sizeof(uint8_t*),
        sldch->payload_length,
        sldch_header_len+sizeof(SLDCH_t)-sizeof(uint8_t*)+sldch->payload_length);

  multicast_link_write_sock(0,
                            &pdu,
                            sldch_header_len+sizeof(SLDCH_t));

}


#endif

void check_and_generate_psdch(PHY_VARS_UE *ue,int frame_tx,int subframe_tx) {
  
  AssertFatal(frame_tx<1024 && frame_tx>=0,"frame %d is illegal\n",frame_tx);
  AssertFatal(subframe_tx<10 && subframe_tx>=0,"subframe %d is illegal\n",subframe_tx);
  SLDCH_t *sldch = ue->sldch;
  AssertFatal(slsch!=NULL,"SLSCH is null\n");
  uint32_t O = ue->sldch->offsetIndicator;
  uint32_t P = ue->sldch->discPeriod;
  uint32_t absSF = (frame_tx*10)+subframe_tx;
  uint32_t absSF_offset,absSF_modP;

  absSF_offset = absSF-O;

  if (absSF_offset < O) return;

  absSF_modP = absSF_offset%P;

  if (absSF_mod == 0) { 
    ue->psdch_coded =0; 
  }

  uint64_t SFpos = ((uint64_t)1) << absSF_modP;
  if ((SFpos & slsch->bitmap1) == 0) return;

  // if we get here, then there is a PSCCH subframe for a potential transmission
  uint32_t sf_index=40,LPSCCH=0;
  for (int i=0;i<40;i++) {
    if (i==absSF_modP) sf_index=LPSCCH;
    if (((((uint64_t)1)<<i) & slsch->bitmap1)>0) LPSCCH++;
  }
  AssertFatal(sf_index<40,"sf_index not set, should not happen (absSF_modP %d)\n",absSF_modP);

  // sf_index now contains the SF index in 0...LPSCCH-1
  // LPSCCH has the number of PSCCH subframes

  // number of resources blocks per slot times 2 slots
  uint32_t M_RB_PSCCH_RP = slsch->N_SL_RB*LPSCCH<<1;
  AssertFatal(slsch->n_pscch < (M_RB_PSCCH_RP>>1)*LPSCCH,"n_pscch not in 0..%d\n",
	      ((M_RB_PSCCH_RP>>1)*LPSCCH)-1);
  // hard-coded to transmission mode one for now (Section 14.2.1.1 from 36.213 Rel14.3)
  uint32_t a1=slsch->n_pscch/LPSCCH;
  uint32_t a2=a1+slsch->n_pscch/LPSCCH+(M_RB_PSCCH_RP>>1);
  uint32_t b1=slsch->n_pscch%LPSCCH;
  uint32_t b2=(slsch->n_pscch + 1 + (a1%(LPSCCH-1)))%LPSCCH;

  LOG_I(PHY,"Checking pscch for absSF %d (LPSCCH %d, M_RB_PSCCH_RP %d, a1 %d, a2 %d, b1 %d, b2 %d) pscch_coded %d\n",
	absSF, LPSCCH, M_RB_PSCCH_RP,a1,a2,b1,b2,ue->pscch_coded);

  ue->slsch_sdu_active = 1;

  if (absSF_modP == b1)      pscch_codingmodulation(ue,frame_tx,subframe_tx,a1,0);	
  else if (absSF_modP == b2) pscch_codingmodulation(ue,frame_tx,subframe_tx,a2,1);
  else return;

}
