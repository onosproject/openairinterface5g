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

/*! \file eNB_scheduler_primitives.c
 * \brief primitives used by eNB for BCH, RACH, ULSCH, DLSCH scheduling
 * \author  Navid Nikaein and Raymond Knopp
 * \date 2010 - 2014
 * \email: navid.nikaein@eurecom.fr
 * \version 1.0
 * @ingroup _mac

 */

#include "assertions.h"
#include "PHY/defs.h"
#include "PHY/extern.h"

#include "SCHED/defs.h"
#include "SCHED/extern.h"

#include "LAYER2/MAC/defs.h"
#include "LAYER2/MAC/extern.h"

#include "LAYER2/MAC/proto.h"
#include "UTIL/LOG/log.h"
#include "UTIL/LOG/vcd_signal_dumper.h"
#include "UTIL/OPT/opt.h"
#include "OCG.h"
#include "OCG_extern.h"

#include "RRC/LITE/extern.h"
#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"
//NB-IoT
#include "defs_nb_iot.h"
#include "proto_nb_iot.h"
//#include "LAYER2/MAC/pre_processor.c"
#include "pdcp.h"

#if defined(ENABLE_ITTI)
# include "intertask_interface.h"
#endif

#define ENABLE_MAC_PAYLOAD_DEBUG
#define DEBUG_eNB_SCHEDULER 1




int NB_rrc_mac_remove_ue(
		module_id_t mod_idP,
		rnti_t rntiP)
{
  int i;
  UE_list_NB_t *UE_list = &eNB_mac_inst_NB[mod_idP].UE_list;
  int UE_id = find_UE_id(mod_idP,rntiP); //may should be changed
  int pCC_id;

  if (UE_id == -1) {
printf("MAC: cannot remove UE rnti %x\n", rntiP);
    LOG_W(MAC,"NB_rrc_mac_remove_ue: UE %x not found\n", rntiP);
    mac_phy_remove_ue(mod_idP, rntiP);
    return 0;
  }

  pCC_id = UE_PCCID(mod_idP,UE_id);

printf("MAC: remove UE %d rnti %x\n", UE_id, rntiP);
  LOG_I(MAC,"Removing UE %d from Primary CC_id %d (rnti %x)\n",UE_id,pCC_id, rntiP);
  dump_ue_list(UE_list,0); //may should be changed

  UE_list->active[UE_id] = FALSE;
  UE_list->num_UEs--;

  // clear all remaining pending transmissions no lcgid in NB-IoT
  /*UE_list->UE_template[pCC_id][UE_id].bsr_info[LCGID0]  = 0;
  UE_list->UE_template[pCC_id][UE_id].bsr_info[LCGID1]  = 0;
  UE_list->UE_template[pCC_id][UE_id].bsr_info[LCGID2]  = 0;
  UE_list->UE_template[pCC_id][UE_id].bsr_info[LCGID3]  = 0;*/

  //UE_list->UE_template[pCC_id][UE_id].ul_SR             = 0;
  UE_list->UE_template[pCC_id][UE_id].rnti              = NOT_A_RNTI;
  UE_list->UE_template[pCC_id][UE_id].ul_active         = FALSE;
  eNB_ulsch_info[mod_idP][pCC_id][UE_id].rnti                        = NOT_A_RNTI;
  eNB_ulsch_info[mod_idP][pCC_id][UE_id].status                      = S_UL_NONE;
  eNB_dlsch_info[mod_idP][pCC_id][UE_id].rnti                        = NOT_A_RNTI;
  eNB_dlsch_info[mod_idP][pCC_id][UE_id].status                      = S_DL_NONE;

  NB_mac_phy_remove_ue(mod_idP,rntiP);

  // check if this has an RA process active
  RA_TEMPLATE_NB *RA_template;
  for (i=0;i<NB_RA_PROC_MAX;i++) {
    RA_template = (RA_TEMPLATE_NB *)&eNB_mac_inst_NB[mod_idP].common_channels[pCC_id].RA_template[i];
    if (RA_template->rnti == rntiP){
      RA_template->RA_active=FALSE;
      RA_template->generate_rar=0;
      RA_template->generate_Msg4=0;
      RA_template->wait_ack_Msg4=0;
      RA_template->timing_offset=0;
      RA_template->RRC_timer=20;
      RA_template->rnti = 0;
      //break;
    }
  }

  return 0;
}

//------------------------------------------------------------------------------
DCI_PDU *NB_get_dci_sdu(module_id_t module_idP, int CC_id,frame_t frameP, sub_frame_t subframeP)
//------------------------------------------------------------------------------
{

  return(&eNB_mac_inst_NB[module_idP].common_channels[CC_id].DCI_pdu);

}

//NB_UL_failure_indication... some of the used primitive haven't defined

