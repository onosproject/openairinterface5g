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

//#include "assertions.h"
//#include "PHY/defs.h"
//#include "PHY/extern.h"

//#include "SCHED/defs.h"
//#include "SCHED/extern.h"

//#include "LAYER2/MAC/defs.h"
//#include "LAYER2/MAC/extern.h"

//#include "LAYER2/MAC/proto.h"
#include "UTIL/LOG/log.h"
//#include "UTIL/LOG/vcd_signal_dumper.h"
//#include "UTIL/OPT/opt.h"
//#include "OCG.h"
//#include "OCG_extern.h"
#include "RRC/LITE/proto_NB_IoT.h"
//#include "RRC/LITE/extern.h"
//#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"
//NB-IoT
//#include "PHY/defs_NB_IoT.h"
#include "LAYER2/MAC/defs_NB_IoT.h"
#include "LAYER2/MAC/proto_NB_IoT.h"
#include "LAYER2/MAC/extern_NB_IoT.h"
#include "LAYER2/MAC/vars_NB_IoT.h"  //////////////////// to comment during test
//#include "openair2/COMMON/platform_types.h"
//#include "LAYER2/MAC/pre_processor.c"
//#include "pdcp.h"

/*
#if defined(ENABLE_ITTI)
# include "intertask_interface.h"
#endif
*/
//#define ENABLE_MAC_PAYLOAD_DEBUG
//#define DEBUG_eNB_SCHEDULER 1

/*TODO NB_mac_phy_remove_ue*/

//------------------------------------------------------------------------------
int find_UE_id_NB_IoT(module_id_t mod_idP, rnti_t rntiP)
//------------------------------------------------------------------------------
{
  int UE_id;
  UE_list_NB_IoT_t *UE_list = &eNB_mac_inst_NB_IoT[mod_idP].UE_list;

  for (UE_id = 0; UE_id < NUMBER_OF_UE_MAX_NB_IoT; UE_id++) {
    if (UE_list->active[UE_id] != TRUE) continue;
    if (UE_list->UE_template[UE_PCCID_NB_IoT(mod_idP,UE_id)][UE_id].rnti==rntiP) {
      return(UE_id);
    }
  }

  return(-1);
}

//------------------------------------------------------------------------------
int UE_PCCID_NB_IoT(module_id_t mod_idP,int ue_idP)
//------------------------------------------------------------------------------
{
  return(eNB_mac_inst_NB_IoT[mod_idP].UE_list.pCC_id[ue_idP]);
}

//------------------------------------------------------------------------------
rnti_t UE_RNTI_NB_IoT(module_id_t mod_idP, int ue_idP)
//------------------------------------------------------------------------------
{

  rnti_t rnti = eNB_mac_inst_NB_IoT[mod_idP].UE_list.UE_template[UE_PCCID_NB_IoT(mod_idP,ue_idP)][ue_idP].rnti;

  if (rnti>0) {
    return (rnti);
  }

  LOG_D(MAC,"[eNB %d] Couldn't find RNTI for UE %d\n",mod_idP,ue_idP);
  //display_backtrace();
  return(NOT_A_RNTI);
}

int add_new_ue_NB_IoT(module_id_t mod_idP, int cc_idP, rnti_t rntiP,int harq_pidP)
{
  int UE_id;
  int i, j;

  UE_list_NB_IoT_t *UE_list = &eNB_mac_inst_NB_IoT[mod_idP].UE_list;

  LOG_D(MAC,"[eNB %d, CC_id %d] Adding UE with rnti %x (next avail %d, num_UEs %d)\n",mod_idP,cc_idP,rntiP,UE_list->avail,UE_list->num_UEs);
  dump_ue_list_NB_IoT(UE_list,0);

  for (i = 0; i < NUMBER_OF_UE_MAX_NB_IoT; i++) {
    if (UE_list->active[i] == TRUE) continue;
printf("MAC: new UE id %d rnti %x\n", i, rntiP);
    UE_id = i;
    UE_list->UE_template[cc_idP][UE_id].rnti       = rntiP;
    UE_list->UE_template[cc_idP][UE_id].configured = FALSE;
    UE_list->numactiveCCs[UE_id]                   = 1;
    UE_list->numactiveULCCs[UE_id]                 = 1;
    UE_list->pCC_id[UE_id]                         = cc_idP;
    UE_list->ordered_CCids[0][UE_id]               = cc_idP;
    UE_list->ordered_ULCCids[0][UE_id]             = cc_idP;
    UE_list->num_UEs++;
    UE_list->active[UE_id]                         = TRUE;
    memset((void*)&UE_list->UE_sched_ctrl[UE_id],0,sizeof(UE_sched_ctrl_NB_IoT));

    for (j=0; j<8; j++) {
      UE_list->UE_template[cc_idP][UE_id].oldNDI[j]    = (j==0)?1:0;   // 1 because first transmission is with format1A (Msg4) for harq_pid 0
      UE_list->UE_template[cc_idP][UE_id].oldNDI_UL[j] = (j==harq_pidP)?0:1; // 1st transmission is with Msg3;
    }

    eNB_ulsch_info_NB_IoT[mod_idP][cc_idP][UE_id].status = S_UL_WAITING_NB_IoT;
    eNB_dlsch_info_NB_IoT[mod_idP][cc_idP][UE_id].status = S_DL_WAITING_NB_IoT;
    LOG_D(MAC,"[eNB %d] Add UE_id %d on Primary CC_id %d: rnti %x\n",mod_idP,UE_id,cc_idP,rntiP);
    dump_ue_list_NB_IoT(UE_list,0);
    return(UE_id);
  }

printf("MAC: cannot add new UE for rnti %x\n", rntiP);
  LOG_E(MAC,"error in add_new_ue(), could not find space in UE_list, Dumping UE list\n");
  dump_ue_list_NB_IoT(UE_list,0);
  return(-1);
}

//--------------------------------------------------------------------------------------------------------
int rrc_mac_remove_ue_NB_IoT(
		module_id_t mod_idP,
		rnti_t rntiP)
{
  int i;
  UE_list_NB_IoT_t *UE_list = &eNB_mac_inst_NB_IoT[mod_idP].UE_list;
  int UE_id = find_UE_id_NB_IoT(mod_idP,rntiP); //may should be changed
  int pCC_id;

  if (UE_id == -1) {
printf("MAC: cannot remove UE rnti %x\n", rntiP);
    LOG_W(MAC,"rrc_mac_remove_ue_NB_IoT: UE %x not found\n", rntiP);
    //NB_mac_phy_remove_ue(mod_idP, rntiP);
    return 0;
  }

  pCC_id = UE_PCCID_NB_IoT(mod_idP,UE_id);

printf("MAC: remove UE %d rnti %x\n", UE_id, rntiP);
  LOG_I(MAC,"Removing UE %d from Primary CC_id %d (rnti %x)\n",UE_id,pCC_id, rntiP);
  //dump_ue_list(UE_list,0); //may should be changed

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
  eNB_ulsch_info_NB_IoT[mod_idP][pCC_id][UE_id].rnti                        = NOT_A_RNTI;
  eNB_ulsch_info_NB_IoT[mod_idP][pCC_id][UE_id].status                      = S_UL_NONE_NB_IoT;
  eNB_dlsch_info_NB_IoT[mod_idP][pCC_id][UE_id].rnti                        = NOT_A_RNTI;
  eNB_dlsch_info_NB_IoT[mod_idP][pCC_id][UE_id].status                      = S_DL_NONE_NB_IoT;

  //NB_mac_phy_remove_ue(mod_idP,rntiP);

  // check if this has an RA process active
  RA_TEMPLATE_NB_IoT *RA_template;
  for (i=0;i<RA_PROC_MAX_NB_IoT;i++) {
    RA_template = (RA_TEMPLATE_NB_IoT *)&eNB_mac_inst_NB_IoT[mod_idP].common_channels[pCC_id].RA_template[i];
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
DCI_PDU_NB_IoT *get_dci_sdu_NB_IoT(module_id_t module_idP, int CC_id,frame_t frameP, sub_frame_t subframeP)
//------------------------------------------------------------------------------
{

  return(&eNB_mac_inst_NB_IoT[module_idP].common_channels[CC_id].DCI_pdu);

}

//NB_UL_failure_indication... some of the used primitive haven't defined

