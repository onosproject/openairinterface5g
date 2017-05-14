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

/*! \file eNB_scheduler_dlsch.c
 * \brief procedures related to eNB for the DLSCH transport channel
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
#include "LAYER2/MAC/proto.h"
#include "LAYER2/MAC/extern.h"
#include "UTIL/LOG/log.h"
#include "UTIL/LOG/vcd_signal_dumper.h"
#include "UTIL/OPT/opt.h"
#include "OCG.h"
#include "OCG_extern.h"

#include "defs-nb.h"
#include "proto-nb.h"

#include "RRC/LITE/extern.h"
#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"

//#include "LAYER2/MAC/pre_processor.c"
#include "pdcp.h"

#include "SIMULATION/TOOLS/defs.h" // for taus

#if defined(ENABLE_ITTI)
# include "intertask_interface.h"
#endif

#include "T.h"

#define ENABLE_MAC_PAYLOAD_DEBUG
//#define DEBUG_eNB_SCHEDULER 1


NB_get_dlsch_sdu(
  module_id_t module_idP,
  int CC_id,
  frame_t frameP,
  rnti_t rntiP,
  uint8_t TBindex
)
//------------------------------------------------------------------------------
{

  int UE_id;
  eNB_MAC_INST_NB *eNB=&eNB_mac_inst_NB[module_idP];

  /*for SIBs*/
  if (rntiP==SI_RNTI) {
    LOG_D(MAC,"[eNB %d] CC_id %d Frame %d Get DLSCH sdu for BCCH \n", module_idP, CC_id, frameP);

    return((unsigned char *)&eNB->common_channels[CC_id].BCCH_pdu.payload[0]);
  }

  UE_id = find_UE_id(module_idP,rntiP);

  if (UE_id != -1) {
    LOG_D(MAC,"[eNB %d] Frame %d:  CC_id %d Get DLSCH sdu for rnti %x => UE_id %d\n",module_idP,frameP,CC_id,rntiP,UE_id);
    return((unsigned char *)&eNB->UE_list.DLSCH_pdu[CC_id][TBindex][UE_id].payload[0]);
  } else {
    LOG_E(MAC,"[eNB %d] Frame %d: CC_id %d UE with RNTI %x does not exist\n", module_idP,frameP,CC_id,rntiP);
    return NULL;
  }

}


