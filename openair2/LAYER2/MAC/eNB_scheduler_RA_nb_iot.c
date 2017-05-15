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

/*! \file eNB_scheduler_RA.c
 * \brief primitives used for random access
 * \author  Navid Nikaein and Raymond Knopp
 * \date 2010 - 2014
 * \email: navid.nikaein@eurecom.fr
 * \version 1.0
 * @ingroup _mac

 */

#include "assertions.h"
#include "platform_types.h"
#include "PHY/defs.h"
#include "PHY/extern.h"
#include "msc.h"

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
#include "proto_nb_iot.h"
#include "defs_nb_iot.h"
#include "math.h"
//#include "LAYER2/MAC/pre_processor.c"
#include "pdcp.h"

#if defined(ENABLE_ITTI)
# include "intertask_interface.h"
#endif

#include "SIMULATION/TOOLS/defs.h" // for taus

#include "T.h"

/*This function should loop all over the preamble index*/
void NB_initiate_ra_proc(module_id_t module_idP, int CC_id,frame_t frameP, uint16_t preamble_index,int16_t timing_offset,sub_frame_t subframeP)
{

  uint8_t i;
  uint8_t carrier_id = 0;/*The index of the UL carrier associated with the NPRACH, the carrier_id of the anchor carrier is 0*/

  RA_TEMPLATE_NB *RA_template = (RA_TEMPLATE_NB *)&eNB_mac_inst_NB[module_idP].common_channels[CC_id].RA_template[0];
    /*preamble index will be a subcarrier index (0-47)*/
  LOG_D(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d Initiating RA procedure for preamble index %d\n",module_idP,CC_id,frameP,preamble_index);

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_INITIATE_RA_PROC,1);
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_INITIATE_RA_PROC,0);
  /*May up to 48 RA Procdeure MAX at the moment*/
  for (i=0; i<NB_RA_PROC_MAX; i++) {
    if (RA_template[i].RA_active==FALSE &&
        RA_template[i].wait_ack_Msg4 == 0) {
      int loop = 0;
      RA_template[i].RA_active=TRUE;
      RA_template[i].generate_rar=1;
      RA_template[i].generate_Msg4=0;
      RA_template[i].wait_ack_Msg4=0;
      RA_template[i].timing_offset=timing_offset;
      /* TODO: find better procedure to allocate RNTI */
      do {
        RA_template[i].rnti = taus();
        loop++;
      } while (loop != 100 &&
               /* TODO: this is not correct, the rnti may be in use without
                * being in the MAC yet. To be refined.
                */
               !(find_UE_id(module_idP, RA_template[i].rnti) == -1 &&
                 /* 1024 and 60000 arbirarily chosen, not coming from standard */
                 RA_template[i].rnti >= 1024 && RA_template[i].rnti < 60000));
      if (loop == 100) { printf("%s:%d:%s: FATAL ERROR! contact the authors\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
      //RA_template[i].RA_rnti = 1+subframeP+(10*f_id);
      /*for NB-IoT, RA_rnti is counted in 36.321 5.1.4*/
      RA_template[i].RA_rnti = 1+floor(frameP/4)+256*carrier_id;
      RA_template[i].preamble_index = preamble_index;
      LOG_D(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d Activating RAR generation for process %d, rnti %x, RA_active %d\n",
            module_idP,CC_id,frameP,i,RA_template[i].rnti,
            RA_template[i].RA_active);

      return;
    }
  }

  LOG_E(MAC,"[eNB %d][RAPROC] FAILURE: CC_id %d Frame %d Initiating RA procedure for preamble index %d\n",module_idP,CC_id,frameP,preamble_index);
}



