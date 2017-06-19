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

/*! \file main.c
 * \brief top init of Layer 2
 * \author  Navid Nikaein and Raymond Knopp, Michele Paffetti
 * \date 2010 - 2014
 * \version 1.0
 * \email: navid.nikaein@eurecom.fr, michele.paffetti@studio.unibo.it
 * @ingroup _mac

 */


#include "asn1_constants.h"
#include "defs_nb_iot.h"
#include "proto_nb_iot.h"
#include "extern.h"

int mac_init_global_param_NB(void)
{

//XXX commented parts are called in the parallel path of OAI
//  Mac_rlc_xface = NULL;
//  LOG_I(MAC,"[MAIN] CALLING RLC_MODULE_INIT...\n");
//
//  if (rlc_module_init()!=0) {
//    return(-1);
//  }
//
//  LOG_I(MAC,"[MAIN] RLC_MODULE_INIT OK, malloc16 for mac_rlc_xface...\n");
//
//  Mac_rlc_xface = (MAC_RLC_XFACE*)malloc16(sizeof(MAC_RLC_XFACE));
//  bzero(Mac_rlc_xface,sizeof(MAC_RLC_XFACE));
//
//  if(Mac_rlc_xface == NULL) {
//    LOG_E(MAC,"[MAIN] FATAL EROOR: Could not allocate memory for Mac_rlc_xface !!!\n");
//    return (-1);
//
//  }
//
//  LOG_I(MAC,"[MAIN] malloc16 OK, mac_rlc_xface @ %p\n",(void *)Mac_rlc_xface);
//
//  mac_xface->mrbch_phy_sync_failure=mrbch_phy_sync_failure;
//  mac_xface->dl_phy_sync_success=dl_phy_sync_success;
//  mac_xface->out_of_sync_ind=rrc_out_of_sync_ind;
//
//  LOG_I(MAC,"[MAIN] RLC interface (mac_rlc_xface) setup and init (maybe no mre used??)\n");

  LOG_I(MAC,"[MAIN] RRC NB-IoT initialization of global params\n");
  rrc_init_global_param_NB();


//  LOG_I(MAC,"[MAIN] PDCP layer init\n");
//#ifdef USER_MODE
//  pdcp_layer_init ();
//#else
//  pdcp_module_init ();
//#endif
//
//  LOG_I(MAC,"[MAIN] Init Global Param Done\n");

  return 0;
}

int mac_top_init_NB()
{

  module_id_t    Mod_id,i,j;
  RA_TEMPLATE_NB *RA_template;
  UE_TEMPLATE_NB *UE_template;
  int size_bytes1,size_bytes2,size_bits1,size_bits2;
  int CC_id;
  int list_el;
  UE_list_NB_t *UE_list; //XXX to review if elements are correct

  // delete the part to init the UE_INST

  //XXX NB_eNB_INST is global and set in lte-softmodem = 1 always (should be modified???)
  LOG_I(MAC,"[MAIN] Init function start:Nb_eNB_INST=%d\n",NB_eNB_INST);

  if (NB_eNB_INST>0) {
    eNB_mac_inst_NB = (eNB_MAC_INST_NB*)malloc16(NB_eNB_INST*sizeof(eNB_MAC_INST_NB));

    if (eNB_mac_inst_NB == NULL) {
      LOG_D(MAC,"[MAIN] can't ALLOCATE %zu Bytes for %d eNB_MAC_INST with size %zu \n",NB_eNB_INST*sizeof(eNB_MAC_INST_NB*),NB_eNB_INST,sizeof(eNB_MAC_INST_NB));
      LOG_I(MAC,"[MAC][MAIN] not enough memory for eNB \n");
      exit(1);
    } else {
      LOG_D(MAC,"[MAIN] ALLOCATE %zu Bytes for %d eNB_MAC_INST @ %p\n",sizeof(eNB_MAC_INST),NB_eNB_INST,eNB_mac_inst_NB);
      bzero(eNB_mac_inst_NB,NB_eNB_INST*sizeof(eNB_MAC_INST_NB));
    }
  } else {
	LOG_I (MAC, "No instance allocated for the MAC layer (NB-IoT)\n");
    eNB_mac_inst_NB = NULL;
  }

  // Initialize Linked-List for Active UEs
  for(Mod_id=0; Mod_id<NB_eNB_INST; Mod_id++) {
    UE_list = &eNB_mac_inst_NB[Mod_id].UE_list;

    UE_list->num_UEs=0;
    UE_list->head=-1;
    UE_list->head_ul=-1;
    UE_list->avail=0;

    for (list_el=0; list_el<NUMBER_OF_UE_MAX-1; list_el++) {
      UE_list->next[list_el]=list_el+1;
      UE_list->next_ul[list_el]=list_el+1;
    }

    UE_list->next[list_el]=-1;
    UE_list->next_ul[list_el]=-1;

  }

  if (Is_rrc_nb_iot_registered == 1) {
    LOG_I(MAC,"[MAIN] calling RRC NB-IoT\n");
#ifndef CELLULAR //nothing to be done yet for cellular
    openair_rrc_top_init_NB();
#endif
  } else {
    LOG_I(MAC,"[MAIN] Running without an RRC\n");
  }

  // initialization for the RA template

  for (i=0; i<NB_eNB_INST; i++)
    for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
      LOG_D(MAC,"[MAIN][eNB %d] CC_id %d initializing RA_template (NB-IoT)\n",i, CC_id);
      LOG_D(MAC, "[MSC_NEW][FRAME 00000][MAC_eNB][MOD %02d][]\n", i);

      RA_template = (RA_TEMPLATE_NB *)&eNB_mac_inst_NB[i].common_channels[CC_id].RA_template[0];

      for (j=0; j<NB_RA_PROC_MAX; j++) {
        size_bytes1 = sizeof(DCIN1_RAR_t);
        size_bytes2 = sizeof(DCIN1_t);
        size_bits1 = sizeof_DCIN1_RAR_t;
        size_bits2 = sizeof_DCIN1_t;

        memcpy((void *)&RA_template[j].RA_alloc_pdu1[0],(void *)&RA_alloc_pdu,size_bytes1);
        memcpy((void *)&RA_template[j].RA_alloc_pdu2[0],(void *)&DLSCH_alloc_pdu1A,size_bytes2);//DLSCH_alloc_pdu1A global!!!!!!

        RA_template[j].RA_dci_size_bytes1 = size_bytes1;
        RA_template[j].RA_dci_size_bytes2 = size_bytes2;
        RA_template[j].RA_dci_size_bits1  = size_bits1;
        RA_template[j].RA_dci_size_bits2  = size_bits2;

        RA_template[j].RA_dci_fmt1        = DCIFormatN1_RAR;
        RA_template[j].RA_dci_fmt2        = DCIFormatN1; //for MSG4
      }

      memset (&eNB_mac_inst_NB[i].eNB_stats,0,sizeof(eNB_STATS_NB));
      UE_template = (UE_TEMPLATE_NB *)&eNB_mac_inst_NB[i].UE_list.UE_template[CC_id][0];

      for (j=0; j<NUMBER_OF_UE_MAX; j++) {
        UE_template[j].rnti=0;
        // initiallize the eNB to UE statistics
        memset (&eNB_mac_inst_NB[i].UE_list.eNB_UE_stats[CC_id][j],0,sizeof(eNB_UE_STATS_NB));
      }
    }


  //ICIC not used

  LOG_I(MAC,"[MAIN][INIT][NB-IoT] Init function finished\n");

  return(0);

}


int l2_init_eNB_NB()
{

  LOG_I(MAC,"[MAIN] Mapping L2 IF-Module functions\n");
  IF_Module_init_L2();

  LOG_I(MAC,"[MAIN] MAC_INIT_GLOBAL_PARAM NB-IoT IN...\n");

  Is_rrc_nb_iot_registered=0;
  mac_init_global_param_NB();
  Is_rrc_nb_iot_registered=1;


  LOG_D(MAC,"[MAIN][NB-IoT] ALL INIT OK\n");

//    mac_xface->macphy_init(eMBMS_active,uecap_xer,cba_group_active,HO_active); (old mac_top_init)
  mac_top_init_NB();

  return(1);
}

