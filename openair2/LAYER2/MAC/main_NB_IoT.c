
/*! \file main_NB_IoT.c
 * \brief top init of Layer 2
 * \author  NTUST BMW LAB./
 * \date 2017
 * \version 1.0
 * \email: 
 */


//#include "asn1_constants.h"
#include "LAYER2/MAC/defs_NB_IoT.h"
#include "LAYER2/MAC/proto_NB_IoT.h"
#include "LAYER2/MAC/extern_NB_IoT.h"
#include "vars_NB_IoT.h"
#include "RRC/LITE/proto_NB_IoT.h"

#define NUM_USS_PP 3
#define USER_NUM_USS 10

int mac_init_global_param_NB_IoT(void)
{
  ///// removed since already called by mac_init_global_param //////////////
  /*
   if (rlc_module_init()!=0) {
     return(-1);
   }
   */
  /////////////////////////////////////////////////////////////////////////////
 // LOG_I(MAC,"[MAIN] RRC NB-IoT initialization of global params\n");
 // rrc_init_global_param_NB_IoT();

 ///// removed since already called by mac_init_global_param //////////////
  /*
  LOG_I(MAC,"[MAIN] PDCP layer init\n");
  #ifdef USER_MODE
  pdcp_layer_init ();
  #else
  pdcp_module_init ();
  #endif
  */
  ////////////////////////////////////////////////////////////////////////
  return 0;
}


// Initial function of the intialization for NB-IoT MAC
void init_mac_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst)
{
  int32_t i, j, k;  

  LOG_I(MAC,"[NB-IoT] MAC start initialization\n");
  
  for(i=0;i<64;++i)
  {
    mac_inst->sib1_flag[i] = 0;
    mac_inst->sib1_count[i] = 0;
  }
  
  //rrc_mac_config_req_NB_IoT(&mac_inst->rrc_config, 1, 1, 0, 0);
  init_tool_sib1(mac_inst);

  //  output handler 
  //  RA
  mac_inst->RA_msg2_list.head = (RA_TEMPLATE_NB_IoT *)0;
  mac_inst->RA_msg3_list.head = (RA_TEMPLATE_NB_IoT *)0;
  mac_inst->RA_msg4_list.head = (RA_TEMPLATE_NB_IoT *)0;
  mac_inst->RA_msg2_list.tail = (RA_TEMPLATE_NB_IoT *)0;
  mac_inst->RA_msg3_list.tail = (RA_TEMPLATE_NB_IoT *)0;
  mac_inst->RA_msg4_list.tail = (RA_TEMPLATE_NB_IoT *)0;

  sib1_NB_IoT_sched_t *config = &mac_inst->rrc_config.sib1_NB_IoT_sched_config;
  
  // DLSF Table
  init_dlsf_info(mac_inst, &DLSF_information);

  //  init sib1 tool
  //int repetition_pattern = 1;// 1:every2frame, 2:every4frame, 3:every8frame, 4:every16frame
  for(i=0;i<8;++i){
    mac_inst->sib1_flag[(i<<1)+config->starting_rf] = 1;
  }

  for(i=0, j=0;i<64;++i){
    if(mac_inst->sib1_flag[i]==1){
      ++j;
    }
    mac_inst->sib1_count[i]=j;
  }
//printf("%d", mac_inst->sib1_period);
  for(i=0, j=0;i<640;++i){
    //printf("*%d", i);
    if(is_dlsf(mac_inst, i)){
      ++j;
    }
    //printf("-");
    if(i%10==9){
      mac_inst->dlsf_table[i/10] = j;
    }
  }

  mac_inst->rrc_config.si_window_length = ms160;
  mac_inst->rrc_config.sibs_NB_IoT_sched[0].si_periodicity = rf64;

  for(i=0;i<256;++i){
    mac_inst->sibs_table[i] = -1;
  }
  for(j=0;j<6;++j){
    if(0x0 != mac_inst->rrc_config.sibs_NB_IoT_sched[j].sib_mapping_info){
      k = mac_inst->rrc_config.sibs_NB_IoT_sched[j].si_periodicity / mac_inst->rrc_config.si_window_length;
      for(i=0;i<(256/k);++i){
        mac_inst->sibs_table[(i*k)+j] = j;
      }
    }
  }

  mac_inst->schedule_subframe_DL = 0;
  mac_inst->schedule_subframe_UL = 0;

  available_resource_DL = available_resource_DL_last = (available_resource_DL_t *)0;

  //  init downlink list 0-100
  init_dl_list(mac_inst);

  for(i=0; i<MAX_NUMBER_OF_UE_MAX_NB_IoT; ++i){
    mac_inst->RA_template[i].active = 0;
    mac_inst->RA_template[i].msg3_retransmit_count = 0;
    mac_inst->RA_template[i].msg4_retransmit_count = 0;
    mac_inst->RA_template[i].ta = 0;
    mac_inst->RA_template[i].preamble_index = 0;
    mac_inst->RA_template[i].ue_rnti = 0x0;
    mac_inst->RA_template[i].ra_rnti = 0x0;
    mac_inst->RA_template[i].next = (RA_TEMPLATE_NB_IoT *)0;
    mac_inst->RA_template[i].prev = (RA_TEMPLATE_NB_IoT *)0;
    mac_inst->RA_template[i].wait_msg4_ack = 0;
    mac_inst->RA_template[i].wait_msg3_ack = 0;
  }
  //3 CE level USS list
  mac_inst->UE_list_spec = (UE_list_NB_IoT_t*)malloc(NUM_USS_PP*sizeof(UE_list_NB_IoT_t));
  //initial UE list
  LOG_I(MAC,"[NB-IoT] Initial UE list\n");

  mac_inst->num_uss_list = NUM_USS_PP;
  for(i=0;i<NUM_USS_PP;++i)
  {
    rrc_mac_config_req_NB_IoT(0,0,0,NULL,NULL,NULL,NULL,NULL,1,i);
    
    (mac_inst->UE_list_spec+i)->head = -1;
    (mac_inst->UE_list_spec+i)->tail = -1;
    (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.R_max = mac_inst->rrc_config.npdcch_ConfigDedicated[i].R_max;
    (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.G = mac_inst->rrc_config.npdcch_ConfigDedicated[i].G;
    (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.a_offset = mac_inst->rrc_config.npdcch_ConfigDedicated[i].a_offset;
    //(mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.R_max = 8;
    //(mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.G = 1;
    //(mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.a_offset = 0;
    (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.T = (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.R_max * (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.G;
    (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.ss_start_uss = (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.T * (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.a_offset;
    for(j=0;j<MAX_NUMBER_OF_UE_MAX_NB_IoT;++j)
    {
      (mac_inst->UE_list_spec+i)->UE_template_NB_IoT[j].active=0;
      (mac_inst->UE_list_spec+i)->UE_template_NB_IoT[j].RRC_connected=0;
      (mac_inst->UE_list_spec+i)->UE_template_NB_IoT[j].direction = -1;
    }
    //SCHEDULE_LOG("[%04d][init_mac_NB_IoT] List_number %d R_max %d G %.1f a_offset %.1f T %d SS_start %d\n", mac_inst->current_subframe, i, (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.R_max, (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.G, (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.a_offset, (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.T, (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.ss_start_uss);
    
    LOG_I(MAC,"[NB-IoT] List_number %d R_max %d G %.1f a_offset %.1f T %d SS_start %d\n", i, (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.R_max, (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.G, (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.a_offset, (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.T, (mac_inst->UE_list_spec+i)->NPDCCH_config_dedicated.ss_start_uss);
  }

/*
  //Initial one UE template
  //UE_TEMPLATE_NB_IoT *UE_info=(UE_TEMPLATE_NB_IoT*)malloc(USER_NUM_USS*sizeof(UE_TEMPLATE_NB_IoT));

  mac_inst->UE_list_spec->head=0;
  mac_inst->UE_list_spec->tail=0;

  for(i=0;i<USER_NUM_USS;++i)
  {
    UE_info_setting(mac_inst->UE_list_spec->UE_template_NB_IoT+i);

    if(i==0)
    {
      (mac_inst->UE_list_spec->UE_template_NB_IoT+i)->prev=-1;
      (mac_inst->UE_list_spec->UE_template_NB_IoT+i)->next=1;
      //mac_inst->UE_list_spec->next[i]=-1;
    }
    else if(i>=USER_NUM_USS-1)
    {
      (mac_inst->UE_list_spec->UE_template_NB_IoT+i)->prev=i-1;
      (mac_inst->UE_list_spec->UE_template_NB_IoT+i)->next=-1;
    }
    else
    {
      (mac_inst->UE_list_spec->UE_template_NB_IoT+i)->prev=i-1;
      (mac_inst->UE_list_spec->UE_template_NB_IoT+i)->next=i+1;
    }
    //mac_inst->UE_list_spec->UE_template_NB_IoT[i]=UE_info[i];
    mac_inst->UE_list_spec->tail=i;
  }
*/
  //UL initial
  //Setting nprach configuration
  setting_nprach();
  //Initialize uplink resource from nprach configuration
  Initialize_Resource();
  //add_UL_Resource(mac_inst);    
  extend_available_resource_DL(mac_inst, mac_inst->current_subframe + 1 + 160);  

}

void mac_top_init_eNB_NB_IoT(void)
{
  // can be an input of this function, but now fix to 0
  module_id_t    i = 0;

  //UE_list_t *UE_list;
  //eNB_MAC_INST_NB_IoT *mac;

  int nb_inst_NB_IOT_MAC= 1;
  
  LOG_I(MAC,"[NB-IoT MAIN] Init function start:nb_nbiot_macrlc_inst=%d\n",nb_inst_NB_IOT_MAC);

  if (nb_inst_NB_IOT_MAC>0) {
    // only one inst exit in legacy OAI

      mac_inst->Mod_id = i;

            // IF Module Initialization linking
      mac_inst->if_inst_NB_IoT = IF_Module_init_NB_IoT(i);
      mac_inst->if_inst_NB_IoT->PHY_config_req = PHY_config_req_NB_IoT;
      mac_inst->if_inst_NB_IoT->schedule_response = schedule_response_NB_IoT;

      //reserve for fapi structure initializati


  } else {
    mac_inst = NULL;
  }
  
  // for NB-IoT UE list initialization will be in init_mac_NB_IoT

}

int l2_init_eNB_NB_IoT(void)
{


  LOG_I(MAC,"[MAIN] MAC_INIT_GLOBAL_PARAM NB-IoT IN...\n");

  mac_inst = (eNB_MAC_INST_NB_IoT *) malloc (sizeof(eNB_MAC_INST_NB_IoT));

  Is_rrc_registered_NB_IoT=0;
  mac_init_global_param_NB_IoT();
  Is_rrc_registered_NB_IoT=1;

  //init_mac_NB_IoT(mac_inst);

  return(1);
}

