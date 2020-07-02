/*
 *  * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 *  * contributor license agreements.  See the NOTICE file distributed with
 *  * this work for additional information regarding copyright ownership.
 *  * The OpenAirInterface Software Alliance licenses this file to You under
 *  * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 *  * except in compliance with the License.
 *  * You may obtain a copy of the License at
 *  *
 *  *      http://www.openairinterface.org/?page_id=698
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  * See the License for the specific language governing permissions and
 *  * limitations under the License.
 *  *-------------------------------------------------------------------------------
 *  * For more information about the OpenAirInterface (OAI) Software Alliance:
 *  *      contact@openairinterface.org
 *  */

/*! \file pdcp_security.c
 *  * \brief PDCP Performance Benchmark 
 *  * \author 
 *  * \email 
 *  * \date 2020
 *  */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "openair2/COMMON/platform_types.h"
#include "common/config/config_load_configmodule.h"
#include "common/utils/LOG/log.h"

#include "openair2/UTIL/MEM/mem_block.h"
#include "openair2/LAYER2/RLC/rlc.h"
#include "openair2/LAYER2/PDCP_v10.1.0/pdcp.h"
#include "openair2/LAYER2/PDCP_v10.1.0/pdcp_sequence_manager.h"
#include "openair2/UTIL/OSA/osa_defs.h"

#include "common/config/config_load_configmodule.h"
#include "common/utils/LOG/log.h"
#include "SIMULATION/TOOLS/sim.h"
#include "UTIL/LISTS/list.h"
#include "OCG_vars.h"

#include "common/config/config_userapi.h"

#include "LTE_RLC-Config.h"
#include "LTE_DRB-ToAddMod.h"
#include "LTE_DRB-ToAddModList.h"
#include "LTE_SRB-ToAddMod.h"
#include "LTE_SRB-ToAddModList.h"
#include "LTE_DL-UM-RLC.h"
#include "LTE_PMCH-InfoList-r9.h"

#include <time.h>
//#include "time_meas.h"

//BUFFER 10

#define DUMMY_BUFFER ((unsigned char*)"123456789")
#define DUMMY_BUFFER_SIZE 10
//BUFFER 50
/*
#define DUMMY_BUFFER ((unsigned char*)"1234567890123456789012345678901234567890123456789")
#define DUMMY_BUFFER_SIZE 50*/
//BUFFER 100                   

/*#define DUMMY_BUFFER ((unsigned char*)"123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789")
#define DUMMY_BUFFER_SIZE 100    
*/

#define NB_REPEAT 1

uint64_t get_softmodem_optmask(void) {return 0;}
nfapi_mode_t nfapi_getmode(void) {return 0;}
softmodem_params_t *get_softmodem_params(void) {return NULL;}

rlc_op_status_t rrc_rlc_config_req   (
    const protocol_ctxt_t *const ctxt_pP,
  const srb_flag_t      srb_flagP,
  const MBMS_flag_t     mbms_flagP,
  const config_action_t actionP,
  const rb_id_t         rb_idP,
  const rlc_info_t      rlc_infoP) {return 0;}

rlc_op_status_t rrc_rlc_config_asn1_req (const protocol_ctxt_t    *const ctxt_pP,
    const LTE_SRB_ToAddModList_t    *const srb2add_listP,
    const LTE_DRB_ToAddModList_t    *const drb2add_listP,
    const LTE_DRB_ToReleaseList_t   *const drb2release_listP,
    const LTE_PMCH_InfoList_r9_t *const pmch_InfoList_r9_pP,
    const uint32_t sourceL2Id,
    const uint32_t destinationL2Id

                                        ) { return 0;}
                                        
                                        
rlc_op_status_t rlc_data_req     (const protocol_ctxt_t *const ctxt_pP,
                                  const srb_flag_t   srb_flagP,
                                  const MBMS_flag_t  MBMS_flagP,
                                  const rb_id_t      rb_idP,
                                  const mui_t        muiP,
                                  confirm_t    confirmP,
                                  sdu_size_t   sdu_sizeP,
                                  mem_block_t *sdu_pP,
                                  const uint32_t *const sourceL2Id,
                                  const uint32_t *const destinationL2Id
                                 ) {return 0;}

void
rrc_data_ind(
  const protocol_ctxt_t *const ctxt_pP,
  const rb_id_t                Srb_id,
  const sdu_size_t             sdu_sizeP,
  const uint8_t   *const       buffer_pP
) {return;}


void rlc_util_print_hex_octets(
  const comp_name_t componentP,
  unsigned char *const dataP,
  const signed long sizeP) {return;}

unsigned char NB_eNB_INST=1;
uint16_t NB_UE_INST=1;
int mbms_rab_id = 2047;
int otg_enabled=0;
int opp_enabled=0;
pdcp_t pdcp_el;
list_t pdu_tx_list;
//Not sure at all about that, but I didn't knew which library I should include and I try with that. 
volatile int oai_exit=0;

//Same thing, but found it like that elsewhere
void exit_function(const char *file, const char *function, const int line, const char *s) {
}

#include "common/ran_context.h"
RAN_CONTEXT_t RC;

int main(int argc, char *argv[])
{
 //   printf("1");
/*	int resQ;
	resQ=1;*/
int ticks;
	/*if(argc < 2)
	{
	    printf("You should pass the test you want as parameter");
	}
	resQ = atoi(argv[1]);
	*/pool_buffer_init();
	list_init(&pdu_tx_list, NULL);
//	printf("2");
	logInit();
  //  printf("3");
	/*pdcp_el.next_pdcp_tx_sn = 0;
	pdcp_el.next_pdcp_rx_sn = 0;
	pdcp_el.tx_hfn = 0;
	pdcp_el.rx_hfn = 0;
	pdcp_el.last_submitted_pdcp_rx_sn = 4095;
	pdcp_el.seq_num_size = 12;
	pdcp_el.cipheringAlgorithm = (resQ==1?EEA1_128_ALG_ID:EEA2_128_ALG_ID);
//	printf("4");
	pdcp_init_seq_numbers(&pdcp_el);*/
//	printf("5");
/*
	protocol_ctxt_t ctxt;
    ctxt.module_id = 0 ;
    ctxt.instance = 0;
    ctxt.rnti = 0;
    ctxt.enb_flag = 1;
    ctxt.frame = 0;
    ctxt.subframe = 0;*/

//Not working...???
/*
time_stats_t *t = malloc(sizeof(time_stats_t));
reset_meas(t);
	start_meas(t);*/

clock_t t;
t=clock();
	for(int i=0; i< NB_REPEAT; i++){
	
protocol_ctxt_t ctxt;
    ctxt.module_id = 0 ;
    ctxt.instance = 0;
    ctxt.rnti = 0;
    ctxt.enb_flag = 1;
    ctxt.frame = 0;
    ctxt.subframe = 0;
pdcp_data_req(&ctxt, //ctxt_pP
		      0,    //srb_flagP
	              3,    // rb_id
	              0,    // muiP
	              0,    //confirmP
                      DUMMY_BUFFER_SIZE, //sdu_buffer_size
                      DUMMY_BUFFER,  // sdu_buffer 
                      PDCP_TRANSMISSION_MODE_DATA, // pdcp_transmission_mod
                      0,0);
}
t=clock()-t;
//	srand(time(0));

/*stop_meas(t);
ticks = (int) (t->diff/NB_REPEAT);
	printf("Average time : %d\n", ticks);*/
printf("%d \n",(int)t);


}
