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


#include "openair2/UTIL/MEM/mem_block.h"
#include "openair2/LAYER2/PDCP_v10.1.0/pdcp.h"
#include "openair2/LAYER2/PDCP_v10.1.0/pdcp_sequence_manager.h"
#include "openair2/UTIL/OSA/osa_defs.h"

#include "common/config/config_userapi.c"


#define DUMMY_BUFFER ((unsigned char*)"123456789")
#define DUMMY_BUFFER_SIZE 10



pdcp_t pdcp_el;
list_t pdu_tx_list;

int main(int argc, char *argv[])
{
	int resQ;
	do{
		printf("Type in the test you want (1 or 2)");
		scanf("%d",&resQ);
	}while(resQ!=1 ||resQ != 2);
	pool_buffer_init();
	list_init(&pdu_tx_list, NULL);
	logInit();

	pdcp_el.next_pdcp_tx_sn = 0;
	pdcp_el.next_pdcp_rx_sn = 0;
	pdcp_el.tx_hfn = 0;
	pdcp_el.rx_hfn = 0;
	pdcp_el.last_submitted_pdcp_rx_sn = 4095;
	pdcp_el.seq_num_size = 12;
	pdcp_el.cipheringAlgorithm = (resQ==1?EEA1_128_ALG_ID:EEA2_128_ALG_ID);
	
	pdcp_init_seq_numbers(&pdcp_el);
	
	
	pdcp_data_req(NULL, //ctxt_pP
		      0,    //srb_flagP
	              3,    // rb_id
	              0,    // muiP
	              0,    //confirmP
                      DUMMY_BUFFER_SIZE, //sdu_buffer_size
                      DUMMY_BUFFER,  // sdu_buffer 
                      PDCP_TRANSMISSION_MODE_DATA, // pdcp_transmission_mod
                      0,0);


}
