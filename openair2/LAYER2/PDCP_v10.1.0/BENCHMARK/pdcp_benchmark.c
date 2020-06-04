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





#define DUMMY_BUFFER ((unsigned char*)"123456789")
#define DUMMY_BUFFER_SIZE 10

typedef struct {
  struct mem_block_t *head;
  struct mem_block_t *tail;
  int                nb_elements;
  char               name[LIST_NAME_MAX_CHAR];
} list_t;

typedef struct pdcp_s {
  //boolean_t     instanciated_instance;
  uint16_t       header_compression_profile;

  /* SR: added this flag to distinguish UE/eNB instance as pdcp_run for virtual
   * mode can receive data on NETLINK for eNB while eNB_flag = 0 and for UE when eNB_flag = 1
   */
  boolean_t is_ue;
  boolean_t is_srb;

  /* Configured security algorithms */
  uint8_t cipheringAlgorithm;
  uint8_t integrityProtAlgorithm;

  /* User-Plane encryption key
   * Control-Plane RRC encryption key
   * Control-Plane RRC integrity key
   * These keys are configured by RRC layer
   */
  uint8_t *kUPenc;
  uint8_t *kRRCint;
  uint8_t *kRRCenc;

  uint8_t security_activated;

  rlc_mode_t rlc_mode;
  uint8_t status_report;
  uint8_t seq_num_size;

  logical_chan_id_t lcid;
  rb_id_t           rb_id;
  /*
   * Sequence number state variables
   *
   * TX and RX window
   */
  pdcp_sn_t next_pdcp_tx_sn;
  pdcp_sn_t next_pdcp_rx_sn;
  pdcp_sn_t maximum_pdcp_rx_sn;
  /*
   * TX and RX Hyper Frame Numbers
   */
  pdcp_hfn_t tx_hfn;
  pdcp_hfn_t rx_hfn;

  /*
   * SN of the last PDCP SDU delivered to upper layers
   */
  pdcp_sn_t  last_submitted_pdcp_rx_sn;

  /*
   * Following array is used as a bitmap holding missing sequence
   * numbers to generate a PDCP Control PDU for PDCP status
   * report (see 6.2.6)
   */
  uint8_t missing_pdu_bitmap[512];
  /*
   * This is intentionally signed since we need a 'NULL' value
   * which is not also a valid sequence number
   */
  short int first_missing_pdu;

} pdcp_t;


pdcp_t pdcp_el;
list_t pdu_tx_list;

int main(int argc, char *argv[])
{
	int resQ;
	do{
		printf("Type in the test you want (1 or 2)");
		scanf("%d",resQ);
	}while(resQ!=1 ||resQ != 2);
	pool_buffer_init();
	list_init(&pdu_tx_list, NULL);
	logInit();

	pdcp_el->next_pdcp_tx_sn = 0;
	pdcp_el->next_pdcp_rx_sn = 0;
	pdcp_el->tx_hfn = 0;
	pdcp_el->rx_hfn = 0;
	/* SN of the last PDCP SDU delivered to upper layers */
	pdcp_el->last_submitted_pdcp_rx_sn = 4095;
	pdcp_el->seq_num_size = 12;
	pdcp_el->cipheringAlgorithm = (resQ==1?"EEA1_128_ALG_ID":"EEA2_128_ALG_ID");
	pdcp_data_req(0, 0, 10, DUMMY_BUFFER, pdcp_el, &pdu_tx_list);



}
