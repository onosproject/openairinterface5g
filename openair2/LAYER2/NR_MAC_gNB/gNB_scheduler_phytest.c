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

/*! \file gNB_scheduler_phytest.c
 * \brief gNB scheduling procedures in phy_test mode
 * \author  Guy De Souza
 * \date 07/2018
 * \email: desouza@eurecom.fr
 * \version 1.0
 * @ingroup _mac
 */

#include "nr_mac_gNB.h"
#include "SCHED_NR/sched_nr.h"
#include "mac_proto.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "PHY/NR_TRANSPORT/nr_dci.h"

extern RAN_CONTEXT_t RC;

/*Scheduling of DLSCH with associated DCI in common search space
 * current version has only a DCI for type 1 PDCCH for C_RNTI*/
void nr_schedule_css_dlsch_phytest(module_id_t   module_idP,
                                   frame_t       frameP,
                                   sub_frame_t   slotP)
{
  uint8_t  CC_id;

  gNB_MAC_INST                        *nr_mac      = RC.nrmac[module_idP];
  //NR_COMMON_channels_t                *cc           = nr_mac->common_channels;
  nfapi_nr_dl_config_request_body_t   *dl_req;
  nfapi_nr_dl_config_request_pdu_t  *dl_config_dci_pdu;
  nfapi_nr_dl_config_request_pdu_t  *dl_config_dlsch_pdu;
  nfapi_tx_request_pdu_t            *TX_req;

  nfapi_nr_config_request_t *cfg = &nr_mac->config[0];
  uint16_t rnti = 0x1234;

  uint16_t sfn_sf = frameP << 7 | slotP;
  int dl_carrier_bandwidth = cfg->rf_config.dl_carrier_bandwidth.value;

  // everything here is hard-coded to 30 kHz
  int scs = get_dlscs(cfg);
  int slots_per_frame = get_spf(cfg);
  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    LOG_I(MAC, "Scheduling common search space DCI type 1 for CC_id %d\n",CC_id);


    dl_req = &nr_mac->DL_req[CC_id].dl_config_request_body;
    dl_config_dci_pdu = &dl_req->dl_config_pdu_list[dl_req->number_pdu];
    memset((void*)dl_config_dci_pdu,0,sizeof(nfapi_nr_dl_config_request_pdu_t));
    dl_config_dci_pdu->pdu_type = NFAPI_NR_DL_CONFIG_DCI_DL_PDU_TYPE;
    dl_config_dci_pdu->pdu_size = (uint8_t)(2+sizeof(nfapi_nr_dl_config_dci_dl_pdu));

    dl_config_dlsch_pdu = &dl_req->dl_config_pdu_list[dl_req->number_pdu+1];
    memset((void*)dl_config_dlsch_pdu,0,sizeof(nfapi_nr_dl_config_request_pdu_t));
    dl_config_dlsch_pdu->pdu_type = NFAPI_NR_DL_CONFIG_DLSCH_PDU_TYPE;
    dl_config_dlsch_pdu->pdu_size = (uint8_t)(2+sizeof(nfapi_nr_dl_config_dlsch_pdu));

    nfapi_nr_dl_config_dci_dl_pdu_rel15_t *pdu_rel15 = &dl_config_dci_pdu->dci_dl_pdu.dci_dl_pdu_rel15;
    nfapi_nr_dl_config_pdcch_parameters_rel15_t *params_rel15 = &dl_config_dci_pdu->dci_dl_pdu.pdcch_params_rel15;
    nfapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_pdu_rel15 = &dl_config_dlsch_pdu->dlsch_pdu.dlsch_pdu_rel15;

    dlsch_pdu_rel15->start_prb = 0;
    dlsch_pdu_rel15->n_prb = 50;
    dlsch_pdu_rel15->start_symbol = 2;
    dlsch_pdu_rel15->nb_symbols = 8;
    dlsch_pdu_rel15->rnti = rnti;
    dlsch_pdu_rel15->nb_layers =1;
    dlsch_pdu_rel15->nb_codewords = 1;
    dlsch_pdu_rel15->mcs_idx = 9;
    dlsch_pdu_rel15->ndi = 1;
    dlsch_pdu_rel15->redundancy_version = 0;


    nr_configure_css_dci_initial(params_rel15,
				 scs, scs, nr_FR1, 0, 0, 0,
         sfn_sf, slotP,
				 slots_per_frame,
				 dl_carrier_bandwidth);

    params_rel15->first_slot = 0;

    pdu_rel15->frequency_domain_assignment = get_RIV(dlsch_pdu_rel15->start_prb, dlsch_pdu_rel15->n_prb, cfg->rf_config.dl_carrier_bandwidth.value);
    pdu_rel15->time_domain_assignment = 3; // row index used here instead of SLIV
    pdu_rel15->vrb_to_prb_mapping = 1;
    pdu_rel15->mcs = 9;
    pdu_rel15->tb_scaling = 1;
    
    pdu_rel15->ra_preamble_index = 25;
    pdu_rel15->format_indicator = 1;
    pdu_rel15->ndi = 1;
    pdu_rel15->rv = 0;
    pdu_rel15->harq_pid = 0;
    pdu_rel15->dai = 2;
    pdu_rel15->tpc = 2;
    pdu_rel15->pucch_resource_indicator = 7;
    pdu_rel15->pdsch_to_harq_feedback_timing_indicator = 7;

    LOG_I(MAC, "[gNB scheduler phytest] DCI type 1 payload: freq_alloc %d, time_alloc %d, vrb to prb %d, mcs %d tb_scaling %d ndi %d rv %d\n",
                pdu_rel15->frequency_domain_assignment,
                pdu_rel15->time_domain_assignment,
                pdu_rel15->vrb_to_prb_mapping,
                pdu_rel15->mcs,
                pdu_rel15->tb_scaling,
                pdu_rel15->ndi,
                pdu_rel15->rv);

    params_rel15->rnti = rnti;
    params_rel15->rnti_type = NFAPI_NR_RNTI_C;
    params_rel15->dci_format = NFAPI_NR_DL_DCI_FORMAT_1_0;
    //params_rel15->aggregation_level = 1;
    LOG_D(MAC, "DCI type 1 params: rmsi_pdcch_config %d, rnti %d, rnti_type %d, dci_format %d\n \
                coreset params: mux_pattern %d, n_rb %d, n_symb %d, rb_offset %d  \n \
                ss params : nb_ss_sets_per_slot %d, first symb %d, nb_slots %d, sfn_mod2 %d, first slot %d\n",
                0,
                params_rel15->rnti,
                params_rel15->rnti_type,
                params_rel15->dci_format,
                params_rel15->mux_pattern,
                params_rel15->n_rb,
                params_rel15->n_symb,
                params_rel15->rb_offset,
                params_rel15->nb_ss_sets_per_slot,
                params_rel15->first_symbol,
                params_rel15->nb_slots,
                params_rel15->sfn_mod2,
                params_rel15->first_slot);
  nr_get_tbs(&dl_config_dlsch_pdu->dlsch_pdu, dl_config_dci_pdu->dci_dl_pdu, *cfg);
  LOG_I(MAC, "DLSCH PDU: start PRB %d n_PRB %d start symbol %d nb_symbols %d nb_layers %d nb_codewords %d mcs %d\n",
  dlsch_pdu_rel15->start_prb,
  dlsch_pdu_rel15->n_prb,
  dlsch_pdu_rel15->start_symbol,
  dlsch_pdu_rel15->nb_symbols,
  dlsch_pdu_rel15->nb_layers,
  dlsch_pdu_rel15->nb_codewords,
  dlsch_pdu_rel15->mcs_idx);

  dl_req->number_dci++;
  dl_req->number_pdsch_rnti++;
  dl_req->number_pdu+=2;

  TX_req = &nr_mac->TX_req[CC_id].tx_request_body.tx_pdu_list[nr_mac->TX_req[CC_id].tx_request_body.number_of_pdus];
  TX_req->pdu_length = 6;
  TX_req->pdu_index = nr_mac->pdu_index[CC_id]++;
  TX_req->num_segments = 1;
  TX_req->segments[0].segment_length = 8;
  nr_mac->TX_req[CC_id].tx_request_body.number_of_pdus++;
  nr_mac->TX_req[CC_id].sfn_sf = sfn_sf;
  nr_mac->TX_req[CC_id].tx_request_body.tl.tag = NFAPI_TX_REQUEST_BODY_TAG;
  nr_mac->TX_req[CC_id].header.message_id = NFAPI_TX_REQUEST;

  TX_req = &nr_mac->TX_req[CC_id].tx_request_body.tx_pdu_list[nr_mac->TX_req[CC_id].tx_request_body.number_of_pdus+1];
  TX_req->pdu_length = dlsch_pdu_rel15->transport_block_size;
  TX_req->pdu_index = nr_mac->pdu_index[CC_id]++;
  TX_req->num_segments = 1;
  TX_req->segments[0].segment_length = 8;
  nr_mac->TX_req[CC_id].tx_request_body.number_of_pdus++;
  nr_mac->TX_req[CC_id].sfn_sf = sfn_sf;
  nr_mac->TX_req[CC_id].tx_request_body.tl.tag = NFAPI_TX_REQUEST_BODY_TAG;
  nr_mac->TX_req[CC_id].header.message_id = NFAPI_TX_REQUEST;
    
  }
}

/*Scheduling of DLSCH with associated DCI in user specific search space
 * current version has only a DCI for type 1 PDCCH for C_RNTI*/
void nr_schedule_uss_dlsch_phytest(module_id_t   module_idP,
                                   frame_t       frameP,
                                   sub_frame_t   slotP)
{
  uint8_t  CC_id;

  gNB_MAC_INST                        *nr_mac      = RC.nrmac[module_idP];
  //NR_COMMON_channels_t                *cc           = nr_mac->common_channels;
  nfapi_nr_dl_config_request_body_t   *dl_req;
  nfapi_nr_dl_config_request_pdu_t  *dl_config_dci_pdu;
  nfapi_nr_dl_config_request_pdu_t  *dl_config_dlsch_pdu;
  nfapi_tx_request_pdu_t            *TX_req;

  nfapi_nr_config_request_t *cfg = &nr_mac->config[0];
  uint16_t rnti = 0x1234;

  uint16_t sfn_sf = frameP << 7 | slotP;
  int dl_carrier_bandwidth = cfg->rf_config.dl_carrier_bandwidth.value;

  
  
  //socket_mac_gNB_data *socket_data;
  //socket_data = calloc(1, sizeof(*socket_data));
  //memset((void *)socket_data, 0, sizeof(socket_data));
  const char *ip_address = "10.102.81.239";
  const uint16_t port = 53108;
  //const uint16_t port = 554; //RTSP port
  //const uint16_t port = 50202; //Custom port
  char* pdu[5888] = {0};
    
  //Connect socket
  //connect_mac_socket_to_data_source(ip_address, port, nr_mac->socket_mac_data);
    
    
  
  
  // everything here is hard-coded to 30 kHz
  int scs = get_dlscs(cfg);
  int slots_per_frame = get_spf(cfg);
  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    LOG_I(MAC, "Scheduling UE specific search space DCI type 1 for CC_id %d\n",CC_id);

    nfapi_nr_coreset_t* coreset = &nr_mac->coreset[CC_id][1];
    nfapi_nr_search_space_t* search_space = &nr_mac->search_space[CC_id][1];

    dl_req = &nr_mac->DL_req[CC_id].dl_config_request_body;
    dl_config_dci_pdu = &dl_req->dl_config_pdu_list[dl_req->number_pdu];
    memset((void*)dl_config_dci_pdu,0,sizeof(nfapi_nr_dl_config_request_pdu_t));
    dl_config_dci_pdu->pdu_type = NFAPI_NR_DL_CONFIG_DCI_DL_PDU_TYPE;
    dl_config_dci_pdu->pdu_size = (uint8_t)(2+sizeof(nfapi_nr_dl_config_dci_dl_pdu));

    dl_config_dlsch_pdu = &dl_req->dl_config_pdu_list[dl_req->number_pdu+1];
    memset((void*)dl_config_dlsch_pdu,0,sizeof(nfapi_nr_dl_config_request_pdu_t));
    dl_config_dlsch_pdu->pdu_type = NFAPI_NR_DL_CONFIG_DLSCH_PDU_TYPE;
    dl_config_dlsch_pdu->pdu_size = (uint8_t)(2+sizeof(nfapi_nr_dl_config_dlsch_pdu));

    nfapi_nr_dl_config_dci_dl_pdu_rel15_t *pdu_rel15 = &dl_config_dci_pdu->dci_dl_pdu.dci_dl_pdu_rel15;
    nfapi_nr_dl_config_pdcch_parameters_rel15_t *params_rel15 = &dl_config_dci_pdu->dci_dl_pdu.pdcch_params_rel15;
    nfapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_pdu_rel15 = &dl_config_dlsch_pdu->dlsch_pdu.dlsch_pdu_rel15;

    dlsch_pdu_rel15->start_prb = 0;
    dlsch_pdu_rel15->n_prb = 50;
    dlsch_pdu_rel15->start_symbol = 2;
    dlsch_pdu_rel15->nb_symbols = 9;
    dlsch_pdu_rel15->rnti = rnti;
    dlsch_pdu_rel15->nb_layers =1;
    dlsch_pdu_rel15->nb_codewords = 1;
    dlsch_pdu_rel15->mcs_idx = 9;
    dlsch_pdu_rel15->ndi = 1;
    dlsch_pdu_rel15->redundancy_version = 0;

    nr_configure_dci_from_pdcch_config(params_rel15,
                                       coreset,
                                       search_space,
                                       *cfg,
                                       dl_carrier_bandwidth);

    pdu_rel15->frequency_domain_assignment = get_RIV(dlsch_pdu_rel15->start_prb, dlsch_pdu_rel15->n_prb, cfg->rf_config.dl_carrier_bandwidth.value);
    pdu_rel15->time_domain_assignment = 3; // row index used here instead of SLIV;
    pdu_rel15->vrb_to_prb_mapping = 1;
    pdu_rel15->mcs = 9;
    pdu_rel15->tb_scaling = 1;

    pdu_rel15->ra_preamble_index = 25;
    pdu_rel15->format_indicator = 1;
    pdu_rel15->ndi = 1;
    pdu_rel15->rv = 0;
    pdu_rel15->harq_pid = 0;
    pdu_rel15->dai = 2;
    pdu_rel15->tpc = 2;
    pdu_rel15->pucch_resource_indicator = 7;
    pdu_rel15->pdsch_to_harq_feedback_timing_indicator = 7;

    LOG_I(MAC, "[gNB scheduler phytest] DCI type 1 payload: freq_alloc %d, time_alloc %d, vrb to prb %d, mcs %d tb_scaling %d ndi %d rv %d\n",
                pdu_rel15->frequency_domain_assignment,
                pdu_rel15->time_domain_assignment,
                pdu_rel15->vrb_to_prb_mapping,
                pdu_rel15->mcs,
                pdu_rel15->tb_scaling,
                pdu_rel15->ndi,
                pdu_rel15->rv);

    params_rel15->rnti = rnti;
    params_rel15->rnti_type = NFAPI_NR_RNTI_C;
    params_rel15->dci_format = NFAPI_NR_DL_DCI_FORMAT_1_0;

    //params_rel15->aggregation_level = 1;
    LOG_I(MAC, "DCI params: rnti %d, rnti_type %d, dci_format %d, config type %d\n \
                coreset params: mux_pattern %d, n_rb %d, n_symb %d, rb_offset %d  \n \
                ss params : first symb %d, ss type %d\n",
                params_rel15->rnti,
                params_rel15->rnti_type,
                params_rel15->config_type,
                params_rel15->dci_format,
                params_rel15->mux_pattern,
                params_rel15->n_rb,
                params_rel15->n_symb,
                params_rel15->rb_offset,
                params_rel15->first_symbol,
                params_rel15->search_space_type);
  nr_get_tbs(&dl_config_dlsch_pdu->dlsch_pdu, dl_config_dci_pdu->dci_dl_pdu, *cfg);
  LOG_I(MAC, "DLSCH PDU: start PRB %d n_PRB %d start symbol %d nb_symbols %d nb_layers %d nb_codewords %d mcs %d\n",
  dlsch_pdu_rel15->start_prb,
  dlsch_pdu_rel15->n_prb,
  dlsch_pdu_rel15->start_symbol,
  dlsch_pdu_rel15->nb_symbols,
  dlsch_pdu_rel15->nb_layers,
  dlsch_pdu_rel15->nb_codewords,
  dlsch_pdu_rel15->mcs_idx);

  dl_req->number_dci++;
  dl_req->number_pdsch_rnti++;
  dl_req->number_pdu+=2;
  
  
  
  //while(1){

  
    //Call get_mac_pdu_from_socket() around here?
    //LOG_I(MAC, "get_mac_pdu_from_socket\n");
    get_mac_pdu_from_socket(pdu, 5888, nr_mac->socket_mac_data);
    //get_mac_pdu_from_socket(pdu, dl_config_dlsch_pdu->dlsch_pdu.dlsch_pdu_rel15.transport_block_size, socket_mac_gNB_data *socket_data);
  
  //}
  
  

  TX_req = &nr_mac->TX_req[CC_id].tx_request_body.tx_pdu_list[nr_mac->TX_req[CC_id].tx_request_body.number_of_pdus];
  TX_req->pdu_length = 6;
  TX_req->pdu_index = nr_mac->pdu_index[CC_id]++;
  TX_req->num_segments = 1;
  TX_req->segments[0].segment_length = 8;
  nr_mac->TX_req[CC_id].tx_request_body.number_of_pdus++;
  nr_mac->TX_req[CC_id].sfn_sf = sfn_sf;
  nr_mac->TX_req[CC_id].tx_request_body.tl.tag = NFAPI_TX_REQUEST_BODY_TAG;
  nr_mac->TX_req[CC_id].header.message_id = NFAPI_TX_REQUEST;

  TX_req = &nr_mac->TX_req[CC_id].tx_request_body.tx_pdu_list[nr_mac->TX_req[CC_id].tx_request_body.number_of_pdus+1];
  TX_req->pdu_length = dlsch_pdu_rel15->transport_block_size;
  TX_req->pdu_index = nr_mac->pdu_index[CC_id]++;
  TX_req->num_segments = 1;
  TX_req->segments[0].segment_length = 8;
  nr_mac->TX_req[CC_id].tx_request_body.number_of_pdus++;
  nr_mac->TX_req[CC_id].sfn_sf = sfn_sf;
  nr_mac->TX_req[CC_id].tx_request_body.tl.tag = NFAPI_TX_REQUEST_BODY_TAG;
  nr_mac->TX_req[CC_id].header.message_id = NFAPI_TX_REQUEST;

  }
  

  //disconnect_mac_socket_from_data_source(nr_mac->socket_mac_data);
  
}




/* Function to get the MAC PDU from socket data */
int get_mac_pdu_from_socket(char* pdu,
		                    uint32_t TBS, 
	                    	socket_mac_gNB_data *socket_data)
{

	//LOG_I(MAC, "[get_mac_pdu_from_socket] TEST\n");
	/* Should I set the socket as non-blocking? */
	//fcntl(socket_data->sd, F_SETFL, O_NONBLOCK);
	
	//If listening to more than one socket (e.g. for TCP and UDP listening) select() should be used
	
	int recv_ret;
	//struct sockaddr *from_addr;
	struct sockaddr_in from_addr;
	socklen_t from_addr_len;
	from_addr_len = sizeof(struct sockaddr_in);

	//LOG_I(MAC, "[get_mac_pdu_from_socket] TEST1, socket_data->sd=%d\n", socket_data->sd);
	//recv_ret = recv(socket_data->sd, pdu, (size_t)TBS, 0);
	recv_ret = recvfrom(socket_data->sd, pdu, (size_t)TBS, 0, &from_addr, &from_addr_len);
	//LOG_I(MAC, "[get_mac_pdu_from_socket] TEST2\n");
	if (recv_ret == -1) {
		/* Failure case */
		
		switch (errno) {
			//case EWOULDBLOCK:
			case EAGAIN:
				return -1;
			default:
				//g_info("recv failed: %s", g_strerror(errno));
				LOG_I(MAC, "[get_mac_pdu_from_socket] ERROR. Received %d bytes in MAC socket. Errno=%d\n", recv_ret, errno);
				return -1;
				break;
			
		}
	//} else if (recv_ret == 0) {
	//	/* We lost the connection with other peer or shutdown asked */
	//	ui_pipe_write_message(socket_data->pipe_fd,
	//						  UI_PIPE_CONNECTION_LOST, NULL, 0);
	//	free(socket_data->ip_address);
	//	free(socket_data);
	//	pthread_exit(NULL);
	}else{
		//Source IP address could be checked to only transmit RTP packets for video flow
		LOG_I(MAC, "[get_mac_pdu_from_socket] Received %d bytes in MAC socket.\n", recv_ret);
	}
	

	return recv_ret;
	
}

/* Function to open a socket and connect to the MAC data source*/
int connect_mac_socket_to_data_source(const char *ip_address,
		                              const uint16_t port, socket_mac_gNB_data *socket_mac)
{
	
	//LOG_I(MAC, "[connect_mac_socket_to_data_source]TEST\n");
	//socket_mac_gNB_data *socket_data;
	//socket_data = calloc(1, sizeof(*socket_data));
	socket_mac->ip_address = strdup(ip_address);
	socket_mac->port    = port;
	socket_mac->sd      = -1;
	struct sockaddr_in  si_me;

	//g_assert(socket_data != NULL);
	
	/* Preparing the socket */
	//if ((socket_data->sd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1) {
	if ((socket_mac->sd = socket(AF_INET, SOCK_DGRAM, 0)) == -1) {
		//g_warning("socket failed: %s", g_strerror(errno));
		LOG_I(MAC, "socket failed\n");
		free(socket_mac->ip_address);
		free(socket_mac);
		return RESULT_FAILED;
	}
	memset((void *)&si_me, 0, sizeof(si_me));

	si_me.sin_family = AF_INET;
	si_me.sin_port = htons(socket_mac->port);
	//si_me.sin_addr.s_addr = htonl(INADDR_ANY);
	//si_me.sin_addr.s_addr = htonl(socket_data->ip_address);
	si_me.sin_addr.s_addr = inet_addr(socket_mac->ip_address);
	
	if (inet_aton(socket_mac->ip_address, &si_me.sin_addr) == 0) {
		//g_warning("inet_aton() failed\n");
		LOG_I(MAC, "inet_aton() failed\n");
		free(socket_mac->ip_address);
		free(socket_mac);
		return RESULT_FAILED;
	}
	
	if(bind(socket_mac->sd, (struct sockaddr *)&si_me, sizeof(struct sockaddr_in)) == -1){
		//g_warning("binding socket failed\n");
		LOG_I(MAC, "binding socket failed\n");
		free(socket_mac->ip_address);
		free(socket_mac);
		return RESULT_FAILED;
	}	
	
	//Set socket as non-blocking
	fcntl(socket_mac->sd, F_SETFL, O_NONBLOCK);
	
	//If more than one socket, there should be a set of sockets to use select() with them
	//The socket should be added to the set in this function.
	
	//memcpy(socket_mac, socket_data, sizeof(socket_data));
	//&socket_mac = &socket_data;
	LOG_I(MAC, "[connect_mac_socket_to_data_source]MAC socket connected; socket_mac->sd= %d\n", socket_mac->sd);
		
	return RESULT_OK;
}

/* Function to close the MAC data source socket */
int disconnect_mac_socket_from_data_source(socket_mac_gNB_data *socket_data)
{
	close(socket_data->sd);
	free(socket_data->ip_address);
	free(socket_data);
	
	LOG_I(MAC, "[disconnect_mac_socket_from_data_source] Disconnecting MAC socket\n");
	
	return RESULT_OK;
}


