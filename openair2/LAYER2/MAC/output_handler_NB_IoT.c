
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

/*! \file output_handler_NB_IoT.c
 * \brief Convert MAC scheduler result and output the FAPI structure by subframe based
 * \author  NTUST BMW Lab./
 * \date 2017
 * \email: 
 * \version 1.0
 *
 */

#include "defs_NB_IoT.h"
#include "proto_NB_IoT.h"
#include "extern_NB_IoT.h"
#include "openair2/RRC/LITE/proto_NB_IoT.h"
#include "openair2/PHY_INTERFACE/IF_Module_NB_IoT.h"

int output_handler(eNB_MAC_INST_NB_IoT *mac_inst, module_id_t module_id, int CC_id, uint32_t hypersfn, uint32_t frame, uint32_t subframe, uint8_t MIB_flag, uint8_t SIB1_flag, uint32_t current_time){
	uint8_t MIB_size = 0;
	uint8_t SIB1_size = 0, i = 0;
	rrc_eNB_carrier_data_NB_IoT_t *carrier = &eNB_rrc_inst_NB_IoT->carrier[0];
	
	Sched_Rsp_NB_IoT_t *SCHED_info = &mac_inst->Sched_INFO;
	
	nfapi_dl_config_request_pdu_t *dl_config_pdu;
	nfapi_hi_dci0_request_pdu_t* hi_dci0_pdu;
	nfapi_ul_config_request_pdu_t* ul_config_pdu = NULL;
	schedule_result_t *tmp;
	int DL_empty = 0, UL_empty = 0; 
	int flag_malloc = 0 ;
	// filled common part of schedule_resoponse
	
	SCHED_info->module_id = module_id;
	SCHED_info->hypersfn = hypersfn;
	SCHED_info->frame = frame;
	SCHED_info->subframe = subframe;
	void *DCI_pdu;

	if(flag_malloc)
	{
		free(SCHED_info->TX_req->tx_request_body.tx_pdu_list);
	    free(SCHED_info->HI_DCI0_req->hi_dci0_request_body.hi_dci0_pdu_list);
	    free(SCHED_info->DL_req->dl_config_request_body.dl_config_pdu_list);
	    free(SCHED_info->UL_req->ul_config_request_body.ul_config_pdu_list);
	    
	    free(SCHED_info->TX_req);
	    free(SCHED_info->HI_DCI0_req);
	    free(SCHED_info->DL_req);
	    free(SCHED_info->UL_req);
	}


	SCHED_info->TX_req = (nfapi_tx_request_t *) malloc (sizeof(nfapi_tx_request_t));
	SCHED_info->TX_req->tx_request_body.tx_pdu_list = (nfapi_tx_request_pdu_t *) malloc (sizeof(nfapi_tx_request_pdu_t));
	SCHED_info->UL_req = (nfapi_ul_config_request_t*) malloc (sizeof(nfapi_ul_config_request_t));
  	SCHED_info->UL_req->ul_config_request_body.ul_config_pdu_list = (nfapi_ul_config_request_pdu_t *)malloc(5 * sizeof(nfapi_ul_config_request_pdu_t));
  	SCHED_info->HI_DCI0_req = (nfapi_hi_dci0_request_t*) malloc (sizeof(nfapi_hi_dci0_request_t));
  	SCHED_info->HI_DCI0_req->hi_dci0_request_body.hi_dci0_pdu_list = (nfapi_hi_dci0_request_pdu_t*)malloc(sizeof(nfapi_hi_dci0_request_pdu_t));
	SCHED_info->DL_req = (nfapi_dl_config_request_t*) malloc (sizeof(nfapi_dl_config_request_t));
	SCHED_info->DL_req->dl_config_request_body.dl_config_pdu_list = (nfapi_dl_config_request_pdu_t*)malloc(sizeof(nfapi_dl_config_request_pdu_t));
	flag_malloc = 1;
	
	SCHED_info->DL_req->dl_config_request_body.number_pdu = 0;
  	SCHED_info->UL_req->ul_config_request_body.number_of_pdus = 0;
  	SCHED_info->HI_DCI0_req->hi_dci0_request_body.number_of_dci = 0;
  	
  	//	process downlink data transmission, there will only be single DL_REQ in one subframe (e.g. 1ms), check common signal first
	if(subframe == 0 /*MIB_flag == 1*/)	//	TODO back to MIB_flag
	{
		uint8_t *MIB_pdu = get_NB_IoT_MIB(carrier,1,subframe,frame,hypersfn);
		//LOG_D(MAC,"[%d]MIB\n",current_time);
		//MIB_size = mac_rrc_data_req_eNB_NB_IoT(*MIB);
		//SCHED_info->DL_req = (nfapi_dl_config_request_t*) malloc (sizeof(nfapi_dl_config_request_t));
		//SCHED_info->DL_req->dl_config_request_body.number_pdu = 0;
		//SCHED_info->DL_req->dl_config_request_body.dl_config_pdu_list = (nfapi_dl_config_request_pdu_t*)malloc(sizeof(nfapi_dl_config_request_pdu_t));
		
		MIB_size = get_NB_IoT_MIB_size();
		//LOG_I(MAC,"NB-IoT get MIB size \n");
		dl_config_pdu = SCHED_info->DL_req->dl_config_request_body.dl_config_pdu_list;
    	SCHED_info->DL_req->dl_config_request_body.number_pdu = 1;
		dl_config_pdu->pdu_type                                               	 = NFAPI_DL_CONFIG_NBCH_PDU_TYPE;
  		dl_config_pdu->pdu_size                                               	 = 2+sizeof(nfapi_dl_config_nbch_pdu_rel13_t);
  		dl_config_pdu->nbch_pdu.nbch_pdu_rel13.length                            = MIB_size;
  		dl_config_pdu->nbch_pdu.nbch_pdu_rel13.pdu_index                         = 1;
  		dl_config_pdu->nbch_pdu.nbch_pdu_rel13.transmission_power                = 6000;
  		// fill MIB PDU
  		//LOG_I(MAC,"B NB-IoT get MIB pdu \n");
  		//MIB_pdu = get_NB_IoT_MIB();
  		//LOG_D(MAC,"A NB-IoT MIB pdu : %d \n",MIB_pdu);
  		//printf("NB-IoT output MIB:%02X , size :%d\n",MIB_pdu, MIB_size);
  		//SCHED_info->TX_req->tx_request_body.tx_pdu_list[dl_config_pdu->NB_IoTch_pdu.NB_IoTch_pdu_rel13.pdu_index].segments[0].segment_data = MIB_pdu;
  		
  		SCHED_info->TX_req->tx_request_body.tx_pdu_list[dl_config_pdu->nbch_pdu.nbch_pdu_rel13.pdu_index].segments[0].segment_data = MIB_pdu;
  		LOG_D(MAC,"NB-IoT fill MIB\n");
  		//LOG_I(MAC,"NB-IoT fill MIB\n");
  		//dl_scheduled(mac_inst->current_subframe, _NPBCH, 0, "MIB");
	}
	else if(SIB1_flag==1)
	//else if((subframe == 4)  && (frame%2==0) && (frame%32<16) /*SIB1_flag == 1*/)	//	TODO back to SIB1_flag
	{
		uint8_t *SIB1_pdu = get_NB_IoT_SIB1(0,0,carrier,208,92,1,3584,28,2,subframe,frame,hypersfn);
		//SIB1_size = mac_rrc_data_req_eNB_NB_IoT(*SIB1);
		//SCHED_info->DL_req = (nfapi_dl_config_request_t*) malloc (sizeof(nfapi_dl_config_request_t));
		//SCHED_info->DL_req->dl_config_request_body.number_pdu = 0;
		//SCHED_info->DL_req->dl_config_request_body.dl_config_pdu_list = (nfapi_dl_config_request_pdu_t*)malloc(sizeof(nfapi_dl_config_request_pdu_t));
	
		SIB1_size = get_NB_IoT_SIB1_size();
		dl_config_pdu = SCHED_info->DL_req->dl_config_request_body.dl_config_pdu_list;
		SCHED_info->DL_req->dl_config_request_body.number_pdu = 1;
		dl_config_pdu->pdu_type                                                   = NFAPI_DL_CONFIG_NDLSCH_PDU_TYPE;
		dl_config_pdu->pdu_size                                                   = 2+sizeof(nfapi_dl_config_ndlsch_pdu_rel13_t);
		dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.length                         = SIB1_size;
		dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.pdu_index					  = 1;
		dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.rnti_type                      = 0;
		dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.rnti                           = 0xFFFF; // SI-rnti
		dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.modulation                     = 2;
		dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.repetition_number              = 10; //value between 0-15, should be get from MIB // to fix later !!!!!!!!!!!!

		dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.number_of_subframes_for_resource_assignment     = 8;
		//SIB1_pdu = get_NB_IoT_SIB1();
		//LOG_D(MAC,"A NB-IoT SIB1 pdu : %d \n",SIB1_pdu);
		
		SCHED_info->TX_req->tx_request_body.tx_pdu_list[dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.pdu_index].segments[0].segment_data = SIB1_pdu;
		LOG_D(MAC,"NB-IoT fill SIB1\n");
		//LOG_I(MAC,"NB-IoT fill SIB1\n");
		//start symbol, Resource assignment, Repetition number, Number of subframe Resource assignment lost for now
		//dl_scheduled(mac_inst->current_subframe, _NPDSCH, SI_RNTI, "SIB1");
	}else if(schedule_result_list_DL == NULL){
		DL_empty = 1;
		//LOG_D(MAC,"no remaining node of DL scheduling result\n");
	}else{
		//	here shouldn't be run into
		if(schedule_result_list_DL->output_subframe < current_time)
		{
			while(schedule_result_list_DL->output_subframe < current_time)
			{
				//LOG_D(MAC,"This error if there is DL scheduling result node before the current time\n");
				tmp = schedule_result_list_DL;
				schedule_result_list_DL = schedule_result_list_DL->next;
				free(tmp);
				//LOG_D(MAC,"test2 current: %d\n",current_time);
				//break;
				if(schedule_result_list_DL == NULL){
					return -1;
				}
			}
			//LOG_D(MAC,"return\n");
			//return -1;
		}
		else if (schedule_result_list_DL->output_subframe == current_time)
		{
			switch(schedule_result_list_DL->channel)
			{
				case NPDCCH:

					if(schedule_result_list_DL->direction == DL)
					{
						LOG_D(MAC,"NB-IoT fill DL_DCI\n");
						//LOG_D(MAC,"Sched Info DL DCI here\n");
						//SCHED_info->DL_req = (nfapi_dl_config_request_t*) malloc (sizeof(nfapi_dl_config_request_t));
						//SCHED_info->DL_req->dl_config_request_body.number_pdu = 0;
						//SCHED_info->DL_req->dl_config_request_body.dl_config_pdu_list = (nfapi_dl_config_request_pdu_t*)malloc(sizeof(nfapi_dl_config_request_pdu_t));
	
						dl_config_pdu = SCHED_info->DL_req->dl_config_request_body.dl_config_pdu_list;
						SCHED_info->DL_req->dl_config_request_body.number_dci = 1;
						DCI_pdu = schedule_result_list_DL->DCI_pdu;
						// not consider the case transmitting 2 DCIs for the moment also not consider N2 now
						dl_config_pdu->pdu_type                                                          = NFAPI_DL_CONFIG_NPDCCH_PDU_TYPE;
						dl_config_pdu->pdu_size                                                          = 2+sizeof(nfapi_dl_config_npdcch_pdu_rel13_t);
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.length				                 = schedule_result_list_DL->sdu_length;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.pdu_index                             = 1;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.ncce_index                            = 0;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.aggregation_level     				 = 1;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.rnti_type                             = schedule_result_list_DL->rnti_type; // 3 = UE-specific RNTI TODO: add RNTI type in scheduling result
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.rnti                                  = schedule_result_list_DL->rnti;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.dci_format                            = 0; // N1
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.scheduling_delay                      = ((DCIFormatN1_t *)DCI_pdu)->Scheddly;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.resource_assignment                   = ((DCIFormatN1_t *)DCI_pdu)->ResAssign;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.repetition_number                     = ((DCIFormatN1_t *)DCI_pdu)->RepNum;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.mcs                                   = ((DCIFormatN1_t *)DCI_pdu)->mcs;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.new_data_indicator                    = ((DCIFormatN1_t *)DCI_pdu)->ndi;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.harq_ack_resource                     = ((DCIFormatN1_t *)DCI_pdu)->HARQackRes;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.npdcch_order_indication               = ((DCIFormatN1_t *)DCI_pdu)->orderIndicator;
						dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.dci_subframe_repetition_number        = ((DCIFormatN1_t *)DCI_pdu)->DCIRep;
					}else if(schedule_result_list_DL->direction == UL)
					{
						LOG_D(MAC,"NB-IoT fill DL_DCI\n");
						//SCHED_info->HI_DCI0_req = (nfapi_hi_dci0_request_t*)malloc(sizeof(nfapi_hi_dci0_request_t));
						//SCHED_info->HI_DCI0_req->hi_dci0_request_body.number_of_dci = 0;
						//SCHED_info->HI_DCI0_req->hi_dci0_request_body.hi_dci0_pdu_list = (nfapi_hi_dci0_request_pdu_t*)malloc(sizeof(nfapi_hi_dci0_request_pdu_t));
						hi_dci0_pdu = SCHED_info->HI_DCI0_req->hi_dci0_request_body.hi_dci0_pdu_list;
						DCI_pdu = schedule_result_list_DL-> DCI_pdu;
						SCHED_info-> HI_DCI0_req->hi_dci0_request_body.number_of_dci =1;
						hi_dci0_pdu->pdu_type                                                            = NFAPI_HI_DCI0_NPDCCH_DCI_PDU_TYPE;
						hi_dci0_pdu->pdu_size                                                            = 2 + sizeof(nfapi_hi_dci0_npdcch_dci_pdu_rel13_t);
						hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.ncce_index                      = 0;
						hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.aggregation_level               = 1;
						hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.rnti                            = schedule_result_list_DL->rnti;
						hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.subcarrier_indication           = ((DCIFormatN0_t *)DCI_pdu)->scind;
						hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.resource_assignment             = ((DCIFormatN0_t *)DCI_pdu)->ResAssign;
						hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.scheduling_delay                = ((DCIFormatN0_t *)DCI_pdu)->Scheddly;
						hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.mcs                             = ((DCIFormatN0_t *)DCI_pdu)->mcs;
						hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.redudancy_version               = ((DCIFormatN0_t *)DCI_pdu)->rv;
						hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.repetition_number               = ((DCIFormatN0_t *)DCI_pdu)->RepNum;
						hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.new_data_indicator              = ((DCIFormatN0_t *)DCI_pdu)->ndi;
						hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.dci_subframe_repetition_number  = ((DCIFormatN0_t *)DCI_pdu)->DCIRep;
					}
					break;
				case NPDSCH:
						LOG_D(MAC,"NB-IoT fill DL Data\n");
												//LOG_I(MAC,"NB-IoT fill DL Data\n");
						//SCHED_info->DL_req = (nfapi_dl_config_request_t*) malloc (sizeof(nfapi_dl_config_request_t));
						//SCHED_info->DL_req->dl_config_request_body.number_pdu = 0;
						//SCHED_info->DL_req->dl_config_request_body.dl_config_pdu_list = (nfapi_dl_config_request_pdu_t*)malloc(sizeof(nfapi_dl_config_request_pdu_t));
		
						dl_config_pdu = SCHED_info->DL_req->dl_config_request_body.dl_config_pdu_list;
					    DCI_pdu = schedule_result_list_DL-> DCI_pdu;
                        //if(schedule_result_list_DL->rnti==SI_RNTI)
                        //    LOG_D(MAC,"frame:%d, subframe:%d NB-IoT fill SIB23 ResAssign: %d\n",frame,subframe,((DCIFormatN1_t *)DCI_pdu)->ResAssign);
                        SCHED_info->DL_req->dl_config_request_body.number_pdu = 1;
						dl_config_pdu->pdu_type                                           = NFAPI_DL_CONFIG_NDLSCH_PDU_TYPE;
						dl_config_pdu->pdu_size                                           = 2+sizeof(nfapi_dl_config_ndlsch_pdu_rel13_t);
						dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.length                 = schedule_result_list_DL->sdu_length;
						dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.pdu_index			  = 1;
						dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.rnti_type              = 1;
						dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.rnti                   = schedule_result_list_DL->rnti; // C-RNTI
						dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.resource_assignment    = ((DCIFormatN1_t *)DCI_pdu)->ResAssign;
						dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.repetition_number      = ((DCIFormatN1_t *)DCI_pdu)->RepNum;
						dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.modulation             = 2;
						dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.number_of_subframes_for_resource_assignment = get_num_sf(dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.resource_assignment);
						SCHED_info->TX_req->tx_request_body.tx_pdu_list[dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.pdu_index].segments[0].segment_data = schedule_result_list_DL->DLSCH_pdu;
						if(schedule_result_list_DL->rnti==SI_RNTI)
							dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.number_of_subframes_for_resource_assignment =((DCIFormatN1_t *)DCI_pdu)->ResAssign;

						//LOG_D(MAC,"A NB-IoT fill DL Data pdu : %d \n",schedule_result_list_DL->DLSCH_pdu);
					break;
				default:
					break;
			}

			if(schedule_result_list_DL->DCI_release == 1)
				free(schedule_result_list_DL->DCI_pdu);
			
            tmp = schedule_result_list_DL;
			schedule_result_list_DL = schedule_result_list_DL->next;
			

			//LOG_D(MAC,"subframe check scheduling result next %d\n",schedule_result_list_DL->output_subframe);
		}
	}
	
	//	process uplink data transmission(s), might be multiple UL_REQ in one subframe (e.g. 1ms)
	if(schedule_result_list_UL==NULL)
	{
		UL_empty = 1;
		LOG_D(MAC,"no remaining node of UL scheduling result\n");
		//LOG_I(MAC,"no remaining node of UL scheduling result\n");
	}else
	{
		//	here shouldn't be run into
		if(schedule_result_list_UL->output_subframe < current_time)
		{
			while(schedule_result_list_UL->output_subframe < current_time)
			{
				//LOG_D(MAC,"This error if there is UL scheduling result node before the current time\n");
				tmp = schedule_result_list_UL;
				schedule_result_list_UL = schedule_result_list_UL->next;
				free(tmp);
				return -1;
			}
			//LOG_D(MAC,"return\n");
			//return -1;
		}
		else if(schedule_result_list_UL->output_subframe == current_time)
		{
			//SCHED_info->UL_req = (nfapi_ul_config_request_t *)malloc(sizeof(nfapi_ul_config_request_t));
			//SCHED_info->UL_req->ul_config_request_body.number_of_pdus = 0;
			SCHED_info->UL_req->ul_config_request_body.ul_config_pdu_list = (nfapi_ul_config_request_pdu_t *)malloc(5 * sizeof(nfapi_ul_config_request_pdu_t));
			ul_config_pdu = SCHED_info->UL_req->ul_config_request_body.ul_config_pdu_list;
		}	
		while(schedule_result_list_UL->output_subframe == current_time)
		{
			if(schedule_result_list_UL->channel == NPUSCH)
			{
				//LOG_D(MAC,"first UL \n");
				LOG_D(MAC,"NB-IoT fill ul_config_pdu\n");
				//SCHED_info->UL_req = (nfapi_ul_config_request_t *)malloc(sizeof(nfapi_ul_config_request_t));
				//SCHED_info->UL_req->ul_config_request_body.number_of_pdus = 0;
				//SCHED_info->UL_req->ul_config_request_body.ul_config_pdu_list = (nfapi_ul_config_request_pdu_t *)malloc(5 * sizeof(nfapi_ul_config_request_pdu_t));
			
				SCHED_info->UL_req->ul_config_request_body.number_of_pdus ++;
				//SCHED_info->UL_req.sfn_sf = ;
				(ul_config_pdu + i) ->pdu_type                                            = NFAPI_UL_CONFIG_NULSCH_PDU_TYPE;
				(ul_config_pdu + i) ->pdu_size                                            = 2 + sizeof(nfapi_ul_config_nulsch_pdu_rel13_t);

				if(schedule_result_list_UL->npusch_format == 0){
					DCI_pdu = schedule_result_list_UL->DCI_pdu;
					// bug here
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.nulsch_format           = 0;
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.size                    = UL_TBS_Table[((DCIFormatN0_t *)DCI_pdu)->mcs][((DCIFormatN0_t *)DCI_pdu)->ResAssign];
					//LOG_D(MAC,"test\n");
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.rnti                    = schedule_result_list_UL->rnti;
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.subcarrier_indication   = ((DCIFormatN0_t *)DCI_pdu)->scind;
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.resource_assignment     = ((DCIFormatN0_t *)DCI_pdu)->ResAssign;
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.mcs                     = ((DCIFormatN0_t *)DCI_pdu)->mcs;
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.redudancy_version       = ((DCIFormatN0_t *)DCI_pdu)->rv;
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.repetition_number       = ((DCIFormatN0_t *)DCI_pdu)->RepNum;
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.new_data_indication     = ((DCIFormatN0_t *)DCI_pdu)->ndi;

				}else if(schedule_result_list_UL->npusch_format == 1){
					DCI_pdu = schedule_result_list_UL->DCI_pdu;
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.nulsch_format       = 1;
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.rnti                = schedule_result_list_UL->rnti;
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.ue_information.ue_information_rel13.total_number_of_repetitions = schedule_result_list_UL->R_harq;
					//for ACK /NACK
					(ul_config_pdu + i) ->nulsch_pdu.nulsch_pdu_rel13.nb_harq_information.nb_harq_information_rel13_fdd.harq_ack_resource = ((DCIFormatN1_t *)DCI_pdu)->HARQackRes;
				}
				
                if(schedule_result_list_UL->DCI_release == 1){
                    free(schedule_result_list_UL->DCI_pdu);
                }
                
                tmp = schedule_result_list_UL;
				schedule_result_list_UL = schedule_result_list_UL->next;
				
				free(tmp);

				i++;

				if(schedule_result_list_UL == NULL)
					break;
			}else{
				//	there should only be NPUSCH, no exception
				LOG_D(MAC,"error\n");
			}
		}
	}

	LOG_D(MAC,"[hypersfn:%2d][frame:%2d][subframe:%2d]filling the schedule response successfully\n",hypersfn,frame,subframe);

	return 0;
}
