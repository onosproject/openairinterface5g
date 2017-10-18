
//#include "openair1/PHY/defs.h"
//#include "openair2/PHY_INTERFACE/IF_Module.h"
//#include "openair1/PHY/extern.h"
#include "LAYER2/MAC/extern.h"
//#include "LAYER2/MAC/proto.h"
#include "openair2/LAYER2/MAC/vars.h"
//#include "common/ran_context.h"
#include "openair2/PHY_INTERFACE/phy_stub_UE.h"






void handle_nfapi_UE_Rx(uint8_t Mod_id, Sched_Rsp_t *Sched_INFO, int eNB_id){
	// copy data from eNB L2 interface to UE L2 interface

	int 					   CC_id	   = Sched_INFO->CC_id;
	nfapi_dl_config_request_t *DL_req      = Sched_INFO->DL_req;
	nfapi_tx_request_t        *Tx_req      = Sched_INFO->TX_req;
	frame_t                   frame        = Sched_INFO->frame;
	sub_frame_t               subframe     = Sched_INFO->subframe;

	uint8_t number_dl_pdu             = DL_req->dl_config_request_body.number_pdu;
	nfapi_dl_config_request_pdu_t *dl_config_pdu;
	nfapi_dl_config_request_pdu_t *dl_config_pdu_tmp;
	int i = 0;

	for (i=0; i<number_dl_pdu; i++)
	{
		dl_config_pdu = &DL_req->dl_config_request_body.dl_config_pdu_list[i];
		switch (dl_config_pdu->pdu_type) {
		case NFAPI_DL_CONFIG_BCH_PDU_TYPE:
			// BCH case
			// Last parameter is 1 if first time synchronization and zero otherwise. Not sure which value to put
			// for our case.
			dl_phy_sync_success(Mod_id,frame,eNB_id, 0);
			break;

		case NFAPI_DL_CONFIG_DCI_DL_PDU_TYPE:
			if (dl_config_pdu->dci_dl_pdu.dci_dl_pdu_rel8.rnti_type == 1) {
				// C-RNTI (Normal DLSCH case)
				dl_config_pdu_tmp = &DL_req->dl_config_request_body.dl_config_pdu_list[i+1];
				if (dl_config_pdu_tmp->pdu_type == NFAPI_DL_CONFIG_DLSCH_PDU_TYPE){
					ue_send_sdu(Mod_id, CC_id, frame, subframe,
							Tx_req->tx_request_body.tx_pdu_list[dl_config_pdu_tmp->dlsch_pdu.dlsch_pdu_rel8.pdu_index].segments[0].segment_data,
							Tx_req->tx_request_body.tx_pdu_list[dl_config_pdu_tmp->dlsch_pdu.dlsch_pdu_rel8.pdu_index].segments[0].segment_length,
							eNB_id);
					i++;
				}
				else {
					LOG_E(MAC,"[UE %d] CCid %d Frame %d, subframe %d : Cannot extract DLSCH PDU from NFAPI\n",Mod_id, CC_id,frame,subframe);
				}
			}
			else if (dl_config_pdu->dci_dl_pdu.dci_dl_pdu_rel8.rnti_type == 2) {
				dl_config_pdu_tmp = &DL_req->dl_config_request_body.dl_config_pdu_list[i+1];
				if(dl_config_pdu_tmp->pdu_type == NFAPI_DL_CONFIG_DLSCH_PDU_TYPE && dl_config_pdu->dci_dl_pdu.dci_dl_pdu_rel8.rnti == 0xFFFF){
					//pdu = Tx_req->tx_request_body.tx_pdu_list[dl_config_pdu->dlsch_pdu.dlsch_pdu_rel8.pdu_index].segments[0].segment_data;
					// Question about eNB_index here. How do we obtain it?
					ue_decode_si(Mod_id, CC_id, frame, eNB_id,
							Tx_req->tx_request_body.tx_pdu_list[dl_config_pdu_tmp->dlsch_pdu.dlsch_pdu_rel8.pdu_index].segments[0].segment_data,
							Tx_req->tx_request_body.tx_pdu_list[dl_config_pdu_tmp->dlsch_pdu.dlsch_pdu_rel8.pdu_index].segments[0].segment_length);
					i++;
				}
				else if(dl_config_pdu_tmp->pdu_type == NFAPI_DL_CONFIG_DLSCH_PDU_TYPE && dl_config_pdu->dci_dl_pdu.dci_dl_pdu_rel8.rnti == 0xFFFE){
					// P_RNTI case
					//pdu = Tx_req->tx_request_body.tx_pdu_list[dl_config_pdu->dlsch_pdu.dlsch_pdu_rel8.pdu_index].segments[0].segment_data;
					// Question about eNB_index here. How do we obtain it?
					ue_decode_p(Mod_id, CC_id, frame, eNB_id,
							Tx_req->tx_request_body.tx_pdu_list[dl_config_pdu_tmp->dlsch_pdu.dlsch_pdu_rel8.pdu_index].segments[0].segment_data,
							Tx_req->tx_request_body.tx_pdu_list[dl_config_pdu_tmp->dlsch_pdu.dlsch_pdu_rel8.pdu_index].segments[0].segment_length);
					i++;
				}
				else if(dl_config_pdu_tmp->pdu_type == NFAPI_DL_CONFIG_DLSCH_PDU_TYPE) {
					// RA-RNTI case
					// ue_process_rar should be called but the problem is that this function currently uses PHY_VARS_UE
					// elements.

					// C-RNTI parameter not actually used. Provided only to comply with existing function definition.
					// Not sure about parameters to fill the preamble index. Originally it comes from PHY.
					const rnti_t c_rnti;
					ue_process_rar(Mod_id, CC_id, frame,
							dl_config_pdu_tmp->dlsch_pdu.dlsch_pdu_rel8.rnti, //RA-RNTI
							Tx_req->tx_request_body.tx_pdu_list[dl_config_pdu_tmp->dlsch_pdu.dlsch_pdu_rel8.pdu_index].segments[0].segment_data,
							c_rnti,
							UE_mac_inst[Mod_id].RA_prach_resources.ra_PreambleIndex,
							Tx_req->tx_request_body.tx_pdu_list[dl_config_pdu_tmp->dlsch_pdu.dlsch_pdu_rel8.pdu_index].segments[0].segment_data);
				}
				else {
					LOG_E(MAC,"[UE %d] CCid %d Frame %d, subframe %d : Cannot extract DLSCH PDU from NFAPI\n",Mod_id, CC_id,frame,subframe);
				}

			}
			break;
		}

	}


}


void fill_rx_indication_UE_MAC(int Mod_id,int frame,int subframe, UL_IND_t *UL_INFO, uint8_t *ulsch_buffer, uint16_t buflen)
{
	  nfapi_rx_indication_pdu_t *pdu;

	  int timing_advance_update;
	  //int sync_pos;

	  /*uint32_t harq_pid = subframe2harq_pid(&eNB->frame_parms,
						frame,subframe);*/


	  pthread_mutex_lock(&UE_mac_inst[Mod_id].UL_INFO_mutex);

	  //eNB->UL_INFO.rx_ind.sfn_sf                    = frame<<4| subframe;
	  //eNB->UL_INFO.rx_ind.rx_indication_body.tl.tag = NFAPI_RX_INDICATION_BODY_TAG;

	  pdu                                    = &UL_INFO->rx_ind.rx_pdu_list[UL_INFO->rx_ind.number_of_pdus];

	  //  pdu->rx_ue_information.handle          = eNB->ulsch[UE_id]->handle;
	  pdu->rx_ue_information.tl.tag          = NFAPI_RX_UE_INFORMATION_TAG;
	  pdu->rx_ue_information.rnti            = UE_mac_inst[Mod_id].crnti;
	  pdu->rx_indication_rel8.tl.tag         = NFAPI_RX_INDICATION_REL8_TAG;
	  //pdu->rx_indication_rel8.length         = eNB->ulsch[UE_id]->harq_processes[harq_pid]->TBS>>3;
	  pdu->rx_indication_rel8.length         = buflen;
	  pdu->rx_indication_rel8.offset         = 1;   // DJP - I dont understand - but broken unless 1 ????  0;  // filled in at the end of the UL_INFO formation
	  pdu->data                              = ulsch_buffer;
	  // estimate timing advance for MAC
	  //sync_pos                               = lte_est_timing_advance_pusch(eNB,UE_id);
	  timing_advance_update                  = 0;  //Panos: Don't know what to put here
	  pdu->rx_indication_rel8.timing_advance = timing_advance_update;

	  //  if (timing_advance_update > 10) { dump_ulsch(eNB,frame,subframe,UE_id); exit(-1);}
	  //  if (timing_advance_update < -10) { dump_ulsch(eNB,frame,subframe,UE_id); exit(-1);}
	  /*switch (eNB->frame_parms.N_RB_DL) {
	  case 6:
	    pdu->rx_indication_rel8.timing_advance = timing_advance_update;
	    break;
	  case 15:
	    pdu->rx_indication_rel8.timing_advance = timing_advance_update/2;
	    break;
	  case 25:
	    pdu->rx_indication_rel8.timing_advance = timing_advance_update/4;
	    break;
	  case 50:
	    pdu->rx_indication_rel8.timing_advance = timing_advance_update/8;
	    break;
	  case 75:
	    pdu->rx_indication_rel8.timing_advance = timing_advance_update/12;
	    break;
	  case 100:
	    pdu->rx_indication_rel8.timing_advance = timing_advance_update/16;
	    break;
	  }
	  // put timing advance command in 0..63 range
	  timing_advance_update += 31;
	  if (timing_advance_update < 0)  timing_advance_update = 0;
	  if (timing_advance_update > 63) timing_advance_update = 63;
	  pdu->rx_indication_rel8.timing_advance = timing_advance_update;*/

	  // estimate UL_CQI for MAC (from antenna port 0 only)

	  // Panos dependency from eNB not sure how to substitute this. Should we hardcode it?
	  //int SNRtimes10 = dB_fixed_times10(eNB->pusch_vars[UE_id]->ulsch_power[0]) - 200;//(10*eNB->measurements.n0_power_dB[0]);
	  int SNRtimes10 = 640;

	  if      (SNRtimes10 < -640) pdu->rx_indication_rel8.ul_cqi=0;
	  else if (SNRtimes10 >  635) pdu->rx_indication_rel8.ul_cqi=255;
	  else                        pdu->rx_indication_rel8.ul_cqi=(640+SNRtimes10)/5;


	  /*LOG_D(PHY,"[PUSCH %d] Filling RX_indication with SNR %d (%d), timing_advance %d (update %d)\n",
		harq_pid,SNRtimes10,pdu->rx_indication_rel8.ul_cqi,pdu->rx_indication_rel8.timing_advance,
		timing_advance_update);*/

	  UL_INFO->rx_ind.number_of_pdus++;
	  pthread_mutex_unlock(&UE_mac_inst[Mod_id].UL_INFO_mutex);


}

void fill_sr_indication_UE_MAC(int Mod_id,int frame,int subframe, UL_IND_t *UL_INFO) {

  pthread_mutex_lock(&UE_mac_inst[Mod_id].UL_INFO_mutex);
  nfapi_sr_indication_pdu_t *pdu =   &UL_INFO->sr_ind.sr_pdu_list[UL_INFO->rx_ind.number_of_pdus];

  pdu->instance_length                                = 0; // don't know what to do with this
  //  pdu->rx_ue_information.handle                       = handle;
  pdu->rx_ue_information.tl.tag                       = NFAPI_RX_UE_INFORMATION_TAG;
  pdu->rx_ue_information.rnti                         = UE_mac_inst[Mod_id].crnti;; //Panos: Is this the right RNTI?


  // Panos dependency from PHY not sure how to substitute this. Should we hardcode it?
  //int SNRtimes10 = dB_fixed_times10(stat) - 200;//(10*eNB->measurements.n0_power_dB[0]);
  int SNRtimes10 = 640;


  if      (SNRtimes10 < -640) pdu->ul_cqi_information.ul_cqi=0;
  else if (SNRtimes10 >  635) pdu->ul_cqi_information.ul_cqi=255;
  else                        pdu->ul_cqi_information.ul_cqi=(640+SNRtimes10)/5;
  pdu->ul_cqi_information.channel = 0;

  UL_INFO->rx_ind.number_of_pdus++;
  pthread_mutex_unlock(&UE_mac_inst[Mod_id].UL_INFO_mutex);
}


void fill_crc_indication_UE_MAC(int Mod_id,int frame,int subframe, UL_IND_t *UL_INFO, uint8_t crc_flag) {

  pthread_mutex_lock(&UE_mac_inst[Mod_id].UL_INFO_mutex);
  nfapi_crc_indication_pdu_t *pdu =   &UL_INFO->crc_ind.crc_pdu_list[UL_INFO->crc_ind.number_of_crcs];

  //eNB->UL_INFO.crc_ind.sfn_sf                         = frame<<4 | subframe;
  //eNB->UL_INFO.crc_ind.crc_indication_body.tl.tag     = NFAPI_CRC_INDICATION_BODY_TAG;

  pdu->instance_length                                = 0; // don't know what to do with this
  //  pdu->rx_ue_information.handle                       = handle;
  pdu->rx_ue_information.tl.tag                       = NFAPI_RX_UE_INFORMATION_TAG;
  pdu->rx_ue_information.rnti                         = UE_mac_inst[Mod_id].crnti;
  pdu->crc_indication_rel8.tl.tag                     = NFAPI_CRC_INDICATION_REL8_TAG;
  pdu->crc_indication_rel8.crc_flag                   = crc_flag;

  UL_INFO->crc_ind.number_of_crcs++;

  LOG_D(PHY, "%s() rnti:%04x pdus:%d\n", __FUNCTION__, pdu->rx_ue_information.rnti, UL_INFO->crc_ind.number_of_crcs);

  pthread_mutex_unlock(&UE_mac_inst[Mod_id].UL_INFO_mutex);
}








