
//#include "openair1/PHY/defs.h"
//#include "openair2/PHY_INTERFACE/IF_Module.h"
//#include "openair1/PHY/extern.h"
#include "LAYER2/MAC/extern.h"
//#include "LAYER2/MAC/proto.h"
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

