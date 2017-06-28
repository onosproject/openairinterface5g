#include "../SCHED/IF_Module_L1_primitives_nb_iot.h"
#include "PHY/defs.h"
#include "PHY/extern.h"
#include "PHY/vars.h"



// do the schedule response and trigger the TX
void schedule_response(Sched_Rsp_t *Sched_INFO)
{

  PHY_VARS_eNB *eNB = PHY_vars_eNB_g[0][Sched_INFO->CC_id];
  eNB_rxtx_proc_t *proc = &eNB->proc.proc_rxtx[0];

  int UE_id = 0;
  int i;

  module_id_t                     Mod_id    = Sched_INFO->module_id;
  uint8_t                         CC_id     = Sched_INFO->CC_id;
  nfapi_dl_config_request_body_t *DL_req    = &Sched_INFO->DL_req;
  nfapi_ul_config_request_t 	 *UL_req    = &Sched_INFO->UL_req;
  nfapi_hi_dci0_request_body_t *HI_DCI0_req = &Sched_INFO->HI_DCI0_req;

  frame_t                         frame     = Sched_INFO->frame;
  sub_frame_t                     subframe  = Sched_INFO->subframe;

  uint8_t number_dl_pdu             = DL_req->number_pdu;
  uint8_t number_ul_pdu				= UL_req->ul_config_request_body.number_of_pdus;
  uint8_t number_ul_dci             = HI_DCI0_req->number_of_dci;
  uint8_t number_pdsch_rnti         = DL_req->number_pdsch_rnti; // for the moment not used

  // at most 2 pdus in the case of NPDCCH
  nfapi_dl_config_request_pdu_t *dl_config_pdu;
  nfapi_ul_config_request_pdu_t *ul_config_pdu;
  nfapi_hi_dci0_request_pdu_t *hi_dci0_pdu;

  for (i=0;i<number_dl_pdu;i++) 
  {
    dl_config_pdu = &DL_req->dl_config_pdu_list[i];
    switch (dl_config_pdu->pdu_type) 
    {
    	case NFAPI_DL_CONFIG_NPDCCH_PDU_TYPE:
      		NB_generate_eNB_dlsch_params(eNB,proc,dl_config_pdu);
      		break;
    	case NFAPI_DL_CONFIG_NBCH_PDU_TYPE:
      		// Do nothing for the moment
      		break;
    	case NFAPI_DL_CONFIG_NDLSCH_PDU_TYPE:
    	// Do the distinction between SIB (type 0) and other downlink data (type 1)
      	//handle_nfapi_dlsch_pdu(eNB,proc,dl_config_pdu,Sched_INFO->sdu[i]);
      	break;
   }
  }
  
  for (i=0;i<number_ul_dci;i++) 
  {
    hi_dci0_pdu = &HI_DCI0_req->hi_dci0_pdu_list[i];
    switch (hi_dci0_pdu->pdu_type) 
    {
    	case NFAPI_HI_DCI0_NPDCCH_DCI_PDU_TYPE:
      		NB_generate_eNB_ulsch_params(eNB,proc,hi_dci0_pdu);
      	break;

  	}
  }
  NB_phy_procedures_eNB_TX(eNB,proc,NULL);

}

void PHY_config_req(PHY_Config_t* config_INFO){


	if(config_INFO->get_MIB != 0){
		//MIB-NB configuration
		NB_phy_config_mib_eNB(config_INFO->mod_id,
							  config_INFO->CC_id,
							  config_INFO->frequency_band_indicator,
							  config_INFO->sch_config.physical_cell_id.value,
							  config_INFO->subframe_config.dl_cyclic_prefix_type.value,
							  config_INFO->subframe_config.ul_cyclic_prefix_type.value,
							  config_INFO->rf_config.tx_antenna_ports.value,
							  config_INFO->dl_CarrierFreq,
							  config_INFO->ul_CarrierFreq);
	}

	if(config_INFO->get_COMMON != 0)
	{
		//Common Configuration included in SIB2-NB
		NB_phy_config_sib2_eNB(config_INFO->mod_id,
							   config_INFO->CC_id,
						       &config_INFO->nb_iot_config, // FIXME to be evaluated is should be passed a pointer
						       &config_INFO->rf_config,
							   &config_INFO->uplink_reference_signal_config,
							   &config_INFO->extra_phy_parms
							   );
	}

	if(config_INFO->get_DEDICATED!= 0)
	{
	//Dedicated Configuration
		if(config_INFO->phy_config_dedicated != NULL){

			NB_phy_config_dedicated_eNB(config_INFO->mod_id,
										config_INFO->CC_id,
										config_INFO->rnti,
										config_INFO->phy_config_dedicated //not defined by fapi specs
										);
		}

	}
}

