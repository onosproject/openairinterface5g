#include "../SCHED/IF_Module_L1_primitives_nb_iot.h"
#include "../SCHED/defs.h"
#include "../SCHED/defs_nb_iot.h"
#include "common/utils/itti/assertions.h"
#include "PHY/defs.h"
#include "PHY/extern.h"
#include "PHY/vars.h"
#include "PHY/INIT/defs_nb_iot.h"



void handle_nfapi_dlsch_pdu_NB(PHY_VARS_eNB *eNB,
						  eNB_rxtx_proc_t *proc,
		       	   	   	   nfapi_dl_config_request_pdu_t *dl_config_pdu,
						   uint8_t *sdu)
{

	NB_IoT_eNB_NDLSCH_t *ndlsch;
	NB_IoT_DL_eNB_HARQ_t *ndlsch_harq;
	nfapi_dl_config_ndlsch_pdu_rel13_t *rel13 = &dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13;
	int UE_id= -1;

  //Check for SI PDU since in NB-IoT there is no DCI for that
  //SIB (type 0), other DLSCH data (type 1)
  if(rel13->rnti_type == 0)
  {
		ndlsch = eNB->ndlsch_SI;
		ndlsch->npdsch_start_symbol = rel13->start_symbol; //start symbol for the ndlsch transmission
		ndlsch_harq = ndlsch->harq_process;

		ndlsch_harq->pdu = sdu;
		//should be from 1 to 8
		ndlsch_harq->resource_assignment = rel13->number_of_subframes_for_resource_assignment;
		ndlsch_harq->repetition_number = rel13->repetition_number;
		ndlsch_harq->modulation = rel13->modulation;
		ndlsch_harq->status = ACTIVE;
		//SI information in reality have no feedback
//        ndlsch_harq->frame = frame;
//        ndlsch_harq->subframe = subframe;
        ndlsch->nrs_antenna_ports = rel13->nrs_antenna_ports_assumed_by_the_ue;
        ndlsch->scrambling_sequence_intialization = rel13->scrambling_sequence_initialization_cinit;

        //managment of TBS size for SI??? (is written inside the SIB1-NB)

  }
  else
  { //ue specific data or RAR

	  //check if the PDU is for RAR
	  if(eNB->ndlsch_ra != NULL && rel13->rnti == eNB->ndlsch_ra->rnti)
	  {
		  eNB->ndlsch_ra->harq_process->pdu = sdu;
		  eNB->ndlsch_ra->npdsch_start_symbol = rel13->start_symbol;
	  }
	  else
	  { //this for ue data
		  //TODO
		  //program addition DLSCH parameters not from DCI (for the moment we only pass the pdu)
		  //int UE_id = find_dlsch(rel13->rnti,eNB,SEARCH_EXIST);


		  UE_id =  find_ue_NB(rel13->rnti,eNB);
	  	  AssertFatal(UE_id==-1,"no existing ue specific dlsch_context\n");

	  	  ndlsch_harq     = eNB->ndlsch[(uint8_t)UE_id]->harq_process;
	  	  AssertFatal(ndlsch_harq!=NULL,"dlsch_harq for ue specific is null\n");

	  	  ndlsch = eNB->ndlsch[(uint8_t)UE_id];
	  	  ndlsch->npdsch_start_symbol = rel13->start_symbol;
	  	  ndlsch_harq->pdu  = sdu;
	  }

  }



}




// do the schedule response and trigger the TX
void schedule_response(Sched_Rsp_t *Sched_INFO)
{

  //XXX check if correct to take eNB like this
  PHY_VARS_eNB *eNB = PHY_vars_eNB_g[0][Sched_INFO->CC_id];
  eNB_rxtx_proc_t *proc = &eNB->proc.proc_rxtx[0];
  NB_IoT_eNB_NPBCH *npbch;

  int i;

  module_id_t                     Mod_id    = Sched_INFO->module_id;
  uint8_t                         CC_id     = Sched_INFO->CC_id;
  nfapi_dl_config_request_body_t *DL_req    = Sched_INFO->DL_req;
  nfapi_ul_config_request_t 	 *UL_req    = Sched_INFO->UL_req;
  nfapi_hi_dci0_request_body_t *HI_DCI0_req = Sched_INFO->HI_DCI0_req;

  frame_t                         frame     = Sched_INFO->frame;
  sub_frame_t                     subframe  = Sched_INFO->subframe;


  AsserFatal(proc->subframe_tx != subframe, "Current subframe %d != NFAPI subframe %d\n",proc->subframe_tx,subframe);
  AsserFatal(proc->frame_tx != frame, "Current sframe %d != NFAPI frame %d\n", proc->frame_tx,frame );

  uint8_t number_dl_pdu             = DL_req->number_pdu;
  uint8_t number_ul_pdu				= UL_req->ul_config_request_body.number_of_pdus;
  uint8_t number_ul_dci             = HI_DCI0_req->number_of_dci;
  uint8_t number_pdsch_rnti         = DL_req->number_pdsch_rnti; // for the moment not used

  // at most 2 pdus (DCI) in the case of NPDCCH
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

    		//XXX for the moment we don't care about the n-bch pdu content since we need only the sdu if tx.request
    		npbch = eNB->npbch;
    		npbch->pdu = Sched_INFO->sdu[i];

      		break;
    	case NFAPI_DL_CONFIG_NDLSCH_PDU_TYPE:

    		handle_nfapi_dlsch_pdu_NB(eNB, proc,dl_config_pdu,Sched_INFO->sdu[i]);

    		break;
    	default:
    		LOG_E(PHY, "dl_config_pdu type not for NB_IoT\n");
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
    	default:
    		LOG_E(PHY, "dl_config_pdu type not for NB_IoT\n");
    		break;

  	}
  }


  for(i = 0; i< number_ul_pdu; i++)
  {
	  ul_config_pdu = &UL_req->ul_config_request_body.ul_config_pdu_list[i];
	  switch(ul_config_pdu->pdu_type)
	  {
	  case NFAPI_UL_CONFIG_NULSCH_PDU_TYPE:
		  //TODO should distinguish between data and between data (npusch format)
		  /*NB: for reception of Msg3 generally not exist a DCI (because scheduling information are implicitly given by the RAR)
		   * but in case of FAPI specs we should receive an UL_config (containing the NULSCH pdu) for configuring the PHY for Msg3 reception from the MAC
		   * (this UL_config most probably will be created by the MAC when fill the RAR)
		   * (most probably we don't have the DL_config (for the RAR transmission) and the UL_CONFIG (for the Msg3 reception) at the same time (same subrame)
		   * since we are working in HD-FDD mode so for sure the UE will transmit the Msg3 in another subframe so make sense to have a UL_CONFIG in subframe
		   * diferent from the DL_CONFIG one)
		   *
		   */
		  break;
	  case NFAPI_UL_CONFIG_NRACH_PDU_TYPE:
		  //TODO just for update the nprach  configuration (given at the beginning through phy_config_sib2)
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
							  config_INFO->cfg->sch_config.physical_cell_id.value,
							  config_INFO->cfg->subframe_config.dl_cyclic_prefix_type.value,
							  config_INFO->cfg->subframe_config.ul_cyclic_prefix_type.value,
							  config_INFO->cfg->rf_config.tx_antenna_ports.value,
							  config_INFO->dl_CarrierFreq,
							  config_INFO->ul_CarrierFreq,
							  config_INFO->cfg->nb_iot_config.prb_index.value,
							  config_INFO->cfg->nb_iot_config.operating_mode.value,
							  config_INFO->cfg->nb_iot_config.control_region_size.value,
							  config_INFO->cfg->nb_iot_config.assumed_crs_aps.value); //defined only in in-band different PCI
	}

	if(config_INFO->get_COMMON != 0)
	{
		//Common Configuration included in SIB2-NB
		NB_phy_config_sib2_eNB(config_INFO->mod_id,
							   config_INFO->CC_id,
						       &config_INFO->cfg->nb_iot_config, // FIXME to be evaluated is should be passed a pointer
						       &config_INFO->cfg->rf_config,
							   &config_INFO->cfg->uplink_reference_signal_config,
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

