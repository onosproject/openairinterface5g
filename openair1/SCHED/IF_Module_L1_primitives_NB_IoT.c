

#include "openair2/PHY_INTERFACE/IF_Module_NB_IoT.h"
#include "../SCHED/IF_Module_L1_primitives_NB_IoT.h"
//#include "../SCHED/defs.h"
#include "../SCHED/defs_NB_IoT.h"
#include "assertions.h"
//#include "PHY/defs.h"
#include "PHY/defs_NB_IoT.h"
//#include "PHY/extern.h"
#include "PHY/extern_NB_IoT.h"
//#include "PHY/vars.h"

#include "PHY/INIT/defs_NB_IoT.h"


void handle_nfapi_dlsch_pdu_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,
						  		   eNB_rxtx_proc_NB_IoT_t *proc,
		       	   	   	           nfapi_dl_config_request_pdu_t *dl_config_pdu,
						   		   uint8_t *sdu)
{

	NB_IoT_eNB_NDLSCH_t *ndlsch;
	NB_IoT_DL_eNB_HARQ_t *ndlsch_harq;
	nfapi_dl_config_ndlsch_pdu_rel13_t *rel13 = &dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13;
	int UE_id= -1;

  //Check for SI PDU since in NB-IoT there is no DCI for that
  //SIB1 (type 0), other DLSCH data (type 1) (include the SI messages) based on our ASSUMPTIONs

	//is SIB1-NB
  if(rel13->rnti_type == 0 && rel13->rnti == 65535)
  {

	  /*
	   * the configuration of the NDLSCH PDU for the SIB1-NB shoudl be the following:
	   * -RNTI type = 0; (BCCH)
	   * -RNTI = OxFFFF (65535)
	   * -Repetition number = 0-15 and should be mapped to 4,8,16 as reported in Table 16.4.1.3-3 TS 36.213 (is the schedulingInoSIB1 of the MIB)
	   * -Number of subframe for resource assignment = may is not neded to know since the scheduling is fixed
	   *  (Spec TS 36.331 "SIB1-NB transmission occur in subframe #4 of every other frame in 16 continuous frame"
	   *  meaning that from the starting point we should transmit the SIB1-NB in 8 subframes  among the 16 available (every other))
	   *
	   * From spec. TS 36.321 v14.2.o pag 31 --> there is an HARQ process for all the broadcast (so we consider it also for SIB1-NB)
	   *
	   */

	  	ndlsch= eNB->ndlsch_SIB1;
	  	ndlsch->ndlsch_type = SIB1;

		ndlsch->npdsch_start_symbol = rel13->start_symbol; //start symbol for the ndlsch transmission
		ndlsch_harq = ndlsch->harq_process;
		ndlsch_harq->pdu = sdu;
		//should be from 1 to 8
		ndlsch_harq->resource_assignment = rel13->number_of_subframes_for_resource_assignment;//maybe we don't care about it since a fixed schedule
		ndlsch_harq->repetition_number = rel13->repetition_number; //is the schedulingInfoSIB1 (value 1-15) of MIB that is mapped into value 4-8-16 (see NDLSCH fapi specs Table 4-47)
		ndlsch_harq->modulation = rel13->modulation;
		ndlsch_harq->status = ACTIVE_NB_IoT;

		//SI information in reality have no feedback (so there is no retransmission from the HARQ view point since no ack and nack)
//        ndlsch_harq->frame = frame;
//        ndlsch_harq->subframe = subframe;

        ndlsch->nrs_antenna_ports = rel13->nrs_antenna_ports_assumed_by_the_ue;
        ndlsch->scrambling_sequence_intialization = rel13->scrambling_sequence_initialization_cinit;




  }
  //is SI message (this is an NDLSCH that will be transmitted very frequently)
  else if(rel13->rnti_type == 1 && rel13->rnti == 65535)
  {
	  /*
	   *
	   * the configuration of the NDLSCH PDU for the SIB1-NB should be the following:
	   * RNTI type = 1;
	   * RNTI = OxFFFF (65535)
	   * Repetition number = 0 and should be mapped to 1 through Table 16.4.1.3-2 TS 36.213
	   * Number of subframe for resource assignment = will be evaluated by the MAC based on the value of the "si-TB" field inside the SIB1-NB (value 2 or 8)
	   *
	   * From spec. TS 36.321 v14.2.o pag 31 --> there is an HARQ process for all the broadcast
	   *
	   * XXX for the moment we are not able to prevent the problem of Error: first transmission but sdu = NULL.
	   * anyway, the PHY layer if have finished the transmission it will not transmit anything and will generate the error
	   *
	   */

	  	ndlsch= eNB->ndlsch_SI;
	  	ndlsch_harq = ndlsch->harq_process;

		//new SI starting transmission (should enter here only the first time for a new transmission)
		if(sdu != NULL)
		{

		  	ndlsch->ndlsch_type = SI_Message;
			ndlsch->npdsch_start_symbol = rel13->start_symbol; //start OFDM symbol for the ndlsch transmission
			ndlsch_harq->pdu = sdu;
			ndlsch_harq->resource_assignment = rel13->number_of_subframes_for_resource_assignment;//value 2 or 8
			ndlsch_harq->repetition_number = rel13->repetition_number;//should be always fix to 0 to be mapped in 1
			ndlsch_harq->modulation = rel13->modulation;


			//SI information in reality have no feedback (so there is no retransmission from the HARQ view point since no sck and nack)
	//        ndlsch_harq->frame = frame;
	//        ndlsch_harq->subframe = subframe;

			ndlsch->nrs_antenna_ports = rel13->nrs_antenna_ports_assumed_by_the_ue;
			ndlsch->scrambling_sequence_intialization = rel13->scrambling_sequence_initialization_cinit;
		}
		else
		{
			//continue the remaining transmission of the previous SI at PHY if any (otherwise nothing)
			//there is no need of repeating the configuration on the ndlsch
			ndlsch_harq->pdu = NULL;
		}

		//Independently if we have the PDU or not (first transmission or repetition) the process is activated for triggering the ndlsch_procedure
	  	ndlsch_harq->status = ACTIVE_NB_IoT;


  }
  //ue specific data or RAR (we already have received the DCI for this)
  else if(rel13->rnti != 65535 && rel13->rnti_type == 1)
  {

	  //check if the PDU is for RAR
	  if(eNB->ndlsch_ra != NULL && rel13->rnti == eNB->ndlsch_ra->rnti) //rnti for the RAR should have been set priviously by the DCI
	  {
		  eNB->ndlsch_ra->harq_process->pdu = sdu;
		  eNB->ndlsch_ra->npdsch_start_symbol = rel13->start_symbol;
		  eNB->ndlsch_ra->active = 1;
	  }
	  else
	  { //this for ue data
		  //TODO
		  //program addition DLSCH parameters not from DCI (for the moment we only pass the pdu)
		  //int UE_id = find_dlsch(rel13->rnti,eNB,SEARCH_EXIST);


		  UE_id =  find_ue_NB_IoT(rel13->rnti,eNB);
	  	  AssertFatal(UE_id==-1,"no existing ue specific dlsch_context\n");

	  	  ndlsch = eNB->ndlsch[(uint8_t)UE_id];
	  	  ndlsch_harq     = eNB->ndlsch[(uint8_t)UE_id]->harq_process;
	  	  AssertFatal(ndlsch_harq!=NULL,"dlsch_harq for ue specific is null\n");

	  	  ndlsch->npdsch_start_symbol = rel13->start_symbol;
	  	  ndlsch_harq->pdu  = sdu;
	  	  ndlsch->active = 1;

	  }

  }
  //I don't know which kind of data is
  else
  {
	  LOG_E(PHY, "handle_nfapi_dlsch_pdu_NB_IoT: Unknown type of data (rnti type %d, rnti %d)\n", rel13->rnti_type, rel13->rnti);
  }

}



/////////////////////////////////////////////////////////////////////////////////////////////////
//Memo for initialization TODO: target/SIMU/USER/init_lte.c/init_lte_eNB --> new_eNB_dlsch(..) //
//this is where the allocation of PHy_vars_eNB_NB_IoT and all the ndlsch structures happen            //
/////////////////////////////////////////////////////////////////////////////////////////////////


// do the schedule response and trigger the TX
void schedule_response_NB_IoT(Sched_Rsp_NB_IoT_t *Sched_INFO)
{

  //XXX check if correct to take eNB like this
  PHY_VARS_eNB_NB_IoT 		*eNB     = PHY_vars_eNB_NB_IoT_g[0][Sched_INFO->CC_id];
  eNB_rxtx_proc_NB_IoT_t 	*proc 	 = &eNB->proc.proc_rxtx[0];
  NB_IoT_eNB_NPBCH_t 		*npbch;
  ///
  int 						i;
  //module_id_t                     Mod_id    = Sched_INFO->module_id;
  //uint8_t                         CC_id     = Sched_INFO->CC_id;
  nfapi_dl_config_request_t 	*DL_req    		= Sched_INFO->DL_req;
  nfapi_ul_config_request_t 	*UL_req    		= Sched_INFO->UL_req;
  nfapi_hi_dci0_request_t 		*HI_DCI0_req 	= Sched_INFO->HI_DCI0_req;
  nfapi_tx_request_t        	*TX_req    		= Sched_INFO->TX_req; 

  //uint32_t                     hypersfn  		= Sched_INFO->hypersfn;
  //frame_t                      frame     		= Sched_INFO->frame;       // unused for instance
  sub_frame_t                    subframe  		= Sched_INFO->subframe;	 

  // implicite declaration of AssertFatal
  //AsserFatal(proc->subframe_tx != subframe, "Current subframe %d != NFAPI subframe %d\n",proc->subframe_tx,subframe);
  //AsserFatal(proc->frame_tx != frame, "Current sframe %d != NFAPI frame %d\n", proc->frame_tx,frame );

  uint8_t number_dl_pdu             = DL_req->dl_config_request_body.number_pdu;
  uint8_t number_ul_pdu				= UL_req->ul_config_request_body.number_of_pdus;
  uint8_t number_ul_dci             = HI_DCI0_req->hi_dci0_request_body.number_of_dci;
  //uint8_t number_pdsch_rnti         = DL_req->number_pdsch_rnti; // for the moment not used
  // at most 2 pdus (DCI) in the case of NPDCCH
  nfapi_dl_config_request_pdu_t 	*dl_config_pdu;
  nfapi_ul_config_request_pdu_t 	*ul_config_pdu;
  nfapi_hi_dci0_request_pdu_t 		*hi_dci0_pdu;

  //clear previous possible allocation (maybe someone else should be added)
  for(int i = 0; i < NUMBER_OF_UE_MAX_NB_IoT; i++)
  {
	  if(eNB->ndlsch[i])
	  {
		  eNB->ndlsch[i]->harq_process->round=0; // may not needed
		  /*clear previous allocation information for all UEs*/
		  eNB->ndlsch[i]->subframe_tx[subframe] = 0;
	  }

	  /*clear the DCI allocation maps for new subframe*/
	  if(eNB->nulsch[i])
	  {
		  eNB->nulsch[i]->harq_process->dci_alloc = 0; //flag for indicating that a DCI has been allocated for UL
		  eNB->nulsch[i]->harq_process->rar_alloc = 0; //Flag indicating that this ULSCH has been allocated by a RAR (for Msg3)
		  //no phich for NB-IoT so no DMRS should be utilized
	  }

  }


  for (i=0;i<number_dl_pdu;i++) //in principle this should be at most 2 (in case of DCI)
  {
    dl_config_pdu = &DL_req->dl_config_request_body.dl_config_pdu_list[i];
    switch (dl_config_pdu->pdu_type) 
    {
    	case NFAPI_DL_CONFIG_NPDCCH_PDU_TYPE:
    		//Remember: there is no DCI for SI information
    		//TODO: separate the ndlsch structure configuration from the DCI (here we will encode only the DCI)
      		generate_eNB_dlsch_params_NB_IoT(eNB,proc,dl_config_pdu);

      		break;
    	case NFAPI_DL_CONFIG_NBCH_PDU_TYPE:

    		// for the moment we don't care about the n-bch pdu content since we need only the sdu if tx.request
    		npbch = eNB->npbch; //in the main of the lte-softmodem they should allocate this memory of PHY_vars
    		npbch->h_sfn_lsb = dl_config_pdu->nbch_pdu.nbch_pdu_rel13.hyper_sfn_2_lsbs;

    		if(TX_req->tx_request_body.tx_pdu_list[dl_config_pdu->nbch_pdu.nbch_pdu_rel13.pdu_index].segments[0].segment_data != NULL)
    			npbch->pdu = TX_req->tx_request_body.tx_pdu_list[dl_config_pdu->nbch_pdu.nbch_pdu_rel13.pdu_index].segments[0].segment_data;
    		else
    			LOG_E(PHY, "Received a schedule_response with N-BCH but no SDU!!\n");

      		break;
    	case NFAPI_DL_CONFIG_NDLSCH_PDU_TYPE:
    		//we can have three types of NDLSCH based on our assumptions: SIB1, SI, Data, RAR
    		//remember that SI messages have no DCI in NB-IoT therefore this is the only way to configure the ndlsch_SI/ndlsch_SIB1 structures ndlsch->active = 1;

    		/*
    		 * OBSERVATION:
    		 * Although 2 DCI may be received over a schedule_response the transmission of the NDLSCH data foresees only 1 NDLSCH PDU at time.
    		 * Therefore is the MAC scheduler that knowing the different timing delay will send the corresponding schedule_response containing the NDLSCH PDU and the MAC PDU
    		 * at the proper DL subframe
    		 * -for this reason the activation of the ndslch structure is done only when we receive the NDLSCH pdu (here) such the in the TX procedure only 1 ue-specific pdu
    		 * 	result active from the loop before calling the ndlsch_procedure
    		 */

    		handle_nfapi_dlsch_pdu_NB_IoT(eNB, proc,dl_config_pdu,TX_req->tx_request_body.tx_pdu_list[dl_config_pdu->ndlsch_pdu.ndlsch_pdu_rel13.pdu_index].segments[0].segment_data);

    		break;
    	default:
    		LOG_E(PHY, "dl_config_pdu type not for NB_IoT\n");
    		break;
   }
  }
  
  for (i=0;i<number_ul_dci;i++) 
  {
    hi_dci0_pdu = &HI_DCI0_req->hi_dci0_request_body.hi_dci0_pdu_list[i];
    switch (hi_dci0_pdu->pdu_type) 
    {
    	case NFAPI_HI_DCI0_NPDCCH_DCI_PDU_TYPE:

      		generate_eNB_ulsch_params_NB_IoT(eNB,proc,hi_dci0_pdu);

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


  //XXX problem: although we may have nothing to transmit this function should be always triggered in order to allow the PHY layer to complete the repetitions
  //of previous Transport Blocks
  //phy_procedures_eNB_TX_NB_IoT(eNB,proc,NULL);
  phy_procedures_eNB_TX_NB_IoT(eNB,proc,0); // check if 0 or NULL ?!

}

void PHY_config_req_NB_IoT(PHY_Config_NB_IoT_t* config_INFO){
	LOG_I(PHY,"[NB-IoT] PHY CONFIG REQ NB-IoT In\n");


	if(config_INFO->get_MIB != 0){

		//MIB-NB configuration
		phy_config_mib_eNB_NB_IoT(config_INFO->mod_id,
								  config_INFO->CC_id,
							  	  config_INFO->cfg->nfapi_config.rf_bands.rf_band[0],//eutraband
							  	  config_INFO->cfg->sch_config.physical_cell_id.value,
							      config_INFO->cfg->subframe_config.dl_cyclic_prefix_type.value,
							  	  config_INFO->cfg->subframe_config.ul_cyclic_prefix_type.value,
							  	  config_INFO->cfg->rf_config.tx_antenna_ports.value,
							  	  config_INFO->cfg->nfapi_config.earfcn.value,
							  	  config_INFO->cfg->nb_iot_config.prb_index.value,
							  	  config_INFO->cfg->nb_iot_config.operating_mode.value,
							  	  config_INFO->cfg->nb_iot_config.control_region_size.value,
							  	  config_INFO->cfg->nb_iot_config.assumed_crs_aps.value); //defined only in in-band different PCI

	}

	if(config_INFO->get_COMMON != 0)
	{
		//Common Configuration included in SIB2-NB
		phy_config_sib2_eNB_NB_IoT(config_INFO->mod_id,
								   config_INFO->CC_id,
						       	   &config_INFO->cfg->nb_iot_config, // FIXME to be evaluated is should be passed a pointer
						       	   &config_INFO->cfg->rf_config,
							   	   &config_INFO->cfg->uplink_reference_signal_config,
							   	   &config_INFO->extra_phy_parms);
	}

	///FOR FAPI is not specified
	if(config_INFO->get_DEDICATED!= 0)
	{
	//Dedicated Configuration

			/*phy_config_dedicated_eNB_NB_IoT(config_INFO->mod_id,
											config_INFO->CC_id,
											config_INFO->rnti,
											&config_INFO->extra_phy_parms);*/

	}

	LOG_I(PHY,"IF Module for PHY Configuration has been done\n");
}

