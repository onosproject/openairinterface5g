#include "PHY_INTERFACE/IF_Module_NB_IoT.h"
#include "LAYER2/MAC/proto_NB_IoT.h"
#include "LAYER2/MAC/extern_NB_IoT.h"

int tmp = 0;

void simulate_preamble(UL_IND_NB_IoT_t *UL_INFO, int CE, int sc)
{
      UL_INFO->nrach_ind.number_of_initial_scs_detected = 1;
      UL_INFO->nrach_ind.nrach_pdu_list[0].nrach_indication_rel13.initial_sc       = sc;
      UL_INFO->nrach_ind.nrach_pdu_list[0].nrach_indication_rel13.timing_advance   = 0;
      UL_INFO->nrach_ind.nrach_pdu_list[0].nrach_indication_rel13.nrach_ce_level   = CE;
}

void enable_preamble_simulation(UL_IND_NB_IoT_t *UL_INFO,int i)
{
  if(i == 1)
  {
    // simulate preamble session
    /*
    if(UL_INFO->frame==60 && UL_INFO->subframe==2 && (tmp%3==0))
    {
      simulate_preamble(UL_INFO,0,2);
      tmp++;
    }
    if(UL_INFO->frame==100 && UL_INFO->subframe==2 && (tmp%3==1))
    {
      simulate_preamble(UL_INFO,1,13);
      tmp++;
    }
    */
    if(UL_INFO->frame==516 && UL_INFO->subframe==8)
    {
      simulate_preamble(UL_INFO,2,26);
      //tmp++;
    }
  }
}

void simulate_msg3(UL_IND_NB_IoT_t *UL_INFO)
{
	uint8_t *msg3 = NULL;
	msg3 = (uint8_t *) malloc (11*sizeof(uint8_t));
  msg3[0] = 0;
  msg3[1] = 58;
  msg3[2] = 42; // 2A
  msg3[3] = 179; // B3
  msg3[4] = 84; // 54
  msg3[5] = 141; // 8D
  msg3[6] = 43; // 2B
  msg3[7] = 52; // 34
  msg3[8] = 64; // 40
  msg3[9] = 0;
  msg3[10] = 0;
	UL_INFO->RX_NPUSCH.number_of_pdus = 1;
	UL_INFO->module_id = 0;
	UL_INFO->CC_id = 0;
	UL_INFO->frame = 521;
	UL_INFO->subframe = 1;
	UL_INFO->RX_NPUSCH.rx_pdu_list = (nfapi_rx_indication_pdu_t * )malloc(sizeof(nfapi_rx_indication_pdu_t));
	UL_INFO->RX_NPUSCH.rx_pdu_list->rx_ue_information.rnti = 0x0101;
	UL_INFO->RX_NPUSCH.rx_pdu_list->data = msg3;
	UL_INFO->RX_NPUSCH.rx_pdu_list->rx_indication_rel8.length = 11; 
}
void enable_msg3_simulation(UL_IND_NB_IoT_t *UL_INFO, int i)
{
  if(i==1)
  {
	if(UL_INFO->frame==521 && UL_INFO->subframe==1)
	{
		simulate_msg3(UL_INFO);
	}
  }
}
// Sched_INFO as a input for the scheduler
void UL_indication_NB_IoT(UL_IND_NB_IoT_t *UL_INFO)
{
    int i=0;
    uint32_t abs_subframe;
    Sched_Rsp_NB_IoT_t *SCHED_info = &mac_inst->Sched_INFO;

    enable_preamble_simulation(UL_INFO,0);

    enable_msg3_simulation(UL_INFO,0);

    //If there is a preamble, do the initiate RA procedure
    if(UL_INFO->nrach_ind.number_of_initial_scs_detected>0)
    {
      // only use one preamble now
      //for(i=0;i<UL_INFO->nrach_ind.number_of_initial_scs_detected;i++)
      for(i=0;i<1;i++)
      {
        // initiate_ra here, some useful inforamtion : 
        LOG_D(MAC,"Init_RA_NB_IoT in, index of sc = %d\n",(UL_INFO->nrach_ind.nrach_pdu_list+i)->nrach_indication_rel13.initial_sc);
        init_RA_NB_IoT(mac_inst,
                      (UL_INFO->nrach_ind.nrach_pdu_list+i)->nrach_indication_rel13.initial_sc,
                      (UL_INFO->nrach_ind.nrach_pdu_list+i)->nrach_indication_rel13.nrach_ce_level,
                      UL_INFO->frame,
                      //timing_offset = Timing_advance * 16
                      (UL_INFO->nrach_ind.nrach_pdu_list+i)->nrach_indication_rel13.timing_advance*16
                      );
      }
    }
    
    UL_INFO->nrach_ind.number_of_initial_scs_detected = 0;

    /* Disable crc function for now
    // crc indication if there is error for this round UL transmission
    if(UL_INFO->crc_ind.number_of_crcs>0)
    {
      for(i=0;i<UL_INFO->crc_ind.number_of_crcs;i++)
      {
        if((UL_INFO->crc_ind.crc_pdu_list+i)->crc_indication_rel8.crc_flag == 0)
        {
          //unsuccessfully received this UE PDU
          //UE_info = get_ue_from_rnti(mac_inst,((UL_INFO->crc_ind.crc_pdu_list)+i)->rx_ue_information.rnti);
          //UE_info->HARQ_round++;
        }
      }
    }
	 */
    // Check if there is any feed back of HARQ
    if(UL_INFO->nb_harq_ind.nb_harq_indication_body.number_of_harqs>0)
    {
      LOG_I(MAC,"Recieved Ack of DL Data, rnti : %x\n",UL_INFO->nb_harq_ind.nb_harq_indication_body.nb_harq_pdu_list[0].rx_ue_information.rnti);
      receive_msg4_ack_NB_IoT(mac_inst,UL_INFO->nb_harq_ind.nb_harq_indication_body.nb_harq_pdu_list[0].rx_ue_information.rnti);
    }

    UL_INFO->nb_harq_ind.nb_harq_indication_body.number_of_harqs = 0;
    //If there is a Uplink SDU which needs to send to MAC

    if(UL_INFO->RX_NPUSCH.number_of_pdus>0)
    {
      for(i=0;i<UL_INFO->RX_NPUSCH.number_of_pdus;i++)
      {
        //For MSG3, Normal Uplink Data, NAK
        rx_sdu_NB_IoT(UL_INFO->module_id,
                      UL_INFO->CC_id,
                      UL_INFO->frame,
                      UL_INFO->subframe,
                      (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.rnti,
                      (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->data,
                      (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_indication_rel8.length
                      );

      }
    }

	  UL_INFO->RX_NPUSCH.number_of_pdus = 0;

    if(UL_INFO->hypersfn==1 && UL_INFO->frame==0)
    {
      LOG_D(MAC,"IF L2 hypersfn:%d frame: %d ,subframe: %d \n",UL_INFO->hypersfn,UL_INFO->frame,UL_INFO->subframe);
    }
    
    abs_subframe = UL_INFO->hypersfn*10240+UL_INFO->frame*10+UL_INFO->subframe +4;
    //abs_subframe = UL_INFO->frame*10+UL_INFO->subframe +4;

    //LOG_I(MAC,"Enter scheduler in subframe %d\n",abs_subframe);
    //scheduler here
    //Schedule subframe should be next four subframe, means that UL_INFO->frame*10+UL_INFO->subframe + 4
    
    eNB_dlsch_ulsch_scheduler_NB_IoT(mac_inst,abs_subframe);
    mac_inst->if_inst_NB_IoT->schedule_response(&mac_inst->Sched_INFO);

    LOG_D(MAC,"After scheduler & schedule response\n");

    /*
    free(SCHED_info->TX_req->tx_request_body.tx_pdu_list);
    free(SCHED_info->HI_DCI0_req->hi_dci0_request_body.hi_dci0_pdu_list);
    free(SCHED_info->DL_req->dl_config_request_body.dl_config_pdu_list);
    free(SCHED_info->UL_req->ul_config_request_body.ul_config_pdu_list);
    
    free(SCHED_info->TX_req);
    free(SCHED_info->HI_DCI0_req);
    free(SCHED_info->DL_req);
    free(SCHED_info->UL_req);
    */
}
