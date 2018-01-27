#include "IF_Module_L2_primitives_NB_IoT.h"
#include "LAYER2/MAC/proto_NB_IoT.h"


// Sched_INFO as a input for the scheduler
void UL_indication_NB_IoT(UL_IND_NB_IoT_t *UL_INFO)
{
    int i=0;
    //UE_TEMPLATE_NB_IoT *UE_info;
    //mac_NB_IoT_t *mac_inst;

      //If there is a preamble, do the initiate RA procedure
      if(UL_INFO->NRACH.number_of_initial_scs_detected>0)
        {
          for(i=0;i<UL_INFO->NRACH.number_of_initial_scs_detected;i++)
            {
              // initiate_ra here, some useful inforamtion : 
              //(UL_INFO->NRACH.nrach_pdu_list+i)->nrach_indication_rel13.initial_sc
              //(UL_INFO->NRACH.nrach_pdu_list+i)->nrach_indication_rel13.timing_advance
              /*init_RA_NB_IoT(UL_INFO->module_id,
                                      UL_INFO->CC_id,
                                      UL_INFO->frame,
                                      (UL_INFO->NRACH.nrach_pdu_list+i)->nrach_indication_rel13.initial_sc,
                                      //timing_offset = Timing_advance * 16
                                      (UL_INFO->NRACH.nrach_pdu_list+i)->nrach_indication_rel13.timing_advance * 16,
                                      UL_INFO->subframe
                                     );*/

            }
        }

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

        /*If there is a Uplink SDU which needs to send to MAC*/

        if(UL_INFO->RX_NPUSCH.number_of_pdus>0)
          {
            for(i=0;i<UL_INFO->RX_NPUSCH.number_of_pdus;i++)
              {
                /*For MSG3, Normal Uplink Data, NAK*/
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

    //scheduler here
    //Schedule subframe should be next four subframe, means that UL_INFO->frame*10+UL_INFO->subframe + 4
    //eNB_dlsch_ulsch_scheduler_NB_IoT(mac_inst,abs_subframe);
}