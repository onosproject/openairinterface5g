#include "IF_Module_L2_primitives_NB_IoT.h"
#include "LAYER2/MAC/proto_NB_IoT.h"
// Sched_INFO as a input for the scheduler
void UL_indication(UL_IND_t *UL_INFO)
{
    int i=0;

      /*If there is a preamble, do the initiate RA procedure*/
      if(UL_INFO->NRACH.number_of_initial_scs_detected>0)
        {
          for(i=0;i<UL_INFO->NRACH.number_of_initial_scs_detected;i++)
            {
              initiate_ra_proc_NB_IoT(UL_INFO->module_id,
                                      UL_INFO->CC_id,
                                      UL_INFO->frame,
                                      (UL_INFO->NRACH.nrach_pdu_list+i)->nrach_indication_rel13.initial_sc,
                                      //timing_offset = Timing_advance * 16
                                      (UL_INFO->NRACH.nrach_pdu_list+i)->nrach_indication_rel13.timing_advance * 16,
                                      UL_INFO->subframe
                                     );
            }
        }
        if(UL_INFO->RX_NPUSCH.number_of_pdus>0)
          {
            /*If there is a Uplink SDU (even MSG3, NAK) need to send to MAC*/
            for(i=0;i<UL_INFO->RX_NPUSCH.number_of_pdus;i++)
              {
                /*For MSG3, Normal Uplink Data, NAK*/
                rx_sdu_NB_IoT(UL_INFO->module_id,
                              UL_INFO->CC_id,
                              UL_INFO->frame,
                              UL_INFO->subframe,
                              (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.rnti,
                              (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->data,
                              (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_indication_rel8.length,
                              (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.harq_pid
                              );


              }

          }

    //eNB_dlsch_ulsch_scheduler_NB_IoT(UL_INFO.module_id,0,UL_INFO.frame,UL_INFO.subframe); TODO: to be implemented
}
