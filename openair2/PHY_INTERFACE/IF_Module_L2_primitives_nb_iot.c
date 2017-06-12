#include "IF_Module_L2_primitives_nb_iot.h"

void UL_indication(UL_IND_t UL_INFO)
{
    int i=0;
    UL_INFO.test=1;
    if(UL_INFO.test == 1)
        {
          /*If there is a preamble, do the initiate RA procedure*/
          if(UL_INFO.Number_SC>0)
            {
              for(i=0;i<UL_INFO.Number_SC;i++)
              {
                NB_initiate_ra_proc(UL_INFO.module_id,
                                    UL_INFO.CC_id,
                                    UL_INFO.frame,
                                    UL_INFO.Preamble_list[UL_INFO.Number_SC].preamble_index,
                                    UL_INFO.Preamble_list[UL_INFO.Number_SC].timing_offset,
                                    UL_INFO.subframe
                                    );
              }
            }
          /*If there is a Uplink SDU (even MSG3, NAK) need to send to MAC*/
          for(i=0;i<UL_INFO.UE_NUM;i++)
              {
                  /*For MSG3, Normal Uplink Data, NAK*/
                  if(UL_INFO.UL_SPEC_Info[i].RNTI)
                      NB_rx_sdu(UL_INFO.module_id,
                                UL_INFO.CC_id,
                                UL_INFO.frame,
                                UL_INFO.subframe,
                                UL_INFO.UL_SPEC_Info[i].RNTI,
                                UL_INFO.UL_SPEC_Info[i].sdu,
                                UL_INFO.UL_SPEC_Info[i].sdu_lenP,
                                UL_INFO.UL_SPEC_Info[i].harq_pidP,
                                UL_INFO.UL_SPEC_Info[i].msg3_flagP
                                );


              }
        }

    //NB_eNB_dlsch_ulsch_scheduler(UL_INFO.module_id,0,UL_INFO.frame,UL_INFO.subframe); TODO: to be implemented
}
