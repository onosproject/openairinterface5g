#include "openair2/PHY_INTERFACE/IF_Module_nb_iot.h"
#include "LAYER2/MAC/extern.h"
#include "LAYER2/MAC/proto_nb_iot.h"

void UL_indication(UL_IND_t UL_INFO)
{
    int i=0;
    UL_INFO.test=1;
    if(UL_INFO.test == 1)
        {
          /*If there is a preamble, do the initiate RA procedure*/
          if(UL_INFO.preamble_index && UL_INFO.timing_offset)
              NB_initiate_ra_proc(UL_INFO.module_id,UL_INFO.CC_id,UL_INFO.frame,UL_INFO.preamble_index,UL_INFO.timing_offset,UL_INFO.subframe);

          /*If there is a Uplink SDU (even MSG3, NAK) need to send to MAC*/
          for(i=0;i<UL_INFO.UE_NUM;i++)
              {
                  /*For MSG3, Normal Uplink Data, NAK*/
                  if(UL_INFO.UL_SPEC_Info[i].rntiP)
                      NB_rx_sdu(UL_INFO.module_id,
                                UL_INFO.CC_id,
                                UL_INFO.frame,
                                UL_INFO.subframe,
                                UL_INFO.UL_SPEC_Info[i].rntiP,
                                UL_INFO.UL_SPEC_Info[i].sdu,
                                UL_INFO.UL_SPEC_Info[i].sdu_lenP,
                                UL_INFO.UL_SPEC_Info[i].harq_pidP,
                                UL_INFO.UL_SPEC_Info[i].msg3_flagP
                                );
                    

              }
        }

    NB_eNB_dlsch_ulsch_scheduler(UL_INFO.module_id,0,UL_INFO.frame,UL_INFO.subframe);
}

void Schedule_Response(Sched_Rsp_t Sched_INFO){
      //todo
}