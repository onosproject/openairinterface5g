
#include "LAYER2/MAC/defs-nb.h"
#include "LAYER2/MAC/extern.h"
#include "LAYER2/MAC/proto_nb_iot"
#include "IF_Module-nb.h"

#define Preamble_list 1;
#define ACK_NAK 2;
#define UL_SDU 3;

void UL_indication(UL_IND_t UL_INFO)
{
    /*Process Uplink Indication*/
	/*if(UL_MSG_flag)
  		{
  			switch(UL_MSG_flag)
  			{
  				case Preamble_list:
  					NB_initiate_ra_proc(module_idP,CC_id,frameP,preamble_index,timing_offset,subframeP);
  					break;
  				case UL_SDU:
  				    NB_rx_sdu(module_idP,CC_id,frameP,subframeP,rntiP,*sduP,sdu_lenP.harq_pidP,*msg3_flag);
  			}
  		} */

    



}