#include "openair1/PHY/defs.h"
#include "openair2/PHY_INTERFACE/IF_Module_UE.h"
#include "openair2/PHY_INTERFACE/UE_MAC_interface.h"
#include "openair1/PHY/extern.h"
#include "LAYER2/MAC/extern.h"
#include "LAYER2/MAC/proto.h"
#include "common/ran_context.h"



void handle_bch(UE_DL_IND_t *UE_DL_INFO)
{

}


void handle_dlsch(UE_DL_IND_t *UE_DL_INFO)
{
	int i;
	UE_MAC_dlsch_indication_pdu_t *dlsch_pdu_ind;

	for (i=0; i<UE_DL_INFO->UE_DLSCH_ind.number_of_pdus; i++)
	{
		dlsch_pdu_ind = UE_DL_INFO->UE_DLSCH_ind.dlsch_ind_list[i];
		switch (dlsch_pdu_ind->pdu_type) {

		case UE_MAC_DL_IND_PDSCH_PDU_TYPE:
				// Call ue_send_sdu()
			break;

		case UE_MAC_DL_IND_SI_PDSCH_PDU_TYPE:
				// Call ue_decode_si()
			break;

		case UE_MAC_DL_IND_P_PDSCH_PDU_TYPE:
				// Call ue_decode_p()
			break;

		case UE_MAC_DL_IND_DLSCH_RAR_PDU_TYPE:
			   // Call ue_process_rar()
			break;
		}
	}

}


void UE_DL_indication(UE_DL_IND_t *UE_DL_INFO)
{

    /*Call handle functions to forward PDUs or control indications to the upper layers.
	handle_bch(UE_DL_INFO);
	handle_dlsch (UE_DL_INFO);
	*/
}

/* Indicate the Txon of Msg1 or Msg3 to the MAC layer of the transmitter side and trigger associated
 * MAC layer operations */

void UE_Tx_indication(UE_Tx_IND_t *UE_Tx_INFO)
{
	switch (UE_Tx_INFO->ind_type) {

	case UE_MAC_Tx_IND_Msg1_TYPE:
		//Call Msg1_transmitted()
	break;

	case UE_MAC_Tx_IND_Msg3_TYPE:
		//Call Msg3_transmitted()
	break;
	}
}





