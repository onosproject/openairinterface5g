/*
 * phy_stub_UE.h
 *
 *  Created on: Sep 14, 2017
 *      Author: montre
 */


#ifndef __PHY_STUB_UE__H__
#define __PHY_STUB_UE__H__

#include <stdint.h>
#include "openair2/PHY_INTERFACE/IF_Module.h"
//#include "openair1/PHY/LTE_TRANSPORT/defs.h"
//#include "openair1/PHY/defs.h"
//#include "openair1/PHY/LTE_TRANSPORT/defs.h"


// Panos: This function should return all the sched_response config messages which concern a specific UE. Inside this
// function we should somehow make the translation of config message's rnti to Mod_ID.
Sched_Rsp_t get_nfapi_sched_response(uint8_t Mod_id);

// This function will be processing DL_config and Tx.requests and trigger all the MAC Rx related calls at the UE side,
// namely:ue_send_sdu(), or ue_decode_si(), or ue_decode_p(), or ue_process_rar() based on the rnti type.
void handle_nfapi_UE_Rx(uint8_t Mod_id, Sched_Rsp_t *Sched_INFO, int eNB_id);

// This function will be processing UL and HI_DCI0 config requests to trigger all the MAC Tx related calls
// at the UE side, namely: ue_get_SR(), ue_get_rach(), ue_get_sdu() based on the pdu configuration type.
// The output of these calls will be put to an UL_IND_t structure which will then be the input to
// send_nfapi_UL_indications().
UL_IND_t generate_nfapi_UL_indications(Sched_Rsp_t sched_response);

// This function should pass the UL indication messages to the eNB side through the socket interface.
void send_nfapi_UL_indications(UL_IND_t UL_INFO);


#endif /* PHY_STUB_UE_H_ */
