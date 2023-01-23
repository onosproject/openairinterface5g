/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file f1ap_cu_interface_management.c
 * \brief f1ap interface management for CU
 * \author EURECOM/NTUST
 * \date 2018
 * \version 0.1
 * \company Eurecom
 * \email: navid.nikaein@eurecom.fr, bing-kai.hong@eurecom.fr
 * \note
 * \warning
 */

#include "f1ap_common.h"
#include "f1ap_encoder.h"
#include "f1ap_decoder.h"
#include "f1ap_itti_messaging.h"
#include "f1ap_cu_interface_management.h"

extern f1ap_setup_req_t *f1ap_du_data_from_du;

int CU_send_RESET(instance_t instance, F1AP_Reset_t *Reset) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int CU_handle_RESET_ACKKNOWLEDGE(instance_t instance,
                                  uint32_t assoc_id,
                                  uint32_t stream,
                                  F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int CU_handle_RESET(instance_t instance,
                     uint32_t assoc_id,
                     uint32_t stream,
                     F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int CU_send_RESET_ACKNOWLEDGE(instance_t instance, F1AP_ResetAcknowledge_t *ResetAcknowledge) {
  AssertFatal(1==0,"Not implemented yet\n");
}


/*
    Error Indication
*/
int CU_handle_ERROR_INDICATION(instance_t instance,
                                uint32_t assoc_id,
                                uint32_t stream,
                                F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int CU_send_ERROR_INDICATION(instance_t instance, F1AP_ErrorIndication_t *ErrorIndication) {
  AssertFatal(1==0,"Not implemented yet\n");
}


/*
    F1 Setup
*/

void *g_cuF1SetupReq;
uint32_t g_cuF1SetupReqSize;

int CU_handle_F1_SETUP_REQUEST(instance_t instance,
                               uint32_t assoc_id,
                               uint32_t stream,
                               F1AP_F1AP_PDU_t *pdu)
{
  LOG_D(F1AP, "CU_handle_F1_SETUP_REQUEST\n");
  
  MessageDef                         *message_p;
  F1AP_F1SetupRequest_t              *container;
  F1AP_F1SetupRequestIEs_t           *ie;
  int i = 0;
   

  DevAssert(pdu != NULL);

  /* Create */
  g_cuF1SetupReq = malloc(sizeof(F1AP_F1SetupRequest_t));
  memset(g_cuF1SetupReq, 0, sizeof(F1AP_F1SetupRequest_t));
  g_cuF1SetupReqSize = sizeof(F1AP_F1SetupRequest_t);

  container = &pdu->choice.initiatingMessage->value.choice.F1SetupRequest;

  /*copying to global buffer for E2Cell Info to be sent during E2SetupReq */
  memcpy(g_cuF1SetupReq, container, sizeof(F1AP_F1SetupRequest_t));
  LOG_I(F1AP, "++ CU Received F1SetupReq for E2Cell Info copied ++\n");


  /* F1 Setup Request == Non UE-related procedure -> stream 0 */
  if (stream != 0) {
    LOG_W(F1AP, "[SCTP %d] Received f1 setup request on stream != 0 (%d)\n",
              assoc_id, stream);
  }

  message_p = itti_alloc_new_message(TASK_RRC_ENB, F1AP_SETUP_REQ); 
  
  /* assoc_id */
  F1AP_SETUP_REQ(message_p).assoc_id = assoc_id;
  
  /* gNB_DU_id */
  // this function exits if the ie is mandatory
  F1AP_FIND_PROTOCOLIE_BY_ID(F1AP_F1SetupRequestIEs_t, ie, container,
                             F1AP_ProtocolIE_ID_id_gNB_DU_ID, true);
  asn_INTEGER2ulong(&ie->value.choice.GNB_DU_ID, &F1AP_SETUP_REQ(message_p).gNB_DU_id);
  LOG_D(F1AP, "F1AP_SETUP_REQ(message_p).gNB_DU_id %lu \n", F1AP_SETUP_REQ(message_p).gNB_DU_id);

  /* gNB_DU_name */
  F1AP_FIND_PROTOCOLIE_BY_ID(F1AP_F1SetupRequestIEs_t, ie, container,
                              F1AP_ProtocolIE_ID_id_gNB_DU_Name, true);
  F1AP_SETUP_REQ(message_p).gNB_DU_name = calloc(ie->value.choice.GNB_DU_Name.size + 1, sizeof(char));
  memcpy(F1AP_SETUP_REQ(message_p).gNB_DU_name, ie->value.choice.GNB_DU_Name.buf,
         ie->value.choice.GNB_DU_Name.size);
  /* Convert the mme name to a printable string */
  F1AP_SETUP_REQ(message_p).gNB_DU_name[ie->value.choice.GNB_DU_Name.size] = '\0';
  LOG_D(F1AP, "F1AP_SETUP_REQ(message_p).gNB_DU_name %s \n", F1AP_SETUP_REQ(message_p).gNB_DU_name);

  /* GNB_DU_Served_Cells_List */
  F1AP_FIND_PROTOCOLIE_BY_ID(F1AP_F1SetupRequestIEs_t, ie, container,
                              F1AP_ProtocolIE_ID_id_gNB_DU_Served_Cells_List, true);
  F1AP_SETUP_REQ(message_p).num_cells_available = ie->value.choice.GNB_DU_Served_Cells_List.list.count;
  LOG_D(F1AP, "F1AP_SETUP_REQ(message_p).num_cells_available %d \n",
        F1AP_SETUP_REQ(message_p).num_cells_available);

  int num_cells_available = F1AP_SETUP_REQ(message_p).num_cells_available;

  for (i=0; i<num_cells_available; i++) {
    F1AP_GNB_DU_Served_Cells_Item_t *served_celles_item_p;

    served_celles_item_p = &(((F1AP_GNB_DU_Served_Cells_ItemIEs_t *)ie->value.choice.GNB_DU_Served_Cells_List.list.array[i])->value.choice.GNB_DU_Served_Cells_Item);
    
    /* tac */
    OCTET_STRING_TO_INT16(&(served_celles_item_p->served_Cell_Information.fiveGS_TAC), F1AP_SETUP_REQ(message_p).tac[i]);
    LOG_D(F1AP, "F1AP_SETUP_REQ(message_p).tac[%d] %d \n",
          i, F1AP_SETUP_REQ(message_p).tac[i]);

    /* - nRCGI */
    TBCD_TO_MCC_MNC(&(served_celles_item_p->served_Cell_Information.nRCGI.pLMN_Identity), F1AP_SETUP_REQ(message_p).mcc[i],
                    F1AP_SETUP_REQ(message_p).mnc[i],
                    F1AP_SETUP_REQ(message_p).mnc_digit_length[i]);
    
    
    // NR cellID
    BIT_STRING_TO_NR_CELL_IDENTITY(&served_celles_item_p->served_Cell_Information.nRCGI.nRCellIdentity,
				   F1AP_SETUP_REQ(message_p).nr_cellid[i]);
    LOG_D(F1AP, "[SCTP %d] Received nRCGI: MCC %d, MNC %d, CELL_ID %llu\n", assoc_id,
          F1AP_SETUP_REQ(message_p).mcc[i],
          F1AP_SETUP_REQ(message_p).mnc[i],
          (long long unsigned int)F1AP_SETUP_REQ(message_p).nr_cellid[i]);
    LOG_D(F1AP, "nr_cellId : %x %x %x %x %x\n",
          served_celles_item_p->served_Cell_Information.nRCGI.nRCellIdentity.buf[0],
          served_celles_item_p->served_Cell_Information.nRCGI.nRCellIdentity.buf[1],
          served_celles_item_p->served_Cell_Information.nRCGI.nRCellIdentity.buf[2],
          served_celles_item_p->served_Cell_Information.nRCGI.nRCellIdentity.buf[3],
          served_celles_item_p->served_Cell_Information.nRCGI.nRCellIdentity.buf[4]);
    /* - nRPCI */
    F1AP_SETUP_REQ(message_p).nr_pci[i] = served_celles_item_p->served_Cell_Information.nRPCI;
    LOG_D(F1AP, "F1AP_SETUP_REQ(message_p).nr_pci[%d] %d \n",
          i, F1AP_SETUP_REQ(message_p).nr_pci[i]);
  
    // System Information
    /* mib */
    F1AP_SETUP_REQ(message_p).mib[i] = calloc(served_celles_item_p->gNB_DU_System_Information->mIB_message.size + 1, sizeof(char));
    memcpy(F1AP_SETUP_REQ(message_p).mib[i], served_celles_item_p->gNB_DU_System_Information->mIB_message.buf,
           served_celles_item_p->gNB_DU_System_Information->mIB_message.size);
    /* Convert the mme name to a printable string */
    F1AP_SETUP_REQ(message_p).mib[i][served_celles_item_p->gNB_DU_System_Information->mIB_message.size] = '\0';
    F1AP_SETUP_REQ(message_p).mib_length[i] = served_celles_item_p->gNB_DU_System_Information->mIB_message.size;
    LOG_D(F1AP, "F1AP_SETUP_REQ(message_p).mib[%d] %s , len = %d \n",
          i, F1AP_SETUP_REQ(message_p).mib[i], F1AP_SETUP_REQ(message_p).mib_length[i]);

    /* sib1 */
    F1AP_SETUP_REQ(message_p).sib1[i] = calloc(served_celles_item_p->gNB_DU_System_Information->sIB1_message.size + 1, sizeof(char));
    memcpy(F1AP_SETUP_REQ(message_p).sib1[i], served_celles_item_p->gNB_DU_System_Information->sIB1_message.buf,
           served_celles_item_p->gNB_DU_System_Information->sIB1_message.size);
    /* Convert the mme name to a printable string */
    F1AP_SETUP_REQ(message_p).sib1[i][served_celles_item_p->gNB_DU_System_Information->sIB1_message.size] = '\0';
    F1AP_SETUP_REQ(message_p).sib1_length[i] = served_celles_item_p->gNB_DU_System_Information->sIB1_message.size;
    LOG_D(F1AP, "F1AP_SETUP_REQ(message_p).sib1[%d] %s , len = %d \n",
          i, F1AP_SETUP_REQ(message_p).sib1[i], F1AP_SETUP_REQ(message_p).sib1_length[i]);
  }

  
  *f1ap_du_data_from_du = F1AP_SETUP_REQ(message_p);
  // char *measurement_timing_information[F1AP_MAX_NB_CELLS];
  // uint8_t ranac[F1AP_MAX_NB_CELLS];

  // int fdd_flag = f1ap_setup_req->fdd_flag;

  // union {
  //   struct {
  //     uint32_t ul_nr_arfcn;
  //     uint8_t ul_scs;
  //     uint8_t ul_nrb;

  //     uint32_t dl_nr_arfcn;
  //     uint8_t dl_scs;
  //     uint8_t dl_nrb;

  //     uint32_t sul_active;
  //     uint32_t sul_nr_arfcn;
  //     uint8_t sul_scs;
  //     uint8_t sul_nrb;

  //     uint8_t num_frequency_bands;
  //     uint16_t nr_band[32];
  //     uint8_t num_sul_frequency_bands;
  //     uint16_t nr_sul_band[32];
  //   } fdd;
  //   struct {

  //     uint32_t nr_arfcn;
  //     uint8_t scs;
  //     uint8_t nrb;

  //     uint32_t sul_active;
  //     uint32_t sul_nr_arfcn;
  //     uint8_t sul_scs;
  //     uint8_t sul_nrb;

  //     uint8_t num_frequency_bands;
  //     uint16_t nr_band[32];
  //     uint8_t num_sul_frequency_bands;
  //     uint16_t nr_sul_band[32];

  //   } tdd;
  // } nr_mode_info[F1AP_MAX_NB_CELLS];

  MSC_LOG_TX_MESSAGE(
  MSC_F1AP_CU,
  MSC_RRC_ENB,
  0,
  0,
  MSC_AS_TIME_FMT" CU_handle_F1_SETUP_REQUEST",
  0,0//MSC_AS_TIME_ARGS(ctxt_pP),
  );

  if (num_cells_available > 0) {
    itti_send_msg_to_task(TASK_RRC_ENB, ENB_MODULE_ID_TO_INSTANCE(instance), message_p);
  } else {
    CU_send_F1_SETUP_FAILURE(instance);
    itti_free(TASK_RRC_ENB,message_p);
    return -1;
  }
  return 0;
}

void *g_cuF1SetupResp;
int32_t g_cuF1SetupRespSize;

int CU_send_F1_SETUP_RESPONSE(instance_t instance,
                               f1ap_setup_resp_t *f1ap_setup_resp) 
{
  
  module_id_t enb_mod_idP;
  module_id_t cu_mod_idP;

  // This should be fixed
  enb_mod_idP = (module_id_t)0;
  cu_mod_idP  = (module_id_t)0;

  F1AP_F1AP_PDU_t           pdu;
  F1AP_F1SetupResponse_t    *out;
  F1AP_F1SetupResponseIEs_t *ie;

  uint8_t  *buffer;
  uint32_t  len;
  int       i = 0;

  /* Create */
  g_cuF1SetupResp = malloc(sizeof(F1AP_F1SetupResponse_t));
  memset(g_cuF1SetupResp, 0, sizeof(F1AP_F1SetupResponse_t));
  g_cuF1SetupRespSize = sizeof(F1AP_F1SetupResponse_t);

  /* 0. Message Type */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = F1AP_F1AP_PDU_PR_successfulOutcome;
  pdu.choice.successfulOutcome = (F1AP_SuccessfulOutcome_t *)calloc(1, sizeof(F1AP_SuccessfulOutcome_t));
  pdu.choice.successfulOutcome->procedureCode = F1AP_ProcedureCode_id_F1Setup;
  pdu.choice.successfulOutcome->criticality   = F1AP_Criticality_reject;
  pdu.choice.successfulOutcome->value.present = F1AP_SuccessfulOutcome__value_PR_F1SetupResponse;
  out = &pdu.choice.successfulOutcome->value.choice.F1SetupResponse;
  
  /* mandatory */
  /* c1. Transaction ID (integer value)*/
  ie = (F1AP_F1SetupResponseIEs_t *)calloc(1, sizeof(F1AP_F1SetupResponseIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_TransactionID;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_F1SetupResponseIEs__value_PR_TransactionID;
  ie->value.choice.TransactionID = F1AP_get_next_transaction_identifier(enb_mod_idP, cu_mod_idP);
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);
 
  /* optional */
  /* c2. GNB_CU_Name */
  if (f1ap_setup_resp->gNB_CU_name != NULL) {
    ie = (F1AP_F1SetupResponseIEs_t *)calloc(1, sizeof(F1AP_F1SetupResponseIEs_t));
    ie->id                        = F1AP_ProtocolIE_ID_id_gNB_CU_Name;
    ie->criticality               = F1AP_Criticality_ignore;
    ie->value.present             = F1AP_F1SetupResponseIEs__value_PR_GNB_CU_Name;
    OCTET_STRING_fromBuf(&ie->value.choice.GNB_CU_Name, f1ap_setup_resp->gNB_CU_name,
                         strlen(f1ap_setup_resp->gNB_CU_name));
    ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);
  }

  /* mandatory */
  /* c3. cells to be Activated list */
  ie = (F1AP_F1SetupResponseIEs_t *)calloc(1, sizeof(F1AP_F1SetupResponseIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_Cells_to_be_Activated_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_F1SetupResponseIEs__value_PR_Cells_to_be_Activated_List;

  int num_cells_to_activate = f1ap_setup_resp->num_cells_to_activate;
  LOG_D(F1AP, "num_cells_to_activate = %d \n", num_cells_to_activate);
  for (i=0;
       i<num_cells_to_activate;
       i++) {

    F1AP_Cells_to_be_Activated_List_ItemIEs_t *cells_to_be_activated_list_item_ies;
    cells_to_be_activated_list_item_ies = (F1AP_Cells_to_be_Activated_List_ItemIEs_t *)calloc(1, sizeof(F1AP_Cells_to_be_Activated_List_ItemIEs_t));
    cells_to_be_activated_list_item_ies->id = F1AP_ProtocolIE_ID_id_Cells_to_be_Activated_List_Item;
    cells_to_be_activated_list_item_ies->criticality = F1AP_Criticality_reject;
    cells_to_be_activated_list_item_ies->value.present = F1AP_Cells_to_be_Activated_List_ItemIEs__value_PR_Cells_to_be_Activated_List_Item;

    /* 3.1 cells to be Activated list item */
    F1AP_Cells_to_be_Activated_List_Item_t cells_to_be_activated_list_item;
    memset((void *)&cells_to_be_activated_list_item, 0, sizeof(F1AP_Cells_to_be_Activated_List_Item_t));

    /* - nRCGI */
    F1AP_NRCGI_t nRCGI;
    memset(&nRCGI, 0, sizeof(F1AP_NRCGI_t));
    MCC_MNC_TO_PLMNID(f1ap_setup_resp->mcc[i], f1ap_setup_resp->mnc[i], f1ap_setup_resp->mnc_digit_length[i],
                                     &nRCGI.pLMN_Identity);
    NR_CELL_ID_TO_BIT_STRING(f1ap_setup_resp->nr_cellid[i], &nRCGI.nRCellIdentity);
    cells_to_be_activated_list_item.nRCGI = nRCGI;

    /* optional */
    /* - nRPCI */
    if (1) {
      cells_to_be_activated_list_item.nRPCI = (F1AP_NRPCI_t *)calloc(1, sizeof(F1AP_NRPCI_t));
      *cells_to_be_activated_list_item.nRPCI = f1ap_setup_resp->nrpci[i];  // int 0..1007
    }

    /* optional */
    /* - gNB-CU System Information */
    if (1) {
      /* 3.1.2 gNB-CUSystem Information */
      F1AP_Cells_to_be_Activated_List_ItemExtIEs_t *cells_to_be_activated_list_itemExtIEs;
      cells_to_be_activated_list_itemExtIEs = (F1AP_Cells_to_be_Activated_List_ItemExtIEs_t *)calloc(1, sizeof(F1AP_Cells_to_be_Activated_List_ItemExtIEs_t));
      cells_to_be_activated_list_itemExtIEs->id                     = F1AP_ProtocolIE_ID_id_gNB_CUSystemInformation;
      cells_to_be_activated_list_itemExtIEs->criticality            = F1AP_Criticality_reject;
      cells_to_be_activated_list_itemExtIEs->extensionValue.present = F1AP_Cells_to_be_Activated_List_ItemExtIEs__extensionValue_PR_GNB_CUSystemInformation;

      F1AP_GNB_CUSystemInformation_t *gNB_CUSystemInformation = (F1AP_GNB_CUSystemInformation_t *)calloc(1, sizeof(F1AP_GNB_CUSystemInformation_t));
      //LOG_I(F1AP, "%s() SI %d size %d: ", __func__, i, f1ap_setup_resp->SI_container_length[i][0]);
      //for (int n = 0; n < f1ap_setup_resp->SI_container_length[i][0]; n++)
      //  printf("%02x ", f1ap_setup_resp->SI_container[i][0][n]);
      //printf("\n");
      OCTET_STRING_fromBuf(&gNB_CUSystemInformation->sImessage,
                           (const char*)f1ap_setup_resp->SI_container[i][0], 
                           f1ap_setup_resp->SI_container_length[i][0]);

      LOG_D(F1AP, "f1ap_setup_resp->SI_container_length = %d \n", f1ap_setup_resp->SI_container_length[0][0]);
      cells_to_be_activated_list_itemExtIEs->extensionValue.choice.GNB_CUSystemInformation = *gNB_CUSystemInformation;


      F1AP_ProtocolExtensionContainer_160P9_t p_160P9_t;
      memset((void *)&p_160P9_t, 0, sizeof(F1AP_ProtocolExtensionContainer_160P9_t));

      ASN_SEQUENCE_ADD(&p_160P9_t.list,
                      cells_to_be_activated_list_itemExtIEs);
      cells_to_be_activated_list_item.iE_Extensions = (struct F1AP_ProtocolExtensionContainer*)&p_160P9_t;

      free(gNB_CUSystemInformation);
      gNB_CUSystemInformation = NULL;
    }
    /* ADD */
    cells_to_be_activated_list_item_ies->value.choice.Cells_to_be_Activated_List_Item = cells_to_be_activated_list_item;
    ASN_SEQUENCE_ADD(&ie->value.choice.Cells_to_be_Activated_List.list,
                  cells_to_be_activated_list_item_ies);
  }
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);

  /*copying to global buffer for E2Cell Info to be sent during E2SetupReq */
  memcpy(g_cuF1SetupResp, out, sizeof(F1AP_F1SetupResponse_t));
  LOG_I(F1AP, "++ CU F1SetupResp for E2Cell Info copied ++\n");


  /* encode */
  if (f1ap_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(F1AP, "Failed to encode F1 setup request\n");
    return -1;
  }

  cu_f1ap_itti_send_sctp_data_req(instance, f1ap_du_data_from_du->assoc_id, buffer, len, 0);

  return 0;
}

int CU_send_F1_SETUP_FAILURE(instance_t instance) {
  LOG_D(F1AP, "CU_send_F1_SETUP_FAILURE\n");
  
  module_id_t enb_mod_idP;
  module_id_t cu_mod_idP;

  // This should be fixed
  enb_mod_idP = (module_id_t)0;
  cu_mod_idP  = (module_id_t)0;

  F1AP_F1AP_PDU_t           pdu;
  F1AP_F1SetupFailure_t    *out;
  F1AP_F1SetupFailureIEs_t *ie;

  uint8_t  *buffer;
  uint32_t  len;

  /* Create */
  /* 0. Message Type */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = F1AP_F1AP_PDU_PR_unsuccessfulOutcome;
  pdu.choice.unsuccessfulOutcome = (F1AP_UnsuccessfulOutcome_t *)calloc(1, sizeof(F1AP_UnsuccessfulOutcome_t));
  pdu.choice.unsuccessfulOutcome->procedureCode = F1AP_ProcedureCode_id_F1Setup;
  pdu.choice.unsuccessfulOutcome->criticality   = F1AP_Criticality_reject;
  pdu.choice.unsuccessfulOutcome->value.present = F1AP_UnsuccessfulOutcome__value_PR_F1SetupFailure;
  out = &pdu.choice.unsuccessfulOutcome->value.choice.F1SetupFailure;

  /* mandatory */
  /* c1. Transaction ID (integer value)*/
  ie = (F1AP_F1SetupFailureIEs_t *)calloc(1, sizeof(F1AP_F1SetupFailureIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_TransactionID;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_F1SetupFailureIEs__value_PR_TransactionID;
  ie->value.choice.TransactionID = F1AP_get_next_transaction_identifier(enb_mod_idP, cu_mod_idP);
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);

  /* mandatory */
  /* c2. Cause */
  ie = (F1AP_F1SetupFailureIEs_t *)calloc(1, sizeof(F1AP_F1SetupFailureIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_Cause;
  ie->criticality               = F1AP_Criticality_ignore;
  ie->value.present             = F1AP_F1SetupFailureIEs__value_PR_Cause;
  ie->value.choice.Cause.present = F1AP_Cause_PR_radioNetwork;
  ie->value.choice.Cause.choice.radioNetwork = F1AP_CauseRadioNetwork_unspecified;
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);

  /* optional */
  /* c3. TimeToWait */
  if (0) {
    ie = (F1AP_F1SetupFailureIEs_t *)calloc(1, sizeof(F1AP_F1SetupFailureIEs_t));
    ie->id                        = F1AP_ProtocolIE_ID_id_TimeToWait;
    ie->criticality               = F1AP_Criticality_ignore;
    ie->value.present             = F1AP_F1SetupFailureIEs__value_PR_TimeToWait;
    ie->value.choice.TimeToWait = F1AP_TimeToWait_v10s;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);
  }

  /* optional */
  /* c4. CriticalityDiagnostics*/
  if (0) {
    ie = (F1AP_F1SetupFailureIEs_t *)calloc(1, sizeof(F1AP_F1SetupFailureIEs_t));
    ie->id                        = F1AP_ProtocolIE_ID_id_CriticalityDiagnostics;
    ie->criticality               = F1AP_Criticality_ignore;
    ie->value.present             = F1AP_F1SetupFailureIEs__value_PR_CriticalityDiagnostics;
    ie->value.choice.CriticalityDiagnostics.procedureCode = (F1AP_ProcedureCode_t *)calloc(1, sizeof(F1AP_ProcedureCode_t));
    *ie->value.choice.CriticalityDiagnostics.procedureCode = F1AP_ProcedureCode_id_UEContextSetup;
    ie->value.choice.CriticalityDiagnostics.triggeringMessage = (F1AP_TriggeringMessage_t *)calloc(1, sizeof(F1AP_TriggeringMessage_t));
    *ie->value.choice.CriticalityDiagnostics.triggeringMessage = F1AP_TriggeringMessage_initiating_message;
    ie->value.choice.CriticalityDiagnostics.procedureCriticality = (F1AP_Criticality_t *)calloc(1, sizeof(F1AP_Criticality_t));
    *ie->value.choice.CriticalityDiagnostics.procedureCriticality = F1AP_Criticality_reject;
    ie->value.choice.CriticalityDiagnostics.transactionID = (F1AP_TransactionID_t *)calloc(1, sizeof(F1AP_TransactionID_t));
    *ie->value.choice.CriticalityDiagnostics.transactionID = 0;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);
  }

  /* encode */
  if (f1ap_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(F1AP, "Failed to encode F1 setup request\n");
    return -1;
  }

  cu_f1ap_itti_send_sctp_data_req(instance, f1ap_du_data_from_du->assoc_id, buffer, len, 0);

  return 0;
}



/*
    gNB-DU Configuration Update
*/

int CU_handle_gNB_DU_CONFIGURATION_UPDATE(instance_t instance,
                                           uint32_t assoc_id,
                                           uint32_t stream,
                                           F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int CU_send_gNB_DU_CONFIGURATION_FAILURE(instance_t instance,
                    F1AP_GNBDUConfigurationUpdateFailure_t *GNBDUConfigurationUpdateFailure) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int CU_send_gNB_DU_CONFIGURATION_UPDATE_ACKNOWLEDGE(instance_t instance,
                    F1AP_GNBDUConfigurationUpdateAcknowledge_t *GNBDUConfigurationUpdateAcknowledge) {
  AssertFatal(1==0,"Not implemented yet\n");
}



/*
    gNB-CU Configuration Update
*/

//void CU_send_gNB_CU_CONFIGURATION_UPDATE(F1AP_GNBCUConfigurationUpdate_t *GNBCUConfigurationUpdate) {
int CU_send_gNB_CU_CONFIGURATION_UPDATE(instance_t instance, module_id_t du_mod_idP) {
  F1AP_F1AP_PDU_t                    pdu;
  F1AP_GNBCUConfigurationUpdate_t    *out;
  F1AP_GNBCUConfigurationUpdateIEs_t *ie;

  uint8_t  *buffer;
  uint32_t  len;
  int       i = 0;

  // for test
  int mcc = 208;
  int mnc = 93;
  int mnc_digit_length = 8;

  /* Create */
  /* 0. Message Type */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = F1AP_F1AP_PDU_PR_initiatingMessage;
  pdu.choice.initiatingMessage = (F1AP_InitiatingMessage_t *)calloc(1, sizeof(F1AP_InitiatingMessage_t));
  pdu.choice.initiatingMessage->procedureCode = F1AP_ProcedureCode_id_gNBCUConfigurationUpdate;
  pdu.choice.initiatingMessage->criticality   = F1AP_Criticality_ignore;
  pdu.choice.initiatingMessage->value.present = F1AP_InitiatingMessage__value_PR_GNBCUConfigurationUpdate;
  out = &pdu.choice.initiatingMessage->value.choice.GNBCUConfigurationUpdate;

  /* mandatory */
  /* c1. Transaction ID (integer value) */
  ie = (F1AP_GNBCUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBCUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_TransactionID;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBCUConfigurationUpdateIEs__value_PR_TransactionID;
  ie->value.choice.TransactionID = F1AP_get_next_transaction_identifier(instance, du_mod_idP);
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);



  /* mandatory */
  /* c2. Cells_to_be_Activated_List */
  ie = (F1AP_GNBCUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBCUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_Cells_to_be_Activated_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBCUConfigurationUpdateIEs__value_PR_Cells_to_be_Activated_List;

  for (i=0;
       i<1;
       i++) {

       F1AP_Cells_to_be_Activated_List_ItemIEs_t *cells_to_be_activated_list_item_ies;
       cells_to_be_activated_list_item_ies = (F1AP_Cells_to_be_Activated_List_ItemIEs_t *)calloc(1, sizeof(F1AP_Cells_to_be_Activated_List_ItemIEs_t));
       cells_to_be_activated_list_item_ies->id = F1AP_ProtocolIE_ID_id_Cells_to_be_Activated_List_Item;
       cells_to_be_activated_list_item_ies->criticality = F1AP_Criticality_reject;
       cells_to_be_activated_list_item_ies->value.present = F1AP_Cells_to_be_Activated_List_ItemIEs__value_PR_Cells_to_be_Activated_List_Item;

     /* 2.1 cells to be Activated list item */
     F1AP_Cells_to_be_Activated_List_Item_t cells_to_be_activated_list_item;
     memset((void *)&cells_to_be_activated_list_item, 0, sizeof(F1AP_Cells_to_be_Activated_List_Item_t));

     /* - nRCGI */
     F1AP_NRCGI_t nRCGI;
     memset(&nRCGI, 0, sizeof(F1AP_NRCGI_t));
     MCC_MNC_TO_PLMNID(mcc, mnc, mnc_digit_length,
                                         &nRCGI.pLMN_Identity);
     NR_CELL_ID_TO_BIT_STRING(123456, &nRCGI.nRCellIdentity);
     cells_to_be_activated_list_item.nRCGI = nRCGI;

     /* optional */
     /* - nRPCI */
     if (0) {
       cells_to_be_activated_list_item.nRPCI = (F1AP_NRPCI_t *)calloc(1, sizeof(F1AP_NRPCI_t));
       *cells_to_be_activated_list_item.nRPCI = 321L;  // int 0..1007
     }

     /* optional */
     /* - gNB-CU System Information */
     //if (1) {

     //}
     /* ADD */
     cells_to_be_activated_list_item_ies->value.choice.Cells_to_be_Activated_List_Item = cells_to_be_activated_list_item;
     ASN_SEQUENCE_ADD(&ie->value.choice.Cells_to_be_Activated_List.list,
                      cells_to_be_activated_list_item_ies);
  }  
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);



  /* mandatory */
  /* c3. Cells_to_be_Deactivated_List */
  ie = (F1AP_GNBCUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBCUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_Cells_to_be_Deactivated_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBCUConfigurationUpdateIEs__value_PR_Cells_to_be_Deactivated_List;

  for (i=0;
       i<1;
       i++) {

       F1AP_Cells_to_be_Deactivated_List_ItemIEs_t *cells_to_be_deactivated_list_item_ies;
       cells_to_be_deactivated_list_item_ies = (F1AP_Cells_to_be_Deactivated_List_ItemIEs_t *)calloc(1, sizeof(F1AP_Cells_to_be_Deactivated_List_ItemIEs_t));
       cells_to_be_deactivated_list_item_ies->id = F1AP_ProtocolIE_ID_id_Cells_to_be_Activated_List_Item;
       cells_to_be_deactivated_list_item_ies->criticality = F1AP_Criticality_reject;
       cells_to_be_deactivated_list_item_ies->value.present = F1AP_Cells_to_be_Deactivated_List_ItemIEs__value_PR_Cells_to_be_Deactivated_List_Item;

       /* 3.1 cells to be Deactivated list item */
       F1AP_Cells_to_be_Deactivated_List_Item_t cells_to_be_deactivated_list_item;
       memset((void *)&cells_to_be_deactivated_list_item, 0, sizeof(F1AP_Cells_to_be_Deactivated_List_Item_t));

       /* - nRCGI */
       F1AP_NRCGI_t nRCGI;
       memset(&nRCGI, 0, sizeof(F1AP_NRCGI_t));
       MCC_MNC_TO_PLMNID(mcc, mnc, mnc_digit_length,
                                           &nRCGI.pLMN_Identity);
       NR_CELL_ID_TO_BIT_STRING(123456, &nRCGI.nRCellIdentity);
       cells_to_be_deactivated_list_item.nRCGI = nRCGI;

       //}
       /* ADD */
       cells_to_be_deactivated_list_item_ies->value.choice.Cells_to_be_Deactivated_List_Item = cells_to_be_deactivated_list_item;
       ASN_SEQUENCE_ADD(&ie->value.choice.Cells_to_be_Deactivated_List.list,
                        cells_to_be_deactivated_list_item_ies);
  }  
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);


  /* mandatory */
  /* c4. GNB_CU_TNL_Association_To_Add_List */
  ie = (F1AP_GNBCUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBCUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_GNB_CU_TNL_Association_To_Add_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBCUConfigurationUpdateIEs__value_PR_GNB_CU_TNL_Association_To_Add_List;

  for (i=0;
       i<1;
       i++) {

       F1AP_GNB_CU_TNL_Association_To_Add_ItemIEs_t *gnb_cu_tnl_association_to_add_item_ies;
       gnb_cu_tnl_association_to_add_item_ies = (F1AP_GNB_CU_TNL_Association_To_Add_ItemIEs_t *)calloc(1, sizeof(F1AP_GNB_CU_TNL_Association_To_Add_ItemIEs_t));
       gnb_cu_tnl_association_to_add_item_ies->id = F1AP_ProtocolIE_ID_id_GNB_CU_TNL_Association_To_Add_Item;
       gnb_cu_tnl_association_to_add_item_ies->criticality = F1AP_Criticality_reject;
       gnb_cu_tnl_association_to_add_item_ies->value.present = F1AP_GNB_CU_TNL_Association_To_Add_ItemIEs__value_PR_GNB_CU_TNL_Association_To_Add_Item;

       /* 4.1 GNB_CU_TNL_Association_To_Add_Item */
       F1AP_GNB_CU_TNL_Association_To_Add_Item_t gnb_cu_tnl_association_to_add_item;
       memset((void *)&gnb_cu_tnl_association_to_add_item, 0, sizeof(F1AP_GNB_CU_TNL_Association_To_Add_Item_t));


       /* 4.1.1 tNLAssociationTransportLayerAddress */
       F1AP_CP_TransportLayerAddress_t transportLayerAddress;
       memset((void *)&transportLayerAddress, 0, sizeof(F1AP_CP_TransportLayerAddress_t));
       transportLayerAddress.present = F1AP_CP_TransportLayerAddress_PR_endpoint_IP_address;
       TRANSPORT_LAYER_ADDRESS_IPv4_TO_BIT_STRING(1234, &transportLayerAddress.choice.endpoint_IP_address);
       
       // memset((void *)&transportLayerAddress, 0, sizeof(F1AP_CP_TransportLayerAddress_t));
       // transportLayerAddress.present = F1AP_CP_TransportLayerAddress_PR_endpoint_IP_address_and_port;
       // transportLayerAddress.choice.endpoint_IP_address_and_port = (F1AP_Endpoint_IP_address_and_port_t *)calloc(1, sizeof(F1AP_Endpoint_IP_address_and_port_t));
       // TRANSPORT_LAYER_ADDRESS_IPv4_TO_BIT_STRING(1234, &transportLayerAddress.choice.endpoint_IP_address_and_port.endpoint_IP_address);

       gnb_cu_tnl_association_to_add_item.tNLAssociationTransportLayerAddress = transportLayerAddress;

       /* 4.1.2 tNLAssociationUsage */
       gnb_cu_tnl_association_to_add_item.tNLAssociationUsage = F1AP_TNLAssociationUsage_non_ue;
       

       /* ADD */
       gnb_cu_tnl_association_to_add_item_ies->value.choice.GNB_CU_TNL_Association_To_Add_Item = gnb_cu_tnl_association_to_add_item;
       ASN_SEQUENCE_ADD(&ie->value.choice.GNB_CU_TNL_Association_To_Add_List.list,
                        gnb_cu_tnl_association_to_add_item_ies);
  }  
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);



  /* mandatory */
  /* c5. GNB_CU_TNL_Association_To_Remove_List */
  ie = (F1AP_GNBCUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBCUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_GNB_CU_TNL_Association_To_Remove_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBCUConfigurationUpdateIEs__value_PR_GNB_CU_TNL_Association_To_Remove_List;
  for (i=0;
       i<1;
       i++) {

       F1AP_GNB_CU_TNL_Association_To_Remove_ItemIEs_t *gnb_cu_tnl_association_to_remove_item_ies;
       gnb_cu_tnl_association_to_remove_item_ies = (F1AP_GNB_CU_TNL_Association_To_Remove_ItemIEs_t *)calloc(1, sizeof(F1AP_GNB_CU_TNL_Association_To_Remove_ItemIEs_t));
       gnb_cu_tnl_association_to_remove_item_ies->id = F1AP_ProtocolIE_ID_id_GNB_CU_TNL_Association_To_Remove_Item;
       gnb_cu_tnl_association_to_remove_item_ies->criticality = F1AP_Criticality_reject;
       gnb_cu_tnl_association_to_remove_item_ies->value.present = F1AP_GNB_CU_TNL_Association_To_Remove_ItemIEs__value_PR_GNB_CU_TNL_Association_To_Remove_Item;

       /* 4.1 GNB_CU_TNL_Association_To_Remove_Item */
       F1AP_GNB_CU_TNL_Association_To_Remove_Item_t gnb_cu_tnl_association_to_remove_item;
       memset((void *)&gnb_cu_tnl_association_to_remove_item, 0, sizeof(F1AP_GNB_CU_TNL_Association_To_Remove_Item_t));


       /* 4.1.1 tNLAssociationTransportLayerAddress */
       F1AP_CP_TransportLayerAddress_t transportLayerAddress;
       memset((void *)&transportLayerAddress, 0, sizeof(F1AP_CP_TransportLayerAddress_t));
       transportLayerAddress.present = F1AP_CP_TransportLayerAddress_PR_endpoint_IP_address;
       TRANSPORT_LAYER_ADDRESS_IPv4_TO_BIT_STRING(1234, &transportLayerAddress.choice.endpoint_IP_address);
       
       // memset((void *)&transportLayerAddress, 0, sizeof(F1AP_CP_TransportLayerAddress_t));
       // transportLayerAddress.present = F1AP_CP_TransportLayerAddress_PR_endpoint_IP_address_and_port;
       // transportLayerAddress.choice.endpoint_IP_address_and_port = (F1AP_Endpoint_IP_address_and_port_t *)calloc(1, sizeof(F1AP_Endpoint_IP_address_and_port_t));
       // TRANSPORT_LAYER_ADDRESS_IPv4_TO_BIT_STRING(1234, &transportLayerAddress.choice.endpoint_IP_address_and_port.endpoint_IP_address);

       gnb_cu_tnl_association_to_remove_item.tNLAssociationTransportLayerAddress = transportLayerAddress;
   

       /* ADD */
       gnb_cu_tnl_association_to_remove_item_ies->value.choice.GNB_CU_TNL_Association_To_Remove_Item = gnb_cu_tnl_association_to_remove_item;
       ASN_SEQUENCE_ADD(&ie->value.choice.GNB_CU_TNL_Association_To_Remove_List.list,
                        gnb_cu_tnl_association_to_remove_item_ies);
  }  
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);

  /* mandatory */
  /* c6. GNB_CU_TNL_Association_To_Update_List */
  ie = (F1AP_GNBCUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBCUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_GNB_CU_TNL_Association_To_Update_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBCUConfigurationUpdateIEs__value_PR_GNB_CU_TNL_Association_To_Update_List;
  for (i=0;
       i<1;
       i++) {

       F1AP_GNB_CU_TNL_Association_To_Update_ItemIEs_t *gnb_cu_tnl_association_to_update_item_ies;
       gnb_cu_tnl_association_to_update_item_ies = (F1AP_GNB_CU_TNL_Association_To_Update_ItemIEs_t *)calloc(1, sizeof(F1AP_GNB_CU_TNL_Association_To_Update_ItemIEs_t));
       gnb_cu_tnl_association_to_update_item_ies->id = F1AP_ProtocolIE_ID_id_GNB_CU_TNL_Association_To_Update_Item;
       gnb_cu_tnl_association_to_update_item_ies->criticality = F1AP_Criticality_reject;
       gnb_cu_tnl_association_to_update_item_ies->value.present = F1AP_GNB_CU_TNL_Association_To_Update_ItemIEs__value_PR_GNB_CU_TNL_Association_To_Update_Item;

       /* 4.1 GNB_CU_TNL_Association_To_Update_Item */
       F1AP_GNB_CU_TNL_Association_To_Update_Item_t gnb_cu_tnl_association_to_update_item;
       memset((void *)&gnb_cu_tnl_association_to_update_item, 0, sizeof(F1AP_GNB_CU_TNL_Association_To_Update_Item_t));


       /* 4.1.1 tNLAssociationTransportLayerAddress */
       F1AP_CP_TransportLayerAddress_t transportLayerAddress;
       memset((void *)&transportLayerAddress, 0, sizeof(F1AP_CP_TransportLayerAddress_t));
       transportLayerAddress.present = F1AP_CP_TransportLayerAddress_PR_endpoint_IP_address;
       TRANSPORT_LAYER_ADDRESS_IPv4_TO_BIT_STRING(1234, &transportLayerAddress.choice.endpoint_IP_address);
       
       // memset((void *)&transportLayerAddress, 0, sizeof(F1AP_CP_TransportLayerAddress_t));
       // transportLayerAddress.present = F1AP_CP_TransportLayerAddress_PR_endpoint_IP_address_and_port;
       // transportLayerAddress.choice.endpoint_IP_address_and_port = (F1AP_Endpoint_IP_address_and_port_t *)calloc(1, sizeof(F1AP_Endpoint_IP_address_and_port_t));
       // TRANSPORT_LAYER_ADDRESS_IPv4_TO_BIT_STRING(1234, &transportLayerAddress.choice.endpoint_IP_address_and_port.endpoint_IP_address);

       gnb_cu_tnl_association_to_update_item.tNLAssociationTransportLayerAddress = transportLayerAddress;
   

       /* 4.1.2 tNLAssociationUsage */
       if (1) {
         gnb_cu_tnl_association_to_update_item.tNLAssociationUsage = (F1AP_TNLAssociationUsage_t *)calloc(1, sizeof(F1AP_TNLAssociationUsage_t));
         *gnb_cu_tnl_association_to_update_item.tNLAssociationUsage = F1AP_TNLAssociationUsage_non_ue;
       }
       
       /* ADD */
       gnb_cu_tnl_association_to_update_item_ies->value.choice.GNB_CU_TNL_Association_To_Update_Item = gnb_cu_tnl_association_to_update_item;
       ASN_SEQUENCE_ADD(&ie->value.choice.GNB_CU_TNL_Association_To_Update_List.list,
                        gnb_cu_tnl_association_to_update_item_ies);
  }  
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);



  /* mandatory */
  /* c7. Cells_to_be_Barred_List */
  ie = (F1AP_GNBCUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBCUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_Cells_to_be_Barred_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBCUConfigurationUpdateIEs__value_PR_Cells_to_be_Barred_List;
  for (i=0;
       i<1;
       i++) {

       F1AP_Cells_to_be_Barred_ItemIEs_t *cells_to_be_barred_item_ies;
       cells_to_be_barred_item_ies = (F1AP_Cells_to_be_Barred_ItemIEs_t *)calloc(1, sizeof(F1AP_Cells_to_be_Barred_ItemIEs_t));
       cells_to_be_barred_item_ies->id = F1AP_ProtocolIE_ID_id_Cells_to_be_Activated_List_Item;
       cells_to_be_barred_item_ies->criticality = F1AP_Criticality_reject;
       cells_to_be_barred_item_ies->value.present = F1AP_Cells_to_be_Barred_ItemIEs__value_PR_Cells_to_be_Barred_Item;

       /* 7.1 cells to be Deactivated list item */
       F1AP_Cells_to_be_Barred_Item_t cells_to_be_barred_item;
       memset((void *)&cells_to_be_barred_item, 0, sizeof(F1AP_Cells_to_be_Barred_Item_t));

       /* - nRCGI */
       F1AP_NRCGI_t nRCGI;
       memset(&nRCGI,0,sizeof(F1AP_NRCGI_t));
       MCC_MNC_TO_PLMNID(mcc, mnc, mnc_digit_length,
                                           &nRCGI.pLMN_Identity);
       NR_CELL_ID_TO_BIT_STRING(123456, &nRCGI.nRCellIdentity);
       cells_to_be_barred_item.nRCGI = nRCGI;
       
       /* 7.2 cellBarred*/
       cells_to_be_barred_item.cellBarred = F1AP_CellBarred_not_barred;

       /* ADD */
       cells_to_be_barred_item_ies->value.choice.Cells_to_be_Barred_Item = cells_to_be_barred_item;
       ASN_SEQUENCE_ADD(&ie->value.choice.Cells_to_be_Barred_List.list,
                        cells_to_be_barred_item_ies);
  }
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);



  /* mandatory */
  /* c8. Protected_EUTRA_Resources_List */
  ie = (F1AP_GNBCUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBCUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_Protected_EUTRA_Resources_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBCUConfigurationUpdateIEs__value_PR_Protected_EUTRA_Resources_List;

  for (i=0;
       i<1;
       i++) {


       F1AP_Protected_EUTRA_Resources_ItemIEs_t *protected_eutra_resources_item_ies;

       /* 8.1 SpectrumSharingGroupID */
       protected_eutra_resources_item_ies = (F1AP_Protected_EUTRA_Resources_ItemIEs_t *)calloc(1, sizeof(F1AP_Protected_EUTRA_Resources_ItemIEs_t));
       protected_eutra_resources_item_ies->id = F1AP_ProtocolIE_ID_id_Protected_EUTRA_Resources_List;
       protected_eutra_resources_item_ies->criticality = F1AP_Criticality_reject;
       protected_eutra_resources_item_ies->value.present = F1AP_Protected_EUTRA_Resources_ItemIEs__value_PR_SpectrumSharingGroupID;
       protected_eutra_resources_item_ies->value.choice.SpectrumSharingGroupID = 1L;

       ASN_SEQUENCE_ADD(&ie->value.choice.Protected_EUTRA_Resources_List.list, protected_eutra_resources_item_ies);

       /* 8.2 ListofEUTRACellsinGNBDUCoordination */
       protected_eutra_resources_item_ies = (F1AP_Protected_EUTRA_Resources_ItemIEs_t *)calloc(1, sizeof(F1AP_Protected_EUTRA_Resources_ItemIEs_t));
       protected_eutra_resources_item_ies->id = F1AP_ProtocolIE_ID_id_Protected_EUTRA_Resources_List;
       protected_eutra_resources_item_ies->criticality = F1AP_Criticality_reject;
       protected_eutra_resources_item_ies->value.present = F1AP_Protected_EUTRA_Resources_ItemIEs__value_PR_ListofEUTRACellsinGNBDUCoordination;

       F1AP_Served_EUTRA_Cells_Information_t served_eutra_cells_information;
       memset((void *)&served_eutra_cells_information, 0, sizeof(F1AP_Served_EUTRA_Cells_Information_t));

       F1AP_EUTRA_Mode_Info_t eUTRA_Mode_Info;
       memset((void *)&eUTRA_Mode_Info, 0, sizeof(F1AP_EUTRA_Mode_Info_t));

       // eUTRAFDD
       eUTRA_Mode_Info.present = F1AP_EUTRA_Mode_Info_PR_eUTRAFDD;
       F1AP_EUTRA_FDD_Info_t *eutra_fdd_info;
       eutra_fdd_info = (F1AP_EUTRA_FDD_Info_t *)calloc(1, sizeof(F1AP_EUTRA_FDD_Info_t));
       eutra_fdd_info->uL_offsetToPointA = 123L;
       eutra_fdd_info->dL_offsetToPointA = 456L;
       eUTRA_Mode_Info.choice.eUTRAFDD = eutra_fdd_info;

       // eUTRATDD
       // eUTRA_Mode_Info.present = F1AP_EUTRA_Mode_Info_PR_eUTRATDD;
       // F1AP_EUTRA_TDD_Info_t *eutra_tdd_info;
       // eutra_tdd_info = (F1AP_EUTRA_TDD_Info_t *)calloc(1, sizeof(F1AP_EUTRA_TDD_Info_t));
       // eutra_tdd_info->uL_offsetToPointA = 123L;
       // eutra_tdd_info->dL_offsetToPointA = 456L;
       // eUTRA_Mode_Info.choice.eUTRATDD = eutra_tdd_info;

       served_eutra_cells_information.eUTRA_Mode_Info = eUTRA_Mode_Info;

       OCTET_STRING_fromBuf(&served_eutra_cells_information.protectedEUTRAResourceIndication, "asdsa1d32sa1d31asd31as",
                       strlen("asdsa1d32sa1d31asd31as"));

       ASN_SEQUENCE_ADD(&protected_eutra_resources_item_ies->value.choice.ListofEUTRACellsinGNBDUCoordination.list, &served_eutra_cells_information);

       ASN_SEQUENCE_ADD(&ie->value.choice.Protected_EUTRA_Resources_List.list, protected_eutra_resources_item_ies);
  }
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);


  /* encode */
  if (f1ap_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(F1AP, "Failed to encode F1 setup request\n");
    return -1;
  }

  cu_f1ap_itti_send_sctp_data_req(instance, f1ap_du_data_from_du->assoc_id, buffer, len, 0);
  return 0;
}

int CU_handle_gNB_CU_CONFIGURATION_UPDATE_FAILURE(instance_t instance,
                                                   uint32_t assoc_id,
                                                   uint32_t stream,
                                                   F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int CU_handle_gNB_CU_CONFIGURATION_UPDATE_ACKNOWLEDGE(instance_t instance,
                                                       uint32_t assoc_id,
                                                       uint32_t stream,
                                                       F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}


int CU_handle_gNB_DU_RESOURCE_COORDINATION_REQUEST(instance_t instance,
                                                    uint32_t assoc_id,
                                                    uint32_t stream,
                                                    F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(0, "Not implemented yet\n");
}

int CU_send_gNB_DU_RESOURCE_COORDINATION_RESPONSE(instance_t instance,
                    F1AP_GNBDUResourceCoordinationResponse_t *GNBDUResourceCoordinationResponse) {
  AssertFatal(0, "Not implemented yet\n");
}
