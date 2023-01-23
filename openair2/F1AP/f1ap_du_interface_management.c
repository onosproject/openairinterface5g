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

/*! \file f1ap_du_interface_management.c
 * \brief f1ap interface management for DU
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
#include "f1ap_du_interface_management.h"
#include "assertions.h"

extern f1ap_setup_req_t *f1ap_du_data;


int DU_handle_RESET(instance_t instance,
                                uint32_t assoc_id,
                                uint32_t stream,
                                F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int DU_send_RESET_ACKKNOWLEDGE(instance_t instance, F1AP_ResetAcknowledge_t *ResetAcknowledge) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int DU_send_RESET(instance_t instance, F1AP_Reset_t *Reset) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int DU_handle_RESET_ACKNOWLEDGE(instance_t instance,
                                uint32_t assoc_id,
                                uint32_t stream,
                                F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}


/*
    Error Indication
*/

int DU_send_ERROR_INDICATION(instance_t instance, F1AP_F1AP_PDU_t *pdu_p) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int DU_handle_ERROR_INDICATION(instance_t instance,
                                uint32_t assoc_id,
                                uint32_t stream,
                                F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}


/*
    F1 Setup
*/

void *g_f1SetupReq;
uint32_t g_f1SetupReqSize;

// SETUP REQUEST
int DU_send_F1_SETUP_REQUEST(instance_t instance) 
{
  module_id_t enb_mod_idP=0;
  module_id_t du_mod_idP=0;

  F1AP_F1AP_PDU_t          pdu; 
  F1AP_F1SetupRequest_t    *out;
  F1AP_F1SetupRequestIEs_t *ie;

  uint8_t  *buffer;
  uint32_t  len;
  int       i = 0;
  int       j = 0;

  /* Create */
  //g_f1SetupReq = malloc(sizeof(F1AP_F1AP_PDU_t));
  //memset(g_f1SetupReq, 0, sizeof(F1AP_F1AP_PDU_t));
  //g_f1SetupReqSize = sizeof(F1AP_F1AP_PDU_t);

  /* 0. pdu Type */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = F1AP_F1AP_PDU_PR_initiatingMessage;
  pdu.choice.initiatingMessage = (F1AP_InitiatingMessage_t *)calloc(1, sizeof(F1AP_InitiatingMessage_t));
  pdu.choice.initiatingMessage->procedureCode = F1AP_ProcedureCode_id_F1Setup;
  pdu.choice.initiatingMessage->criticality   = F1AP_Criticality_reject;
  pdu.choice.initiatingMessage->value.present = F1AP_InitiatingMessage__value_PR_F1SetupRequest;
  out = &pdu.choice.initiatingMessage->value.choice.F1SetupRequest;

  /* mandatory */
  /* c1. Transaction ID (integer value) */
  ie = (F1AP_F1SetupRequestIEs_t *)calloc(1, sizeof(F1AP_F1SetupRequestIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_TransactionID;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_F1SetupRequestIEs__value_PR_TransactionID;
  ie->value.choice.TransactionID = F1AP_get_next_transaction_identifier(enb_mod_idP, du_mod_idP);
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);

  /* mandatory */
  /* c2. GNB_DU_ID (integer value) */
  ie = (F1AP_F1SetupRequestIEs_t *)calloc(1, sizeof(F1AP_F1SetupRequestIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_gNB_DU_ID;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_F1SetupRequestIEs__value_PR_GNB_DU_ID;
  asn_int642INTEGER(&ie->value.choice.GNB_DU_ID, f1ap_du_data->gNB_DU_id);
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);

  /* optional */
  /* c3. GNB_DU_Name */
  if (f1ap_du_data->gNB_DU_name != NULL) {
    ie = (F1AP_F1SetupRequestIEs_t *)calloc(1, sizeof(F1AP_F1SetupRequestIEs_t));
    ie->id                        = F1AP_ProtocolIE_ID_id_gNB_DU_Name;
    ie->criticality               = F1AP_Criticality_ignore;
    ie->value.present             = F1AP_F1SetupRequestIEs__value_PR_GNB_DU_Name;
    OCTET_STRING_fromBuf(&ie->value.choice.GNB_DU_Name, f1ap_du_data->gNB_DU_name,
                         strlen(f1ap_du_data->gNB_DU_name));
    ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);
  }

  /* mandatory */
  /* c4. served cells list */
  ie = (F1AP_F1SetupRequestIEs_t *)calloc(1, sizeof(F1AP_F1SetupRequestIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_gNB_DU_Served_Cells_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_F1SetupRequestIEs__value_PR_GNB_DU_Served_Cells_List;

  int num_cells_available = f1ap_du_data->num_cells_available;
  LOG_D(F1AP, "num_cells_available = %d \n", num_cells_available);
  for (i=0;
       i<num_cells_available;
       i++) {
        /* mandatory */
        /* 4.1 served cells item */

        F1AP_GNB_DU_Served_Cells_ItemIEs_t *gnb_du_served_cell_list_item_ies;
        gnb_du_served_cell_list_item_ies = (F1AP_GNB_DU_Served_Cells_ItemIEs_t *)calloc(1, sizeof(F1AP_GNB_DU_Served_Cells_ItemIEs_t));
        gnb_du_served_cell_list_item_ies->id = F1AP_ProtocolIE_ID_id_GNB_DU_Served_Cells_Item;
        gnb_du_served_cell_list_item_ies->criticality = F1AP_Criticality_reject;
        gnb_du_served_cell_list_item_ies->value.present = F1AP_GNB_DU_Served_Cells_ItemIEs__value_PR_GNB_DU_Served_Cells_Item;
        

        F1AP_GNB_DU_Served_Cells_Item_t gnb_du_served_cells_item;
        memset((void *)&gnb_du_served_cells_item, 0, sizeof(F1AP_GNB_DU_Served_Cells_Item_t));

        /* 4.1.1 served cell Information */
        F1AP_Served_Cell_Information_t served_cell_information;

        memset((void *)&served_cell_information, 0, sizeof(F1AP_Served_Cell_Information_t));

        /* - nRCGI */
        F1AP_NRCGI_t nRCGI;
        memset(&nRCGI, 0, sizeof(F1AP_NRCGI_t));
        MCC_MNC_TO_PLMNID(f1ap_du_data->mcc[i], f1ap_du_data->mnc[i], f1ap_du_data->mnc_digit_length[i], &nRCGI.pLMN_Identity);
        LOG_I(F1AP, "plmn: (%d,%d)\n",f1ap_du_data->mcc[i],f1ap_du_data->mnc[i]);
        //MCC_MNC_TO_PLMNID(208, 95, 2, &nRCGI.pLMN_Identity);

        NR_CELL_ID_TO_BIT_STRING(f1ap_du_data->nr_cellid[i], &nRCGI.nRCellIdentity);
        LOG_I(F1AP, "nRCellIdentity (%llx): %x.%x.%x.%x.%x\n",
              (long long unsigned int)f1ap_du_data->nr_cellid[i],
              nRCGI.nRCellIdentity.buf[0],
              nRCGI.nRCellIdentity.buf[1],
              nRCGI.nRCellIdentity.buf[2],
              nRCGI.nRCellIdentity.buf[3],
              nRCGI.nRCellIdentity.buf[4]);

        served_cell_information.nRCGI = nRCGI;

        /* - nRPCI */
        served_cell_information.nRPCI = f1ap_du_data->nr_pci[i];  // int 0..1007

        /* - fiveGS_TAC */
        OCTET_STRING_fromBuf(&served_cell_information.fiveGS_TAC,
                             (const char*)&f1ap_du_data->tac[i],
                             3);

        /* - Configured_EPS_TAC */
        if(0){
          served_cell_information.configured_EPS_TAC = (F1AP_Configured_EPS_TAC_t *)calloc(1, sizeof(F1AP_Configured_EPS_TAC_t));
          OCTET_STRING_fromBuf(served_cell_information.configured_EPS_TAC,
                             "2",
                             2);
        }

        /* - broadcast PLMNs */
        // RK: add the num_available_broadcast_PLMNs to the message 
        int num_available_broadcast_PLMNs = 1; //f1ap_du_data->num_available_broadcast_PLMNs;
        LOG_D(F1AP, "num_available_broadcast_PLMNs = %d \n", num_available_broadcast_PLMNs);
        for (j=0;
            j<num_available_broadcast_PLMNs;    // num_available_broadcast_PLMNs
            j++) {
            /* > PLMN BroadcastPLMNs Item */
            F1AP_BroadcastPLMNs_Item_t *broadcastPLMNs_Item = (F1AP_BroadcastPLMNs_Item_t *)calloc(1, sizeof(F1AP_BroadcastPLMNs_Item_t));
            //MCC_MNC_TO_PLMNID(208, 95, 2, &broadcastPLMNs_Item->pLMN_Identity);
            MCC_MNC_TO_PLMNID(f1ap_du_data->mcc[i], f1ap_du_data->mnc[i], f1ap_du_data->mnc_digit_length[i], &broadcastPLMNs_Item->pLMN_Identity);
            ASN_SEQUENCE_ADD(&served_cell_information.servedPLMNs.list, broadcastPLMNs_Item);
        }

        // // /* - CHOICE NR-MODE-Info */
        F1AP_NR_Mode_Info_t nR_Mode_Info;
        //f1ap_du_data->fdd_flag = 1;
        if (f1ap_du_data->fdd_flag) { // FDD
          nR_Mode_Info.present = F1AP_NR_Mode_Info_PR_fDD;
          F1AP_FDD_Info_t *fDD_Info = (F1AP_FDD_Info_t *)calloc(1, sizeof(F1AP_FDD_Info_t));

          /* FDD.1 UL NRFreqInfo */
            /* FDD.1.1 UL NRFreqInfo ARFCN */
            fDD_Info->uL_NRFreqInfo.nRARFCN = f1ap_du_data->nr_mode_info[i].fdd.ul_nr_arfcn; // Integer

            /* FDD.1.2 F1AP_SUL_Information */
            if(0) { // Optional
              F1AP_SUL_Information_t *fdd_sul_info = (F1AP_SUL_Information_t *)calloc(1, sizeof(F1AP_SUL_Information_t));
              fdd_sul_info->sUL_NRARFCN = 0;
              fdd_sul_info->sUL_transmission_Bandwidth.nRSCS = 0;
              fdd_sul_info->sUL_transmission_Bandwidth.nRNRB = 0;
              fDD_Info->uL_NRFreqInfo.sul_Information = fdd_sul_info;
            }

            /* FDD.1.3 freqBandListNr */
            int fdd_ul_num_available_freq_Bands = f1ap_du_data->nr_mode_info[i].fdd.ul_num_frequency_bands;
            LOG_D(F1AP, "fdd_ul_num_available_freq_Bands = %d \n", fdd_ul_num_available_freq_Bands);
            int fdd_ul_j;
            for (fdd_ul_j=0;
                 fdd_ul_j<fdd_ul_num_available_freq_Bands;
                 fdd_ul_j++) {

                  F1AP_FreqBandNrItem_t nr_freqBandNrItem;
                  memset((void *)&nr_freqBandNrItem, 0, sizeof(F1AP_FreqBandNrItem_t));
                  /* FDD.1.3.1 freqBandIndicatorNr*/
                  nr_freqBandNrItem.freqBandIndicatorNr = f1ap_du_data->nr_mode_info[i].fdd.ul_nr_band[fdd_ul_j]; //

                  /* FDD.1.3.2 supportedSULBandList*/
                  int num_available_supported_SULBands = f1ap_du_data->nr_mode_info[i].fdd.ul_num_sul_frequency_bands;
                  LOG_D(F1AP, "num_available_supported_SULBands = %d \n", num_available_supported_SULBands);
                  int fdd_ul_k;
                  for (fdd_ul_k=0;
                       fdd_ul_k<num_available_supported_SULBands;
                       fdd_ul_k++) {
                        F1AP_SupportedSULFreqBandItem_t nr_supportedSULFreqBandItem;
                        memset((void *)&nr_supportedSULFreqBandItem, 0, sizeof(F1AP_SupportedSULFreqBandItem_t));
                          /* FDD.1.3.2.1 freqBandIndicatorNr */
                          nr_supportedSULFreqBandItem.freqBandIndicatorNr = f1ap_du_data->nr_mode_info[i].fdd.ul_nr_sul_band[fdd_ul_k]; //
                        ASN_SEQUENCE_ADD(&nr_freqBandNrItem.supportedSULBandList.list, &nr_supportedSULFreqBandItem);
                  } // for FDD : UL supported_SULBands
                  ASN_SEQUENCE_ADD(&fDD_Info->uL_NRFreqInfo.freqBandListNr.list, &nr_freqBandNrItem);
            } // for FDD : UL freq_Bands
           
          /* FDD.2 DL NRFreqInfo */
            /* FDD.2.1 DL NRFreqInfo ARFCN */
            fDD_Info->dL_NRFreqInfo.nRARFCN = f1ap_du_data->nr_mode_info[i].fdd.dl_nr_arfcn; // Integer

            /* FDD.2.2 F1AP_SUL_Information */
            if(0) { // Optional
              F1AP_SUL_Information_t *fdd_sul_info = (F1AP_SUL_Information_t *)calloc(1, sizeof(F1AP_SUL_Information_t));
              fdd_sul_info->sUL_NRARFCN = 0;
              fdd_sul_info->sUL_transmission_Bandwidth.nRSCS = 0;
              fdd_sul_info->sUL_transmission_Bandwidth.nRNRB = 0;
              fDD_Info->dL_NRFreqInfo.sul_Information = fdd_sul_info;
            }

            /* FDD.2.3 freqBandListNr */
            int fdd_dl_num_available_freq_Bands = f1ap_du_data->nr_mode_info[i].fdd.dl_num_frequency_bands;
            LOG_D(F1AP, "fdd_dl_num_available_freq_Bands = %d \n", fdd_dl_num_available_freq_Bands);
            int fdd_dl_j;
            for (fdd_dl_j=0;
                 fdd_dl_j<fdd_dl_num_available_freq_Bands;
                 fdd_dl_j++) {

                  F1AP_FreqBandNrItem_t nr_freqBandNrItem;
                  memset((void *)&nr_freqBandNrItem, 0, sizeof(F1AP_FreqBandNrItem_t));
                  /* FDD.2.3.1 freqBandIndicatorNr*/
                  nr_freqBandNrItem.freqBandIndicatorNr = f1ap_du_data->nr_mode_info[i].fdd.dl_nr_band[fdd_dl_j]; //

                  /* FDD.2.3.2 supportedSULBandList*/
                  int num_available_supported_SULBands = f1ap_du_data->nr_mode_info[i].fdd.dl_num_sul_frequency_bands;
                  LOG_D(F1AP, "num_available_supported_SULBands = %d \n", num_available_supported_SULBands);
                  int fdd_dl_k;
                  for (fdd_dl_k=0;
                       fdd_dl_k<num_available_supported_SULBands;
                       fdd_dl_k++) {
                        F1AP_SupportedSULFreqBandItem_t nr_supportedSULFreqBandItem;
                        memset((void *)&nr_supportedSULFreqBandItem, 0, sizeof(F1AP_SupportedSULFreqBandItem_t));
                          /* FDD.2.3.2.1 freqBandIndicatorNr */
                          nr_supportedSULFreqBandItem.freqBandIndicatorNr = f1ap_du_data->nr_mode_info[i].fdd.dl_nr_sul_band[fdd_dl_k]; //
                        ASN_SEQUENCE_ADD(&nr_freqBandNrItem.supportedSULBandList.list, &nr_supportedSULFreqBandItem);
                  } // for FDD : DL supported_SULBands
                  ASN_SEQUENCE_ADD(&fDD_Info->dL_NRFreqInfo.freqBandListNr.list, &nr_freqBandNrItem);
            } // for FDD : DL freq_Bands

          /* FDD.3 UL Transmission Bandwidth */
          fDD_Info->uL_Transmission_Bandwidth.nRSCS = f1ap_du_data->nr_mode_info[i].fdd.ul_scs;
          fDD_Info->uL_Transmission_Bandwidth.nRNRB = f1ap_du_data->nr_mode_info[i].fdd.ul_nrb;
          /* FDD.4 DL Transmission Bandwidth */
          fDD_Info->dL_Transmission_Bandwidth.nRSCS = f1ap_du_data->nr_mode_info[i].fdd.dl_scs;
          fDD_Info->dL_Transmission_Bandwidth.nRNRB = f1ap_du_data->nr_mode_info[i].fdd.dl_nrb;
          
          nR_Mode_Info.choice.fDD = fDD_Info;
        } else { // TDD
          nR_Mode_Info.present = F1AP_NR_Mode_Info_PR_tDD;
          F1AP_TDD_Info_t *tDD_Info = (F1AP_TDD_Info_t *)calloc(1, sizeof(F1AP_TDD_Info_t));

          /* TDD.1 nRFreqInfo */
            /* TDD.1.1 nRFreqInfo ARFCN */
            tDD_Info->nRFreqInfo.nRARFCN = f1ap_du_data->nr_mode_info[i].tdd.nr_arfcn; // Integer

            /* TDD.1.2 F1AP_SUL_Information */
            if(0) { // Optional
              F1AP_SUL_Information_t *tdd_sul_info = (F1AP_SUL_Information_t *)calloc(1, sizeof(F1AP_SUL_Information_t));
              tdd_sul_info->sUL_NRARFCN = 0;
              tdd_sul_info->sUL_transmission_Bandwidth.nRSCS = 0;
              tdd_sul_info->sUL_transmission_Bandwidth.nRNRB = 0;
              tDD_Info->nRFreqInfo.sul_Information = tdd_sul_info;
            }

            /* TDD.1.3 freqBandListNr */
            int tdd_num_available_freq_Bands = f1ap_du_data->nr_mode_info[i].tdd.num_frequency_bands;
            LOG_D(F1AP, "tdd_num_available_freq_Bands = %d \n", tdd_num_available_freq_Bands);
            int j;
            for (j=0;
                 j<tdd_num_available_freq_Bands;
                 j++) {

                  F1AP_FreqBandNrItem_t nr_freqBandNrItem;
                  memset((void *)&nr_freqBandNrItem, 0, sizeof(F1AP_FreqBandNrItem_t));
                  /* TDD.1.3.1 freqBandIndicatorNr*/
                  nr_freqBandNrItem.freqBandIndicatorNr = *f1ap_du_data->nr_mode_info[i].tdd.nr_band; //

                  /* TDD.1.3.2 supportedSULBandList*/
                  int num_available_supported_SULBands = f1ap_du_data->nr_mode_info[i].tdd.num_sul_frequency_bands;
                  LOG_D(F1AP, "num_available_supported_SULBands = %d \n", num_available_supported_SULBands);
                  int k;
                  for (k=0;
                       k<num_available_supported_SULBands;
                       k++) {
                        F1AP_SupportedSULFreqBandItem_t nr_supportedSULFreqBandItem;
                        memset((void *)&nr_supportedSULFreqBandItem, 0, sizeof(F1AP_SupportedSULFreqBandItem_t));
                          /* TDD.1.3.2.1 freqBandIndicatorNr */
                          nr_supportedSULFreqBandItem.freqBandIndicatorNr = *f1ap_du_data->nr_mode_info[i].tdd.nr_sul_band; //
                        ASN_SEQUENCE_ADD(&nr_freqBandNrItem.supportedSULBandList.list, &nr_supportedSULFreqBandItem);
                  } // for TDD : supported_SULBands
                  ASN_SEQUENCE_ADD(&tDD_Info->nRFreqInfo.freqBandListNr.list, &nr_freqBandNrItem);
            } // for TDD : freq_Bands

          /* TDD.2 transmission_Bandwidth */
          tDD_Info->transmission_Bandwidth.nRSCS = f1ap_du_data->nr_mode_info[i].tdd.scs;
          tDD_Info->transmission_Bandwidth.nRNRB = f1ap_du_data->nr_mode_info[i].tdd.nrb;
     
          nR_Mode_Info.choice.tDD = tDD_Info;
        } // if nR_Mode_Info
        
        served_cell_information.nR_Mode_Info = nR_Mode_Info;

        /* - measurementTimingConfiguration */
        char *measurementTimingConfiguration = f1ap_du_data->measurement_timing_information[i]; // sept. 2018

        OCTET_STRING_fromBuf(&served_cell_information.measurementTimingConfiguration,
                             measurementTimingConfiguration,
                             strlen(measurementTimingConfiguration));
        gnb_du_served_cells_item.served_Cell_Information = served_cell_information; //

        /* 4.1.2 gNB-DU System Information */
        F1AP_GNB_DU_System_Information_t *gNB_DU_System_Information = (F1AP_GNB_DU_System_Information_t *)calloc(1, sizeof(F1AP_GNB_DU_System_Information_t));

        OCTET_STRING_fromBuf(&gNB_DU_System_Information->mIB_message,  // sept. 2018
                             (const char*)f1ap_du_data->mib[i],//f1ap_du_data->mib,
                             f1ap_du_data->mib_length[i]);

        OCTET_STRING_fromBuf(&gNB_DU_System_Information->sIB1_message,  // sept. 2018
                             (const char*)f1ap_du_data->sib1[i],
                             f1ap_du_data->sib1_length[i]);

        gnb_du_served_cells_item.gNB_DU_System_Information = gNB_DU_System_Information; //

        /* ADD */
        gnb_du_served_cell_list_item_ies->value.choice.GNB_DU_Served_Cells_Item = gnb_du_served_cells_item;

        ASN_SEQUENCE_ADD(&ie->value.choice.GNB_DU_Served_Cells_List.list, 
                        gnb_du_served_cell_list_item_ies);

  }
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);

#if 1
  /* RRC Version */
  /* gNB-DU RRC version  (BIT STRING) */
  ie = (F1AP_F1SetupRequestIEs_t *)calloc(1, sizeof(F1AP_F1SetupRequestIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_GNB_DU_RRC_Version;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_F1SetupRequestIEs__value_PR_RRC_Version;
  //asn_int642INTEGER(&ie->value.choice.GNB_DU_ID, f1ap_du_data->gNB_DU_id);
  ie->value.choice.RRC_Version.latest_RRC_Version.buf = calloc(1, sizeof(uint8_t));
  ie->value.choice.RRC_Version.latest_RRC_Version.buf[0] = 0;
  //ie->value.choice.RRC_Version.latest_RRC_Version.buf[1] = 0;
  //ie->value.choice.RRC_Version.latest_RRC_Version.buf[2] = 0;
  ie->value.choice.RRC_Version.latest_RRC_Version.size = 1;
  ie->value.choice.RRC_Version.latest_RRC_Version.bits_unused = 5;

  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);
#endif

  /*copying to global buffer for E2Cell Info to be sent during E2SetupReq */
  //memcpy(g_f1SetupReq, &pdu, sizeof(F1AP_F1AP_PDU_t));
  //LOG_I(F1AP, "++ F1SetupReq for E2Cell Info copied PCI:%u ++\n",f1ap_du_data->nr_pci[0]);

  /* encode */
  if (f1ap_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(F1AP, "Failed to encode F1 setup request\n");
    return -1;
  }

  /* Create */
  g_f1SetupReq = malloc(len);
  memset(g_f1SetupReq, 0, len);
  g_f1SetupReqSize = len;
  
  /*copying to global buffer for E2Cell Info to be sent during E2SetupReq */
  memcpy(g_f1SetupReq, buffer, len);
  LOG_I(F1AP, "++ F1SetupReq for E2Cell Info copied PCI:%u ++\n",f1ap_du_data->nr_pci[0]);
  
  MSC_LOG_TX_MESSAGE(
  MSC_F1AP_DU,
  MSC_F1AP_CU,
  (const char *)buffer,
  len,
  MSC_AS_TIME_FMT" F1_SETUP_REQUEST initiatingMessage gNB_DU_name %s",
  0,0,//MSC_AS_TIME_ARGS(ctxt_pP),
  f1ap_du_data->gNB_DU_name);

  du_f1ap_itti_send_sctp_data_req(instance, f1ap_du_data->assoc_id, buffer, len, 0);

  return 0;
}

void *g_f1SetupResp;
uint32_t g_f1SetupRespSize;
int DU_handle_F1_SETUP_RESPONSE(instance_t instance,
				uint32_t               assoc_id,
				uint32_t               stream,
				F1AP_F1AP_PDU_t       *pdu)
{

   LOG_D(F1AP, "DU_handle_F1_SETUP_RESPONSE\n");

   AssertFatal(pdu->present == F1AP_F1AP_PDU_PR_successfulOutcome,
	       "pdu->present != F1AP_F1AP_PDU_PR_successfulOutcome\n");
   AssertFatal(pdu->choice.successfulOutcome->procedureCode  == F1AP_ProcedureCode_id_F1Setup,
	       "pdu->choice.successfulOutcome->procedureCode != F1AP_ProcedureCode_id_F1Setup\n");
   AssertFatal(pdu->choice.successfulOutcome->criticality  == F1AP_Criticality_reject,
	       "pdu->choice.successfulOutcome->criticality != F1AP_Criticality_reject\n");
   AssertFatal(pdu->choice.successfulOutcome->value.present  == F1AP_SuccessfulOutcome__value_PR_F1SetupResponse,
	       "pdu->choice.successfulOutcome->value.present != F1AP_SuccessfulOutcome__value_PR_F1SetupResponse\n");

   F1AP_F1SetupResponse_t    *in = &pdu->choice.successfulOutcome->value.choice.F1SetupResponse;

   g_f1SetupResp = malloc(sizeof(F1AP_F1SetupResponse_t));
   memset(g_f1SetupResp, 0, sizeof(F1AP_F1SetupResponse_t));
   g_f1SetupRespSize = sizeof(F1AP_F1SetupResponse_t);

   /*copying to global buffer for E2Cell Info to be sent during E2SetupResp */
   memcpy(g_f1SetupResp, in, sizeof(F1AP_F1SetupResponse_t));
   LOG_I(F1AP, "++ F1SetupResp for E2Cell Info copied ++\n");

   F1AP_F1SetupResponseIEs_t *ie;
   int TransactionId = -1;
   int num_cells_to_activate = 0;
   F1AP_Cells_to_be_Activated_List_Item_t *cell;

   MessageDef *msg_p = itti_alloc_new_message (TASK_DU_F1, F1AP_SETUP_RESP);

   LOG_D(F1AP, "F1AP: F1Setup-Resp: protocolIEs.list.count %d\n",
         in->protocolIEs.list.count);
   for (int i=0;i < in->protocolIEs.list.count; i++) {
     ie = in->protocolIEs.list.array[i];
     switch (ie->id) {
     case F1AP_ProtocolIE_ID_id_TransactionID:
       AssertFatal(ie->criticality == F1AP_Criticality_reject,
		   "ie->criticality != F1AP_Criticality_reject\n");
       AssertFatal(ie->value.present == F1AP_F1SetupResponseIEs__value_PR_TransactionID,
		   "ie->value.present != F1AP_F1SetupResponseIEs__value_PR_TransactionID\n");
       TransactionId=ie->value.choice.TransactionID;
       LOG_D(F1AP, "F1AP: F1Setup-Resp: TransactionId %d\n",
             TransactionId);
       break;
     case F1AP_ProtocolIE_ID_id_gNB_CU_Name:
       AssertFatal(ie->criticality == F1AP_Criticality_ignore,
		   "ie->criticality != F1AP_Criticality_ignore\n");
       AssertFatal(ie->value.present == F1AP_F1SetupResponseIEs__value_PR_GNB_CU_Name,
		   "ie->value.present != F1AP_F1SetupResponseIEs__value_PR_TransactionID\n");
       F1AP_SETUP_RESP (msg_p).gNB_CU_name = malloc(ie->value.choice.GNB_CU_Name.size+1);
       memcpy(F1AP_SETUP_RESP (msg_p).gNB_CU_name,ie->value.choice.GNB_CU_Name.buf,ie->value.choice.GNB_CU_Name.size);
       F1AP_SETUP_RESP (msg_p).gNB_CU_name[ie->value.choice.GNB_CU_Name.size]='\0';
       LOG_D(F1AP, "F1AP: F1Setup-Resp: gNB_CU_name %s\n",
             F1AP_SETUP_RESP (msg_p).gNB_CU_name);
       break;
     case F1AP_ProtocolIE_ID_id_Cells_to_be_Activated_List:
       AssertFatal(ie->criticality == F1AP_Criticality_reject,
		   "ie->criticality != F1AP_Criticality_reject\n");
       AssertFatal(ie->value.present == F1AP_F1SetupResponseIEs__value_PR_Cells_to_be_Activated_List,
		   "ie->value.present != F1AP_F1SetupResponseIEs__value_PR_Cells_to_be_Activated_List\n");
       num_cells_to_activate = ie->value.choice.Cells_to_be_Activated_List.list.count;
       LOG_D(F1AP, "F1AP: Activating %d cells\n",num_cells_to_activate);
       for (int i=0;i<num_cells_to_activate;i++) {
	 
	 F1AP_Cells_to_be_Activated_List_ItemIEs_t *cells_to_be_activated_list_item_ies = (F1AP_Cells_to_be_Activated_List_ItemIEs_t *) ie->value.choice.Cells_to_be_Activated_List.list.array[i];

	 AssertFatal(cells_to_be_activated_list_item_ies->id == F1AP_ProtocolIE_ID_id_Cells_to_be_Activated_List_Item,
		   "cells_to_be_activated_list_item_ies->id != F1AP_ProtocolIE_ID_id_Cells_to_be_Activated_List_Item");
	 AssertFatal(cells_to_be_activated_list_item_ies->criticality == F1AP_Criticality_reject,
		     "cells_to_be_activated_list_item_ies->criticality == F1AP_Criticality_reject");
	 AssertFatal(cells_to_be_activated_list_item_ies->value.present == F1AP_Cells_to_be_Activated_List_ItemIEs__value_PR_Cells_to_be_Activated_List_Item,
		     "cells_to_be_activated_list_item_ies->value.present == F1AP_Cells_to_be_Activated_List_ItemIEs__value_PR_Cells_to_be_Activated_List_Item");

	 cell = &cells_to_be_activated_list_item_ies->value.choice.Cells_to_be_Activated_List_Item;

	 TBCD_TO_MCC_MNC(&cell->nRCGI.pLMN_Identity, F1AP_SETUP_RESP (msg_p).mcc[i], F1AP_SETUP_RESP (msg_p).mnc[i], F1AP_SETUP_RESP (msg_p).mnc_digit_length[i]);
	 AssertFatal(cell->nRPCI != NULL, "nRPCI is null\n");
	 LOG_D(F1AP, "nr_cellId : %x %x %x %x %x\n",
         cell->nRCGI.nRCellIdentity.buf[0],
         cell->nRCGI.nRCellIdentity.buf[1],
         cell->nRCGI.nRCellIdentity.buf[2],
         cell->nRCGI.nRCellIdentity.buf[3],
         cell->nRCGI.nRCellIdentity.buf[4]);
	 BIT_STRING_TO_NR_CELL_IDENTITY(&cell->nRCGI.nRCellIdentity,
					F1AP_SETUP_RESP (msg_p).nr_cellid[i]);
	 F1AP_SETUP_RESP (msg_p).nrpci[i] = *cell->nRPCI;

	 F1AP_ProtocolExtensionContainer_160P9_t *ext = (F1AP_ProtocolExtensionContainer_160P9_t *)cell->iE_Extensions;
	 AssertFatal(ext!=NULL,"Extension for SI is null\n");
	 F1AP_SETUP_RESP (msg_p).num_SI[i] = ext->list.count;
	 AssertFatal(ext->list.count==1,"At least one SI message should be there, and only 1 for now!\n");
   LOG_D(F1AP, "F1AP: F1Setup-Resp Cell %d MCC %d MNC %d NRCellid %lx num_si %d\n",
         i, F1AP_SETUP_RESP (msg_p).mcc[i], F1AP_SETUP_RESP (msg_p).mnc[i],
         F1AP_SETUP_RESP (msg_p).nr_cellid[i], F1AP_SETUP_RESP (msg_p).num_SI[i]);
	 for (int si =0;si < ext->list.count;si++) {
	   size_t size = ext->list.array[si]->extensionValue.choice.GNB_CUSystemInformation.sImessage.size;
	   F1AP_SETUP_RESP (msg_p).SI_container_length[i][si] = size;
     LOG_D(F1AP, "F1AP: F1Setup-Resp SI_container_length[%d][%d] %ld bytes\n", i, si, size);
	   F1AP_SETUP_RESP (msg_p).SI_container[i][si] = malloc(F1AP_SETUP_RESP (msg_p).SI_container_length[i][si]);

	   memcpy((void*)F1AP_SETUP_RESP (msg_p).SI_container[i][si],
		  (void*)ext->list.array[si]->extensionValue.choice.GNB_CUSystemInformation.sImessage.buf,
		  F1AP_SETUP_RESP (msg_p).SI_container_length[i][si]);
	 }
       }
       break;
     }
   }
   AssertFatal(TransactionId!=-1,"TransactionId was not sent\n");
   AssertFatal(num_cells_to_activate>0,"No cells activated\n");
   F1AP_SETUP_RESP (msg_p).num_cells_to_activate = num_cells_to_activate;

   for (int i=0;i<num_cells_to_activate;i++)  
     AssertFatal(F1AP_SETUP_RESP (msg_p).num_SI[i] > 0, "System Information %d is missing",i);

   MSC_LOG_RX_MESSAGE(
    MSC_F1AP_DU,
    MSC_F1AP_CU,
    0,
    0,
    MSC_AS_TIME_FMT" DU_handle_F1_SETUP_RESPONSE successfulOutcome assoc_id %d",
    0,0,//MSC_AS_TIME_ARGS(ctxt_pP),
    assoc_id);
 
   LOG_D(F1AP, "Sending F1AP_SETUP_RESP ITTI message to ENB_APP with assoc_id (%d->%d)\n",
         assoc_id,ENB_MODULE_ID_TO_INSTANCE(assoc_id));
   itti_send_msg_to_task(TASK_ENB_APP, ENB_MODULE_ID_TO_INSTANCE(assoc_id), msg_p);

   return 0;
}

// SETUP FAILURE
int DU_handle_F1_SETUP_FAILURE(instance_t instance,
                               uint32_t assoc_id,
                               uint32_t stream,
                               F1AP_F1AP_PDU_t *pdu) {
  LOG_E(F1AP, "DU_handle_F1_SETUP_FAILURE\n");
  return 0;
}


/*
    gNB-DU Configuration Update
*/

//void DU_send_gNB_DU_CONFIGURATION_UPDATE(F1AP_GNBDUConfigurationUpdate_t *GNBDUConfigurationUpdate) {
int DU_send_gNB_DU_CONFIGURATION_UPDATE(instance_t instance,
                                         instance_t du_mod_idP,
                                         f1ap_setup_req_t *f1ap_setup_req) {
  F1AP_F1AP_PDU_t                     pdu;
  F1AP_GNBDUConfigurationUpdate_t     *out;
  F1AP_GNBDUConfigurationUpdateIEs_t  *ie;

  uint8_t  *buffer;
  uint32_t  len;
  int       i = 0;
  int       j = 0;

  /* Create */
  /* 0. Message Type */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = F1AP_F1AP_PDU_PR_initiatingMessage;
  pdu.choice.initiatingMessage = (F1AP_InitiatingMessage_t *)calloc(1, sizeof(F1AP_InitiatingMessage_t));
  pdu.choice.initiatingMessage->procedureCode = F1AP_ProcedureCode_id_gNBDUConfigurationUpdate;
  pdu.choice.initiatingMessage->criticality   = F1AP_Criticality_reject;
  pdu.choice.initiatingMessage->value.present = F1AP_InitiatingMessage__value_PR_GNBDUConfigurationUpdate;
  out = &pdu.choice.initiatingMessage->value.choice.GNBDUConfigurationUpdate;

  /* mandatory */
  /* c1. Transaction ID (integer value) */
  ie = (F1AP_GNBDUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBDUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_TransactionID;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBDUConfigurationUpdateIEs__value_PR_TransactionID;
  ie->value.choice.TransactionID = F1AP_get_next_transaction_identifier(instance, du_mod_idP);
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);


  /* mandatory */
  /* c2. Served_Cells_To_Add */
  ie = (F1AP_GNBDUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBDUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_Served_Cells_To_Add_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBDUConfigurationUpdateIEs__value_PR_Served_Cells_To_Add_List;

  for (j=0;
       j<1;
       j++) {
        //
        F1AP_Served_Cells_To_Add_ItemIEs_t *served_cells_to_add_item_ies;
        served_cells_to_add_item_ies = (F1AP_Served_Cells_To_Add_ItemIEs_t *)calloc(1, sizeof(F1AP_Served_Cells_To_Add_ItemIEs_t));
        served_cells_to_add_item_ies->id            = F1AP_ProtocolIE_ID_id_Served_Cells_To_Add_Item;
        served_cells_to_add_item_ies->criticality   = F1AP_Criticality_reject;
        served_cells_to_add_item_ies->value.present = F1AP_Served_Cells_To_Add_ItemIEs__value_PR_Served_Cells_To_Add_Item;
        
        F1AP_Served_Cells_To_Add_Item_t served_cells_to_add_item;
        memset((void *)&served_cells_to_add_item, 0, sizeof(F1AP_Served_Cells_To_Add_Item_t));

        /* 2.1.1 serverd cell Information */
        F1AP_Served_Cell_Information_t served_cell_information;

        memset((void *)&served_cell_information, 0, sizeof(F1AP_Served_Cell_Information_t));
        /* - nRCGI */
        F1AP_NRCGI_t nRCGI;
        memset(&nRCGI, 0, sizeof(F1AP_NRCGI_t));
        MCC_MNC_TO_PLMNID(f1ap_setup_req->mcc[i], f1ap_setup_req->mnc[i], f1ap_setup_req->mnc_digit_length[i], &nRCGI.pLMN_Identity);
        LOG_D(F1AP, "nr_cellId : %x %x %x %x %x\n",
              nRCGI.nRCellIdentity.buf[0],
              nRCGI.nRCellIdentity.buf[1],
              nRCGI.nRCellIdentity.buf[2],
              nRCGI.nRCellIdentity.buf[3],
              nRCGI.nRCellIdentity.buf[4]);
        NR_CELL_ID_TO_BIT_STRING(f1ap_setup_req->nr_cellid[i], &nRCGI.nRCellIdentity);
        served_cell_information.nRCGI = nRCGI;

        /* - nRPCI */
        served_cell_information.nRPCI = f1ap_setup_req->nr_pci[i];  // int 0..1007

        /* - fiveGS_TAC */
        OCTET_STRING_fromBuf(&served_cell_information.fiveGS_TAC,
                             (const char *) &f1ap_setup_req->tac[i],
                             3);

        /* - Configured_EPS_TAC */
        if(1){
          served_cell_information.configured_EPS_TAC = (F1AP_Configured_EPS_TAC_t *)calloc(1, sizeof(F1AP_Configured_EPS_TAC_t));
          OCTET_STRING_fromBuf(served_cell_information.configured_EPS_TAC,
                             "2",
                             2);
        }

        /* - broadcast PLMNs */
        int maxnoofBPLMNS = 1;
        for (i=0;
            i<maxnoofBPLMNS;
            i++) {
            /* > PLMN BroadcastPLMNs Item */
            F1AP_BroadcastPLMNs_Item_t *broadcastPLMNs_Item = (F1AP_BroadcastPLMNs_Item_t *)calloc(1, sizeof(F1AP_BroadcastPLMNs_Item_t));
            //memset((void *)&broadcastPLMNs_Item, 0, sizeof(F1AP_BroadcastPLMNs_Item_t));
            MCC_MNC_TO_PLMNID(f1ap_setup_req->mcc[i], f1ap_setup_req->mnc[i], f1ap_setup_req->mnc_digit_length[i], &broadcastPLMNs_Item->pLMN_Identity);
            ASN_SEQUENCE_ADD(&served_cell_information.servedPLMNs.list, broadcastPLMNs_Item);
        }

        // // /* - CHOICE NR-MODE-Info */
        F1AP_NR_Mode_Info_t nR_Mode_Info;

        if (f1ap_setup_req->fdd_flag) {
          nR_Mode_Info.present = F1AP_NR_Mode_Info_PR_fDD;
          /* > FDD >> FDD Info */
          F1AP_FDD_Info_t *fDD_Info = (F1AP_FDD_Info_t *)calloc(1, sizeof(F1AP_FDD_Info_t));
          /* >>> UL NRFreqInfo */
          fDD_Info->uL_NRFreqInfo.nRARFCN = 999L;

          F1AP_FreqBandNrItem_t ul_freqBandNrItem;
          memset((void *)&ul_freqBandNrItem, 0, sizeof(F1AP_FreqBandNrItem_t));
          ul_freqBandNrItem.freqBandIndicatorNr = 888L;

            F1AP_SupportedSULFreqBandItem_t ul_supportedSULFreqBandItem;
            memset((void *)&ul_supportedSULFreqBandItem, 0, sizeof(F1AP_SupportedSULFreqBandItem_t));
            ul_supportedSULFreqBandItem.freqBandIndicatorNr = 777L;
            ASN_SEQUENCE_ADD(&ul_freqBandNrItem.supportedSULBandList.list, &ul_supportedSULFreqBandItem);

          ASN_SEQUENCE_ADD(&fDD_Info->uL_NRFreqInfo.freqBandListNr.list, &ul_freqBandNrItem);

          /* >>> DL NRFreqInfo */
          fDD_Info->dL_NRFreqInfo.nRARFCN = 666L;

          F1AP_FreqBandNrItem_t dl_freqBandNrItem;
          memset((void *)&dl_freqBandNrItem, 0, sizeof(F1AP_FreqBandNrItem_t));
          dl_freqBandNrItem.freqBandIndicatorNr = 555L;

            F1AP_SupportedSULFreqBandItem_t dl_supportedSULFreqBandItem;
            memset((void *)&dl_supportedSULFreqBandItem, 0, sizeof(F1AP_SupportedSULFreqBandItem_t));
            dl_supportedSULFreqBandItem.freqBandIndicatorNr = 444L;
            ASN_SEQUENCE_ADD(&dl_freqBandNrItem.supportedSULBandList.list, &dl_supportedSULFreqBandItem);

          ASN_SEQUENCE_ADD(&fDD_Info->dL_NRFreqInfo.freqBandListNr.list, &dl_freqBandNrItem);

          /* >>> UL Transmission Bandwidth */
          fDD_Info->uL_Transmission_Bandwidth.nRSCS = F1AP_NRSCS_scs15;
          fDD_Info->uL_Transmission_Bandwidth.nRNRB = F1AP_NRNRB_nrb11;
          /* >>> DL Transmission Bandwidth */
          fDD_Info->dL_Transmission_Bandwidth.nRSCS = F1AP_NRSCS_scs15;
          fDD_Info->dL_Transmission_Bandwidth.nRNRB = F1AP_NRNRB_nrb11;
          
          nR_Mode_Info.choice.fDD = fDD_Info;
        } else { // TDD
          nR_Mode_Info.present = F1AP_NR_Mode_Info_PR_tDD;

          /* > TDD >> TDD Info */
          F1AP_TDD_Info_t *tDD_Info = (F1AP_TDD_Info_t *)calloc(1, sizeof(F1AP_TDD_Info_t));
          /* >>> ARFCN */
          tDD_Info->nRFreqInfo.nRARFCN = 999L; // Integer
          F1AP_FreqBandNrItem_t nr_freqBandNrItem;
          memset((void *)&nr_freqBandNrItem, 0, sizeof(F1AP_FreqBandNrItem_t));
          nr_freqBandNrItem.freqBandIndicatorNr = 555L;

            F1AP_SupportedSULFreqBandItem_t nr_supportedSULFreqBandItem;
            memset((void *)&nr_supportedSULFreqBandItem, 0, sizeof(F1AP_SupportedSULFreqBandItem_t));
            nr_supportedSULFreqBandItem.freqBandIndicatorNr = 444L;
            ASN_SEQUENCE_ADD(&nr_freqBandNrItem.supportedSULBandList.list, &nr_supportedSULFreqBandItem);

          ASN_SEQUENCE_ADD(&tDD_Info->nRFreqInfo.freqBandListNr.list, &nr_freqBandNrItem);

          tDD_Info->transmission_Bandwidth.nRSCS= F1AP_NRSCS_scs15;
          tDD_Info->transmission_Bandwidth.nRNRB= F1AP_NRNRB_nrb11;
     
          nR_Mode_Info.choice.tDD = tDD_Info;
        } 
        
        served_cell_information.nR_Mode_Info = nR_Mode_Info;

        /* - measurementTimingConfiguration */
        char *measurementTimingConfiguration = "0"; // sept. 2018

        OCTET_STRING_fromBuf(&served_cell_information.measurementTimingConfiguration,
                             measurementTimingConfiguration,
                             strlen(measurementTimingConfiguration));
        served_cells_to_add_item.served_Cell_Information = served_cell_information; //

        /* 2.1.2 gNB-DU System Information */
        F1AP_GNB_DU_System_Information_t *gNB_DU_System_Information = (F1AP_GNB_DU_System_Information_t *)calloc(1, sizeof(F1AP_GNB_DU_System_Information_t));

        OCTET_STRING_fromBuf(&gNB_DU_System_Information->mIB_message,  // sept. 2018
                             "1",
                             sizeof("1"));

        OCTET_STRING_fromBuf(&gNB_DU_System_Information->sIB1_message,  // sept. 2018
                             "1",
                             sizeof("1"));
        served_cells_to_add_item.gNB_DU_System_Information = gNB_DU_System_Information; //

        /* ADD */
        served_cells_to_add_item_ies->value.choice.Served_Cells_To_Add_Item = served_cells_to_add_item;

        ASN_SEQUENCE_ADD(&ie->value.choice.Served_Cells_To_Add_List.list, 
                        served_cells_to_add_item_ies);

  }
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);


  /* mandatory */
  /* c3. Served_Cells_To_Modify */
  ie = (F1AP_GNBDUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBDUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_Served_Cells_To_Modify_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBDUConfigurationUpdateIEs__value_PR_Served_Cells_To_Modify_List;

  for (i=0;
       i<1;
       i++) {
        //
        F1AP_Served_Cells_To_Modify_ItemIEs_t *served_cells_to_modify_item_ies;
        served_cells_to_modify_item_ies = (F1AP_Served_Cells_To_Modify_ItemIEs_t *)calloc(1, sizeof(F1AP_Served_Cells_To_Modify_ItemIEs_t));
        served_cells_to_modify_item_ies->id            = F1AP_ProtocolIE_ID_id_Served_Cells_To_Modify_Item;
        served_cells_to_modify_item_ies->criticality   = F1AP_Criticality_reject;
        served_cells_to_modify_item_ies->value.present = F1AP_Served_Cells_To_Modify_ItemIEs__value_PR_Served_Cells_To_Modify_Item;
        
        F1AP_Served_Cells_To_Modify_Item_t served_cells_to_modify_item;
        memset((void *)&served_cells_to_modify_item, 0, sizeof(F1AP_Served_Cells_To_Modify_Item_t));

        /* 3.1 oldNRCGI */
        F1AP_NRCGI_t oldNRCGI;
        memset(&oldNRCGI, 0, sizeof(F1AP_NRCGI_t));
        MCC_MNC_TO_PLMNID(f1ap_setup_req->mcc[i], f1ap_setup_req->mnc[i], f1ap_setup_req->mnc_digit_length[i],
                                         &oldNRCGI.pLMN_Identity);
        NR_CELL_ID_TO_BIT_STRING(f1ap_setup_req->nr_cellid[i], &oldNRCGI.nRCellIdentity);
        served_cells_to_modify_item.oldNRCGI = oldNRCGI;


        /* 3.2.1 serverd cell Information */
        F1AP_Served_Cell_Information_t served_cell_information;
        memset((void *)&served_cell_information, 0, sizeof(F1AP_Served_Cell_Information_t));

        /* - nRCGI */
        F1AP_NRCGI_t nRCGI;
        memset(&nRCGI, 0, sizeof(F1AP_NRCGI_t));
        MCC_MNC_TO_PLMNID(f1ap_setup_req->mcc[i], f1ap_setup_req->mnc[i], f1ap_setup_req->mnc_digit_length[i],
                                         &nRCGI.pLMN_Identity);
        NR_CELL_ID_TO_BIT_STRING(f1ap_setup_req->nr_cellid[i], &nRCGI.nRCellIdentity);
        served_cell_information.nRCGI = nRCGI;

        /* - nRPCI */
        served_cell_information.nRPCI = f1ap_setup_req->nr_pci[i];  // int 0..1007

        /* - fiveGS_TAC */
        OCTET_STRING_fromBuf(&served_cell_information.fiveGS_TAC,
                             (const char *) &f1ap_setup_req->tac[i],
                             3);

        /* - Configured_EPS_TAC */
        if(1){
          served_cell_information.configured_EPS_TAC = (F1AP_Configured_EPS_TAC_t *)calloc(1, sizeof(F1AP_Configured_EPS_TAC_t));
          OCTET_STRING_fromBuf(served_cell_information.configured_EPS_TAC,
                             "2",
                             2);
        }

        /* - broadcast PLMNs */
        int maxnoofBPLMNS = 1;
        for (i=0;
            i<maxnoofBPLMNS;
            i++) {
            /* > PLMN BroadcastPLMNs Item */
            F1AP_BroadcastPLMNs_Item_t *broadcastPLMNs_Item = (F1AP_BroadcastPLMNs_Item_t *)calloc(1, sizeof(F1AP_BroadcastPLMNs_Item_t));
            //memset((void *)&broadcastPLMNs_Item, 0, sizeof(F1AP_BroadcastPLMNs_Item_t));
            MCC_MNC_TO_PLMNID(f1ap_setup_req->mcc[i], f1ap_setup_req->mnc[i], f1ap_setup_req->mnc_digit_length[i], &broadcastPLMNs_Item->pLMN_Identity);
            ASN_SEQUENCE_ADD(&served_cell_information.servedPLMNs.list, broadcastPLMNs_Item);
        }

        // // /* - CHOICE NR-MODE-Info */
        F1AP_NR_Mode_Info_t nR_Mode_Info;

        if (f1ap_setup_req->fdd_flag) {
          nR_Mode_Info.present = F1AP_NR_Mode_Info_PR_fDD;
          /* > FDD >> FDD Info */
          F1AP_FDD_Info_t *fDD_Info = (F1AP_FDD_Info_t *)calloc(1, sizeof(F1AP_FDD_Info_t));
          /* >>> UL NRFreqInfo */
          fDD_Info->uL_NRFreqInfo.nRARFCN = 999L;

          F1AP_FreqBandNrItem_t ul_freqBandNrItem;
          memset((void *)&ul_freqBandNrItem, 0, sizeof(F1AP_FreqBandNrItem_t));
          ul_freqBandNrItem.freqBandIndicatorNr = 888L;

            F1AP_SupportedSULFreqBandItem_t ul_supportedSULFreqBandItem;
            memset((void *)&ul_supportedSULFreqBandItem, 0, sizeof(F1AP_SupportedSULFreqBandItem_t));
            ul_supportedSULFreqBandItem.freqBandIndicatorNr = 777L;
            ASN_SEQUENCE_ADD(&ul_freqBandNrItem.supportedSULBandList.list, &ul_supportedSULFreqBandItem);

          ASN_SEQUENCE_ADD(&fDD_Info->uL_NRFreqInfo.freqBandListNr.list, &ul_freqBandNrItem);

          /* >>> DL NRFreqInfo */
          fDD_Info->dL_NRFreqInfo.nRARFCN = 666L;

          F1AP_FreqBandNrItem_t dl_freqBandNrItem;
          memset((void *)&dl_freqBandNrItem, 0, sizeof(F1AP_FreqBandNrItem_t));
          dl_freqBandNrItem.freqBandIndicatorNr = 555L;

            F1AP_SupportedSULFreqBandItem_t dl_supportedSULFreqBandItem;
            memset((void *)&dl_supportedSULFreqBandItem, 0, sizeof(F1AP_SupportedSULFreqBandItem_t));
            dl_supportedSULFreqBandItem.freqBandIndicatorNr = 444L;
            ASN_SEQUENCE_ADD(&dl_freqBandNrItem.supportedSULBandList.list, &dl_supportedSULFreqBandItem);

          ASN_SEQUENCE_ADD(&fDD_Info->dL_NRFreqInfo.freqBandListNr.list, &dl_freqBandNrItem);

          /* >>> UL Transmission Bandwidth */
          fDD_Info->uL_Transmission_Bandwidth.nRSCS = F1AP_NRSCS_scs15;
          fDD_Info->uL_Transmission_Bandwidth.nRNRB = F1AP_NRNRB_nrb11;
          /* >>> DL Transmission Bandwidth */
          fDD_Info->dL_Transmission_Bandwidth.nRSCS = F1AP_NRSCS_scs15;
          fDD_Info->dL_Transmission_Bandwidth.nRNRB = F1AP_NRNRB_nrb11;
          
          nR_Mode_Info.choice.fDD = fDD_Info;
        } else { // TDD
          nR_Mode_Info.present = F1AP_NR_Mode_Info_PR_tDD;

          /* > TDD >> TDD Info */
          F1AP_TDD_Info_t *tDD_Info = (F1AP_TDD_Info_t *)calloc(1, sizeof(F1AP_TDD_Info_t));
          /* >>> ARFCN */
          tDD_Info->nRFreqInfo.nRARFCN = 999L; // Integer
          F1AP_FreqBandNrItem_t nr_freqBandNrItem;
          memset((void *)&nr_freqBandNrItem, 0, sizeof(F1AP_FreqBandNrItem_t));
          nr_freqBandNrItem.freqBandIndicatorNr = 555L;

            F1AP_SupportedSULFreqBandItem_t nr_supportedSULFreqBandItem;
            memset((void *)&nr_supportedSULFreqBandItem, 0, sizeof(F1AP_SupportedSULFreqBandItem_t));
            nr_supportedSULFreqBandItem.freqBandIndicatorNr = 444L;
            ASN_SEQUENCE_ADD(&nr_freqBandNrItem.supportedSULBandList.list, &nr_supportedSULFreqBandItem);

          ASN_SEQUENCE_ADD(&tDD_Info->nRFreqInfo.freqBandListNr.list, &nr_freqBandNrItem);

          tDD_Info->transmission_Bandwidth.nRSCS= F1AP_NRSCS_scs15;
          tDD_Info->transmission_Bandwidth.nRNRB= F1AP_NRNRB_nrb11;
     
          nR_Mode_Info.choice.tDD = tDD_Info;
        } 
        
        served_cell_information.nR_Mode_Info = nR_Mode_Info;

        /* - measurementTimingConfiguration */
        char *measurementTimingConfiguration = "0"; // sept. 2018

        OCTET_STRING_fromBuf(&served_cell_information.measurementTimingConfiguration,
                             measurementTimingConfiguration,
                             strlen(measurementTimingConfiguration));
        served_cells_to_modify_item.served_Cell_Information = served_cell_information; //

        /* 3.2.2 gNB-DU System Information */
        F1AP_GNB_DU_System_Information_t *gNB_DU_System_Information = (F1AP_GNB_DU_System_Information_t *)calloc(1, sizeof(F1AP_GNB_DU_System_Information_t));

        OCTET_STRING_fromBuf(&gNB_DU_System_Information->mIB_message,  // sept. 2018
                             "1",
                             sizeof("1"));

        OCTET_STRING_fromBuf(&gNB_DU_System_Information->sIB1_message,  // sept. 2018
                             "1",
                             sizeof("1"));
        served_cells_to_modify_item.gNB_DU_System_Information = gNB_DU_System_Information; //

        /* ADD */
        served_cells_to_modify_item_ies->value.choice.Served_Cells_To_Modify_Item = served_cells_to_modify_item;

        ASN_SEQUENCE_ADD(&ie->value.choice.Served_Cells_To_Modify_List.list, 
                        served_cells_to_modify_item_ies);

  }
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);


  /* mandatory */
  /* c4. Served_Cells_To_Delete */
  ie = (F1AP_GNBDUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBDUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_Served_Cells_To_Delete_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBDUConfigurationUpdateIEs__value_PR_Served_Cells_To_Delete_List;

  for (i=0;
       i<1;
       i++) {
        //
        F1AP_Served_Cells_To_Delete_ItemIEs_t *served_cells_to_delete_item_ies;
        served_cells_to_delete_item_ies = (F1AP_Served_Cells_To_Delete_ItemIEs_t *)calloc(1, sizeof(F1AP_Served_Cells_To_Delete_ItemIEs_t));
        served_cells_to_delete_item_ies->id            = F1AP_ProtocolIE_ID_id_Served_Cells_To_Delete_Item;
        served_cells_to_delete_item_ies->criticality   = F1AP_Criticality_reject;
        served_cells_to_delete_item_ies->value.present = F1AP_Served_Cells_To_Delete_ItemIEs__value_PR_Served_Cells_To_Delete_Item;
        
        F1AP_Served_Cells_To_Delete_Item_t served_cells_to_delete_item;
        memset((void *)&served_cells_to_delete_item, 0, sizeof(F1AP_Served_Cells_To_Delete_Item_t));

        /* 3.1 oldNRCGI */
        F1AP_NRCGI_t oldNRCGI;
        memset(&oldNRCGI, 0, sizeof(F1AP_NRCGI_t));
        MCC_MNC_TO_PLMNID(f1ap_setup_req->mcc[i], f1ap_setup_req->mnc[i], f1ap_setup_req->mnc_digit_length[i],
                                         &oldNRCGI.pLMN_Identity);
        NR_CELL_ID_TO_BIT_STRING(f1ap_setup_req->nr_cellid[i], &oldNRCGI.nRCellIdentity);
        served_cells_to_delete_item.oldNRCGI = oldNRCGI;

        /* ADD */
        served_cells_to_delete_item_ies->value.choice.Served_Cells_To_Delete_Item = served_cells_to_delete_item;

        ASN_SEQUENCE_ADD(&ie->value.choice.Served_Cells_To_Delete_List.list, 
                         served_cells_to_delete_item_ies);
  }
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);


  /* mandatory */
  /* c5. Active_Cells_List */
  ie = (F1AP_GNBDUConfigurationUpdateIEs_t *)calloc(1, sizeof(F1AP_GNBDUConfigurationUpdateIEs_t));
  ie->id                        = F1AP_ProtocolIE_ID_id_Active_Cells_List;
  ie->criticality               = F1AP_Criticality_reject;
  ie->value.present             = F1AP_GNBDUConfigurationUpdateIEs__value_PR_Active_Cells_List;

  for (i=0;
       i<1;
       i++) {
        //
        F1AP_Active_Cells_ItemIEs_t *active_cells_item_ies;
        active_cells_item_ies = (F1AP_Active_Cells_ItemIEs_t *)calloc(1, sizeof(F1AP_Active_Cells_ItemIEs_t));
        active_cells_item_ies->id            = F1AP_ProtocolIE_ID_id_Active_Cells_Item;
        active_cells_item_ies->criticality   = F1AP_Criticality_reject;
        active_cells_item_ies->value.present = F1AP_Active_Cells_ItemIEs__value_PR_Active_Cells_Item;
        
        F1AP_Active_Cells_Item_t active_cells_item;
        memset((void *)&active_cells_item, 0, sizeof(F1AP_Active_Cells_Item_t));

        /* 3.1 oldNRCGI */
        F1AP_NRCGI_t nRCGI;
        memset(&nRCGI, 0, sizeof(F1AP_NRCGI_t));
        MCC_MNC_TO_PLMNID(f1ap_setup_req->mcc[i], f1ap_setup_req->mnc[i], f1ap_setup_req->mnc_digit_length[i],
                                         &nRCGI.pLMN_Identity);
        NR_CELL_ID_TO_BIT_STRING(f1ap_setup_req->nr_cellid[i], &nRCGI.nRCellIdentity);
        active_cells_item.nRCGI = nRCGI;
        
        /* ADD */
        active_cells_item_ies->value.choice.Active_Cells_Item = active_cells_item;

        ASN_SEQUENCE_ADD(&ie->value.choice.Active_Cells_List.list, 
                         active_cells_item_ies);
  }
  ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);


  if (f1ap_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(F1AP, "Failed to encode F1 setup request\n");
    return -1;
  }

  return 0;
}



int DU_handle_gNB_DU_CONFIGURATION_FAILURE(instance_t instance,
                                uint32_t assoc_id,
                                uint32_t stream,
                                F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int DU_handle_gNB_DU_CONFIGURATION_UPDATE_ACKNOWLEDGE(instance_t instance,
                                uint32_t assoc_id,
                                uint32_t stream,
                                F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}


int DU_handle_gNB_CU_CONFIGURATION_UPDATE(instance_t instance,
                                uint32_t assoc_id,
                                uint32_t stream,
                                F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int DU_send_gNB_CU_CONFIGURATION_UPDATE_FAILURE(instance_t instance,
                    F1AP_GNBCUConfigurationUpdateFailure_t *GNBCUConfigurationUpdateFailure) {
  AssertFatal(1==0,"Not implemented yet\n");
}

int DU_send_gNB_CU_CONFIGURATION_UPDATE_ACKNOWLEDGE(instance_t instance,
                    F1AP_GNBCUConfigurationUpdateAcknowledge_t *GNBCUConfigurationUpdateAcknowledge) {
  AssertFatal(1==0,"Not implemented yet\n");
}


int DU_send_gNB_DU_RESOURCE_COORDINATION_REQUEST(instance_t instance,
                    F1AP_GNBDUResourceCoordinationRequest_t *GNBDUResourceCoordinationRequest) {
  AssertFatal(0, "Not implemented yet\n");
}

int DU_handle_gNB_DU_RESOURCE_COORDINATION_RESPONSE(instance_t instance,
                                uint32_t assoc_id,
                                uint32_t stream,
                                F1AP_F1AP_PDU_t *pdu) {
  AssertFatal(0, "Not implemented yet\n");
}
