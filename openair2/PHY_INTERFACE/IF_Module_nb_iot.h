
/*This is the interface module between PHY
*Provided the FAPI style interface structures for P7.
*Provide the semi-FAPI style interface for P5 (configuration)
*
*/

#ifndef __IF_MODULE_NB_IoT__H__
#define __IF_MODULE_NB_IoT__H__

#include "openair1/PHY/LTE_TRANSPORT/defs_nb_iot.h"
#include "PhysicalConfigDedicated-NB-r13.h"
#include "openair2/PHY_INTERFACE/IF_Module_nb_iot.h"
#include "temp_nfapi_interface.h"

#define SCH_PAYLOAD_SIZE_MAX 4096
#define BCCH_PAYLOAD_SIZE_MAX 128
#define NUMBER_OF_UE_MAX 20


// P5 FAPI-like configuration structures

typedef struct{

	/*OAI config. parameters*/
	module_id_t mod_id;
	int CC_id;
	uint16_t rnti;
	int get_MIB; //should be different from 0 only when the mib!= null (NB_rrc_mac_config_req_eNB)
	int get_COMMON;
	int get_DEDICATED;

	//In FAPI specs --> is inside the nb_iot_rssi_request (P4 Network Monitor Mode procedure)
	//In OAI is called eutra_band
	uint8_t frequency_band_indicator; //parameter carried by the SIB1-NB, is and index of the Table 5.5-1 TS 36.101

	//In 3GPP specs (TS 36.101 Table 5.7.3-1 and ch 5.7.3F) see also SIB2-NB freqInfo.ul-carrierFreq
	//this parameters should be evaluated based of the EUTRA Absolute Radio Frequency Channel Number (EARFCN)
	//in FAPI this value is given inside th BROADCAST DETECT request (P4 Network Monito Mode procedure)
	//in OAI we set the dl_CarrierFrequenci at configuration time (see COMMON/rrc_messages_types.h)
	//then adding an offset for the ul_CarrierFreq ( see RU-RAU split approach - init_SI)
	uint32_t dl_CarrierFreq;
	uint32_t ul_CarrierFreq;


	/*FAPI style config. parameters*/
	nfapi_uplink_reference_signal_config_t uplink_reference_signal_config;
	nfapi_subframe_config_t subframe_config;
	nfapi_rf_config_t rf_config;
	nfapi_sch_config_t sch_config;
	nfapi_nb_iot_config_t nb_iot_config;
	nfapi_l23_config_t l23_config;

	/*Dedicated configuration --> not supported by FAPI?*/
	PhysicalConfigDedicated_NB_r13_t *phy_config_dedicated;


}PHY_Config_t;



// uplink subframe P7

typedef struct{

 	//index of the preamble, detected initial subcarrier (0-47)
 	uint16_t preamble_index;
 	//timing offset by PHY
 	int16_t timing_offset;
 	//Indicates the NRACH CE level as configured in CONFIG (0,1,2 = CE level 0,1,2)
 	uint8_t NRACH_CE_Level;
 	//RA-RNTI
 	uint16_t RNTI;
 	//Timing Advance
 	uint16_t TA;

}NRACH_t;

/*UL_SPEC_t:
* A struture mainly describes the UE specific information. (for NB_rx_sdu)
* Corresponding to the RX_ULSCH.indication, CRC.inidcation, NB_HARQ.indication in FAPI
*/
typedef struct{

	// 0 = format 1 (data), 1 = formaat 2 (ACK/NACK)
	uint8_t NPUSCH_format;
	//An opaque handling returned in the RX.indication
	uint32_t OPA_handle;
	//rnti
	uint16_t RNTI;
	//Pointer to sdu
	uint8_t *sdu;
	//Pointer to sdu length 
	uint16_t sdu_lenP;
	//HARQ ID
	int harq_pidP;
	//MSG3 flag
	uint8_t *msg3_flagP;
	//CRC indication for the message is corrected or not for the Uplink HARQ
	uint8_t crc_ind;

	//This ACK NACK is for the Downlink HARQ feedback received by the NPUSCH from UE
	uint8_t NAK;

}UL_SPEC_t;


/*UL_IND_t:
* A structure handles all the uplink information.
*/
typedef struct{

 	/*Start at the common part*/

 	int test;

 	//Module ID
 	module_id_t module_id;
 	//CC ID
 	int CC_id;
 	//frame 
 	frame_t frame;
 	//subframe
 	sub_frame_t subframe;

 	/*preamble part*/

 	//number of the subcarrier detected in the same time
 	uint8_t Number_SC;
 	//NRACH Indication parameters list
 	NRACH_t Preamble_list[48];

 	/*UE specific part*/

 	//Number of availble UE for Uplink
 	int UE_NUM;
 	//Uplink Schedule information
 	UL_SPEC_t UL_SPEC_Info[NUMBER_OF_UE_MAX]; 

 }UL_IND_t;

 // Downlink subframe P7

 typedef struct{

 	//The length (in bytes)
 	uint16_t Length;
 	//PDU index 
 	uint16_t PDU_index;
 	//Transmission Power
 	uint16_t Trans_Power;
 	//HYPER SFN 2lsbs
 	uint16_t HyperSFN2lsbs;
 	//NPBCH pdu payload
 	uint8_t npbch_pdu_payload[4]; 

 }npbch_t;

 typedef struct{

 	//The length (in bytes)
 	uint16_t Length;
 	//PDU index 
 	uint16_t PDU_index;
 	//start symbol 0-4 0 for guard-band and standalone operating
 	uint8_t start_symbol;
 	//RNTI type,0 = BCCH(SIB), 1 for DL data
 	uint8_t RNTI_type;
 	// RNTI 
 	uint16_t RNTI;
 	// SIB payload
 	uint8_t nbcch_pdu_payload[BCCH_PAYLOAD_SIZE_MAX];
 	// NDLSCH payload
 	uint8_t ndlsch_pdu_payload[SCH_PAYLOAD_SIZE_MAX];

 }npdsch_t;

 typedef struct{

 	// The length (in bytes)
 	uint16_t Length;
 	// PDU index 
 	uint16_t PDU_index;
 	// NCCE index value 0 -> 1
 	uint8_t NCCE_index;
 	// Aggregation level
 	uint8_t aggregation;
 	// start symbol
 	uint8_t start_symbol;
 	// RNTI type,0 = TC-RNTI, 1 = RA-RNTI, 2 = P-RNTI 3 = other
 	uint8_t RNTI_type;
 	// RNTI 
 	uint16_t RNTI;
 	// Scrambliing re-initialization batch index from FAPI specs (1-4)
 	uint8_t batch_index;
 	// NRS antenna ports assumed by the UE from FAPI specs (1-2)
 	uint8_t num_antenna;
 	// Number of DCI
  	uint8_t Num_dci;
  	// Format of DCI
 	DCI_format_NB_t DCI_Format;
 	// Content of DCI
 	DCI_CONTENT *DCI_Content;

 }npdcch_t;

typedef union{

	npdcch_t NB_DCI;
 	
 	npdsch_t NB_DLSCH;

 	npbch_t NB_BCH;

}NB_DL_u;


typedef struct{

 	/*Start at the common part*/

 	//Module ID
	module_id_t module_idP; 
 	//CC ID
 	int CC_id;
 	//frame
 	frame_t frameP;
 	//subframe
 	sub_frame_t subframeP;

 	NB_DL_u NB_DL;

}Sched_Rsp_t;


/*IF_Module_t*/
typedef struct IF_Module_s{
	//define the function pointer
	void (*UL_indication)(UL_IND_t UL_INFO);
	void (*schedule_response)(Sched_Rsp_t Sched_INFO);
	void (*PHY_config_req)(PHY_Config_t* config_INFO);

}IF_Module_t;

/*Initial */

//int IF_Module_init(IF_Module_t *if_inst);

IF_Module_t* IF_Module_init_L1(IF_Module_t *if_inst);
IF_Module_t* IF_Module_init_L2(IF_Module_t *if_inst);


#endif
