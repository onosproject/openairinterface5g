
/*This is the interface module between PHY
*Provided the FAPI style interface structures for P7.
*Provide the semi-FAPI style interface for P5 (configuration)
*
*/

#ifndef __IF_MODULE_NB_IoT__H__
#define __IF_MODULE_NB_IoT__H__

#include "openair1/PHY/LTE_TRANSPORT/defs_nb_iot.h"
#include "PhysicalConfigDedicated-NB-r13.h"


#define SCH_PAYLOAD_SIZE_MAX 4096
#define BCCH_PAYLOAD_SIZE_MAX 128
#define NUMBER_OF_UE_MAX 20


// P5 FAPI-like configuration structures

typedef struct{
	uint16_t duplex_mode;
	uint16_t pcfich_power_offset;
	uint16_t p_b; //refers to DL power allocation (see TS 36.213 ch 5.2
	uint16_t dl_cyclic_prefix_type;
	uint16_t ul_cyclic_prefix_type;
}subframe_config_t;

typedef struct{
	uint16_t dl_channel_bandwidth;
	uint16_t ul_channel_bandwidth;
	uint16_t reference_signal_power;
	uint16_t tx_antenna_ports;
	uint16_t rx_antenna_ports;
}rf_config_t;

typedef struct{
	uint16_t primary_sinchronization_signal_epre_eprers;
	uint16_t secondary_sinchronization_signal_epre_eprers;
	uint16_t physical_cell_id; //aka Ncell_id
}sch_config_t;

typedef struct{

	uint16_t operating_mode;
	uint16_t anchor;
	uint16_t prb_index;
	uint16_t control_region_size;
	uint16_t assumed_crs_aps;

	//enable or disable configuration #0 (value: 0 = Disable, 1 = Enable)
	uint16_t nprach_config_0_enabled;
	//periodicity of NPRACH resource (value 0,1,2,3,4,5,6,7 correspond to 40,80,160,240,320,640,1280,2560ms)
	uint16_t nprach_config_0_sf_periodicity;
	//NPRACH resource starting time after period (value 0,1,2,3,4,5,6,7 correspond to 8,16,32,64,128,256,512,1024ms)
	uint16_t nprach_config_0_start_time;
	//Frequency location of an NPRACH resource within a PRB (value 0,1,2,3,4,5,6 correspond to 0,12,24,36,2,18,34
	uint16_t nprach_config_0_subcarrier_offset;
	//Number of Subcarriers in NPRACH resource (value 0,1,2,3 correspond to 12,24,36,48)
	uint16_t nprach_config_0_number_of_subcarriers;
	//Cyclic prefix length for NPRACH transmission (value: 0 = 66.7usec, 1 = 266.7usec)
	uint16_t nprach_config_0_cp_length;
	//Number of repetitions for NPRACH transmission (value: 0,1,2,3,4,5,6,7 correspond to 1,2,4,8,16,32,64,128)
	uint16_t nprach_config_0_number_of_repetitions_per_attempts;

	uint16_t nprach_config_1_enabled;
	uint16_t nprach_config_1_sf_periodicity;
	uint16_t nprach_config_1_start_time;
	uint16_t nprach_config_1_subcarrier_offset;
	uint16_t nprach_config_1_number_of_subcarriers;
	uint16_t nprach_config_1_cp_length;
	uint16_t nprach_config_1_number_of_repetitions_per_attempts;

	uint16_t nprach_config_2_enabled;
	uint16_t nprach_config_2_sf_periodicity;
	uint16_t nprach_config_2_start_time;
	uint16_t nprach_config_2_subcarrier_offset;
	uint16_t nprach_config_2_number_of_subcarriers;
	uint16_t nprach_config_2_cp_length;
	uint16_t nprach_config_2_number_of_repetitions_per_attempts;

	//4 bits
	uint16_t three_tone_base_sequence;/*OPTIONAL*/
	//2bits
	uint16_t six_tone_base_sequence; /*OPTIONAL*/
	//5 bits
	uint16_t twelve_tone_base_sequence; /*OPTIONAL*/
	uint16_t three_tone_cyclic_shift;
	uint16_t six_tone_cyclic_shift;

	//Enable/disable the DL gap
	uint16_t dl_gap_config_enable;
	//Threshold on the maximum number of repetitions configured for NPDCCH before application of DL transmission gap config.
	//value 0,1,2,3 correspond to 32,64,128,256
	uint16_t dl_gap_threshold;
	//Periodicity of a DL tranmission gap (value 0,1,2,3 correspond to 64,128,256,512sf)
	uint16_t dl_gap_periodicity;
	//Coefficent to calculate the gap duration of a DL transmission (value 0,1,2,3 correspond to oneEight, oneFourth, threeEight, oneHalf)
	uint16_t dl_gap_duration_coefficent;

}nb_iot_config_t;

typedef struct{
	uint16_t data_report_mode;
	uint16_t sfn_sf;
}l23_config_t;

typedef struct{

	/*OAI config. parameters*/
	module_id_t mod_id;
	int CC_id;
	int rnti;
	int get_MIB; //should be different from 0 only when the mib!= null (NB_rrc_mac_config_req_eNB)

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
	subframe_config_t subframe_config;
	rf_config_t rf_config;
	sch_config_t sch_config;
	nb_iot_config_t nb_iot_config;
	l23_config_t l23_config;

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

//	Calvin 20170531 start

/*IF_Module_t*/
typedef struct IF_Module_s{
	void (*UL_indication)(UL_IND_t UL_INFO);
	void (*Schedule_Response)(Sched_Rsp_t Sched_INFO);
	void (*PHY_config_req)(PHY_Config_t config_INFO);
}IF_Module_t;

/*Initial */
int IF_Module_init(IF_Module_t *if_inst);

//	Calvin 20170531 end

/*Interface for uplink, transmitting the Preamble(list), ULSCH SDU, NAK, Tick (trigger scheduler)
*/
void UL_indication(UL_IND_t UL_INFO);

/*Interface for Downlink, transmitting the DLSCH SDU, DCI SDU*/
void Schedule_Response(Sched_Rsp_t Sched_INFO);

/*Interface for PHY Configuration
 * Trigger the phy_config_xxx functions using parameters from the shared PHY_Config structure
 * */
void PHY_config_req(PHY_Config_t config_INFO);


#endif
