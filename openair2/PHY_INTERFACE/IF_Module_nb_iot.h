
/*This is the interface module between PHY
*Provided the FAPI style interface structures for P7.
*Provide the semi-FAPI style interface for P5 (configuration)
*
*/

#ifndef __IF_MODULE_NB_IoT__H__
#define __IF_MODULE_NB_IoT__H__

#include "nfapi_interface.h"
#include "openair1/PHY/LTE_TRANSPORT/defs_nb_iot.h"
#include "PhysicalConfigDedicated-NB-r13.h"
#include "openair2/PHY_INTERFACE/IF_Module_nb_iot.h"
#include "openair2/COMMON/platform_types.h"

#define SCH_PAYLOAD_SIZE_MAX 4096
#define BCCH_PAYLOAD_SIZE_MAX 128



// P5 FAPI-like configuration structures-------------------------------------------------------------------------------

/*MP: MISSED COMMON CONFIG. of SIB2-NB in FAPI SPECS (may non needed)*/
typedef struct{
	//nprach_config
	uint16_t nprach_config_0_subcarrier_MSG3_range_start;
	uint16_t nprach_config_1_subcarrier_MSG3_range_start;
	uint16_t nprach_config_2_subcarrier_MSG3_range_start;
	uint16_t nprach_config_0_max_num_preamble_attempt_CE;
	uint16_t nprach_config_1_max_num_preamble_attempt_CE;
	uint16_t nprach_config_2_max_num_preamble_attempt_CE;
	uint16_t nprach_config_0_npdcch_num_repetitions_RA;
	uint16_t nprach_config_1_npdcch_num_repetitions_RA;
	uint16_t nprach_config_2_npdcch_num_repetitions_RA;
	uint16_t nprach_config_0_npdcch_startSF_CSS_RA;
	uint16_t nprach_config_1_npdcch_startSF_CSS_RA;
	uint16_t nprach_config_2_npdcch_startSF_CSS_RA;
	uint16_t nprach_config_0_npdcch_offset_RA;
	uint16_t nprach_config_1_npdcch_offset_RA;
	uint16_t nprach_config_2_npdcch_offset_RA;

	//npusch ConfigCommon (carried by the NULSCH PDU in FAPI--> so maybe not a static parameter)
	//not used
	ACK_NACK_NumRepetitions_NB_r13_t *ack_nack_numRepetitions_MSG4; //pointer to the first cell of a list of ack_nack_num_repetitions

    //ulPowerControlCommon (UE side)
    uint16_t p0_nominal_npusch;
	uint16_t alpha;
	uint16_t delta_preamle_MSG3;

}extra_phyConfigCommon_t;

typedef struct{

	/*OAI config. parameters*/
	module_id_t mod_id;
	int CC_id;
	uint16_t rnti;
	int get_MIB; //should be different from 0 only when the mib!= null (NB_rrc_mac_config_req_eNB)
	int get_COMMON;
	int get_DEDICATED;

	//ID of the Resource Block dedicated to NB-IoT
	//For Nb-IoT only a restricted values of PRB indexes are allowed (see Rhode&Shwartz pag9)
	//unsigned short NB_IoT_RB_ID; (should coincide with PRB index)

	//In 3GPP specs (TS 36.101 Table 5.7.3-1 and ch 5.7.3F) see also SIB2-NB freqInfo.ul-carrierFreq
	//this parameters should be evaluated based of the EUTRA Absolute Radio Frequency Channel Number (EARFCN)
	//in FAPI this value is given inside the BROADCAST DETECT request (P4 Network Monitor Mode procedure)
	//in OAI we set the dl_CarrierFrequenci at configuration time (see COMMON/rrc_messages_types.h)
	//then adding an offset for the ul_CarrierFreq ( see RU-RAU split approach - init_SI)
	//uint32_t dl_CarrierFreq;
	//uint32_t ul_CarrierFreq; --> problem solved since we directly evaluate the EARFCN has requested by FAPI specs


	/*FAPI style config. parameters
	 *
	 * useful config message contents:
	 * -nfapi_uplink_reference_signal_config_t uplink_reference_signal_config
	 * -nfapi_subframe_config_t subframe_config;
	 * -nfapi_rf_config_t rf_config;
	 * -nfapi_sch_config_t sch_config;
	 * -nfapi_nb_iot_config_t nb_iot_config;
	 * -nfapi_l23_config_t l23_config;
	 * -nfapi_config --> EARFCN
	 *
	 * */
	//XXX where allocate memory??
	nfapi_config_request_t* cfg;


	/*Dedicated configuration -->not supported by FAPI (may not needed)
	 * In OAI at least are needed when we manage the phy_procedures_eNB_TX in which we call the phy_config_dedicated_eNB_step2
	 * that use the physicalConfigDedicated info previously stored in the PHY_VARS_eNB structure through the phy_config_dedicated procedure
	 */
	PhysicalConfigDedicated_NB_r13_t *phy_config_dedicated;

	/*MP: MISSED COMMON CONFIG. of SIB2-NB in FAPI SPECS (may non needed)*/
	extra_phyConfigCommon_t extra_phy_parms;

}PHY_Config_t;



// uplink subframe P7---------------------------------------------------------------------------------


/*UL_IND_t:
* A structure handles all the uplink information.
* Corresponding to the NRACH.indicaiton, UL_Config_indication, RX_ULSCH.indication, CRC.inidcation, NB_HARQ.indication in FAPI
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

 	nfapi_nrach_indication_body_t NRACH;

 	/*Uplink data part*/

 	/*indication of the harq feedback*/
 	nfapi_nb_harq_indication_t nb_harq_ind;
 	/*indication of the uplink data PDU*/
  	nfapi_rx_indication_body_t RX_NPUSCH;
 	/*crc_indication*/
 	nfapi_crc_indication_body_t crc_ind;

 }UL_IND_t;

 // Downlink subframe P7


typedef struct{

 	/*Start at the common part*/

 	//Module ID
	module_id_t module_id; 
 	//CC ID
 	int CC_id;
 	//frame
 	frame_t frame;
 	//subframe
 	sub_frame_t subframe;

  	/// nFAPI DL Config Request
  	nfapi_dl_config_request_body_t *DL_req;
  	/// nFAPI UL Config Request
  	nfapi_ul_config_request_t *UL_req;
  	/// nFAPI HI_DCI Request
  	nfapi_hi_dci0_request_body_t *HI_DCI0_req;
  	/// Pointers to DL SDUs
  	uint8_t **sdu;

}Sched_Rsp_t;


/*IF_Module_t a group for gathering the Interface
It should be allocated at the main () in lte-softmodem.c*/
typedef struct IF_Module_s{
	//define the function pointer
	void (*UL_indication)(UL_IND_t *UL_INFO);
	void (*schedule_response)(Sched_Rsp_t *Sched_INFO);
	void (*PHY_config_req)(PHY_Config_t* config_INFO);

}IF_Module_t;

/*Initial */

//int IF_Module_init(IF_Module_t *if_inst);

void IF_Module_init_L1(void);
void IF_Module_init_L2(void);


#endif
