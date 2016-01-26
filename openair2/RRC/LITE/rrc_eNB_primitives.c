/*******************************************************************************
    OpenAirInterface
    Copyright(c) 1999 - 2015 Eurecom

    OpenAirInterface is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.


    OpenAirInterface is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with OpenAirInterface.The full GNU General Public License is
    included in this distribution in the file called "COPYING". If not,
    see <http://www.gnu.org/licenses/>.

   Contact Information
   OpenAirInterface Admin: openair_admin@eurecom.fr
   OpenAirInterface Tech : openair_tech@eurecom.fr
   OpenAirInterface Dev  : openair4g-devel@lists.eurecom.fr

   Address      : Eurecom, Campus SophiaTech, 450 Route des Chappes, CS 50193 - 06904 Biot Sophia Antipolis cedex, FRANCE

 *******************************************************************************/

/*! \file event_handler.h
* \brief primitives to handle event acting on oai
* \author Konstantinos Alexandris,
* \date 2015
* \version 0.5
* @ingroup _oai
*/

#include "targets/SIMU/USER/oaisim.h"
#include "rrc_eNB_primitives.h"
#include "UTIL/OCG/OCG.h"

// HO parameters initialization
void init_HO(Handover_info* ho_info)
{
	int enb_module_id=0;

	for(enb_module_id=0; enb_module_id<NUMBER_OF_eNB_MAX; enb_module_id++) {
		// Set ofn parameter
		set_hys(enb_module_id,ho_info->hys);
		set_ttt_ms(enb_module_id,ho_info->ttt_ms);
		set_ofn(enb_module_id,ho_info->ofn);
		set_ocn(enb_module_id,ho_info->ocn);
		set_ofs(enb_module_id,ho_info->ofs);
		set_ocs(enb_module_id,ho_info->ocs);
		set_off(enb_module_id,ho_info->off);
		set_rsrp_filter_coeff(enb_module_id,ho_info->rsrp_filter_coeff);
		set_rsrq_filter_coeff(enb_module_id,ho_info->rsrq_filter_coeff);
	}
}

// Getters/setters for HO parameters

// Hysteresis
void set_hys(int enb_module_id,long hys){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	eNB_mac_inst->ho_info.hys=hys;
	printf("Hysteresis for eNB %d is set to %ld\n",enb_module_id,eNB_mac_inst->ho_info.hys);
}

long get_hys(int enb_module_id){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	return eNB_mac_inst->ho_info.hys;
}

// Time to trigger
void set_ttt_ms(int enb_module_id,long ttt_ms){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	eNB_mac_inst->ho_info.ttt_ms=ttt_ms;
	printf("Time to trigger for eNB %d is set to %ld\n",enb_module_id,eNB_mac_inst->ho_info.ttt_ms);
}

long get_ttt_ms(int enb_module_id){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	return eNB_mac_inst->ho_info.ttt_ms;
}

// OFN
void set_ofn(int enb_module_id,long ofn){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	eNB_mac_inst->ho_info.ofn=ofn;
	printf("OFN for eNB %d is set to %ld\n",enb_module_id,eNB_mac_inst->ho_info.ofn);
}

long get_ofn(int enb_module_id){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	return eNB_mac_inst->ho_info.ofn;
}

// OCN
void set_ocn(int enb_module_id,long ocn){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	eNB_mac_inst->ho_info.ocn=ocn;
	printf("OCN for eNB %d is set to %ld\n",enb_module_id,eNB_mac_inst->ho_info.ocn);
}

long get_ocn(int enb_module_id){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	return eNB_mac_inst->ho_info.ocn;
}

// OFS
void set_ofs(int enb_module_id,long ofs){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	eNB_mac_inst->ho_info.ofs=ofs;
	printf("OFS for eNB %d is set to %ld\n",enb_module_id,eNB_mac_inst->ho_info.ofs);
}

long get_ofs(int enb_module_id){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	return eNB_mac_inst->ho_info.ofs;
}

// OCS
void set_ocs(int enb_module_id,long ocs){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	eNB_mac_inst->ho_info.ocs=ocs;
	printf("OCS for eNB %d is set to %ld\n",enb_module_id,eNB_mac_inst->ho_info.ocs);
}

long get_ocs(int enb_module_id){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	return eNB_mac_inst->ho_info.ocs;
}

// OFF (a3_offset)
void set_off(int enb_module_id,long off){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	eNB_mac_inst->ho_info.off=off;
	printf("OFF for eNB %d is set to %ld\n",enb_module_id,eNB_mac_inst->ho_info.off);
}

long get_off(int enb_module_id){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	return eNB_mac_inst->ho_info.off;
}

// L3 filtering coefficient RSRP
void set_rsrp_filter_coeff(int enb_module_id,float rsrp_filter_coeff){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	eNB_mac_inst->ho_info.rsrp_filter_coeff=rsrp_filter_coeff;
	printf("L3 RSRP filtering coefficient for eNB %d is set to %f\n",enb_module_id,eNB_mac_inst->ho_info.rsrp_filter_coeff);
}

float get_rsrp_filter_coeff(int enb_module_id){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	return eNB_mac_inst->ho_info.rsrp_filter_coeff;
}

// L3 filtering coefficient RSRP
void set_rsrq_filter_coeff(int enb_module_id,float rsrq_filter_coeff){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	eNB_mac_inst->ho_info.rsrq_filter_coeff=rsrq_filter_coeff;
	printf("L3 RSRQ filtering coefficient for eNB %d is set to %f\n",enb_module_id,eNB_mac_inst->ho_info.rsrq_filter_coeff);
}

float get_rsrq_filter_coeff(int enb_module_id){
	eNB_MAC_INST *eNB_mac_inst = get_eNB_mac_inst(enb_module_id);
	return eNB_mac_inst->ho_info.rsrq_filter_coeff;
}
