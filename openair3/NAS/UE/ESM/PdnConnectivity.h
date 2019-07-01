/*
 * PdnConnectivity.h
 *
 *  Created on: Jun 14, 2019
 *      Author: root
 */

#ifndef OPENAIR3_NAS_UE_ESM_PDNCONNECTIVITY_H_
#define OPENAIR3_NAS_UE_ESM_PDNCONNECTIVITY_H_


#include <stdio.h>
#include "esmData.h"
#include "esm_proc.h"



int _pdn_connectivity_create(esm_data_t *esm_data, int pid, const OctetString *apn,
                                    esm_proc_pdn_type_t pdn_type, int is_emergency);
int _pdn_connectivity_update(esm_data_t *esm_data, int pid, const OctetString *apn,
                                    esm_proc_pdn_type_t pdn_type, const OctetString *pdn_addr, int esm_cause);
int _pdn_connectivity_delete(esm_data_t *esm_data, int pid);

int _pdn_connectivity_set_pti(esm_data_t *esm_data, int pid, int pti);
int _pdn_connectivity_find_apn(esm_data_t *esm_data, const OctetString *apn);
int _pdn_connectivity_find_pdn(esm_data_t * esm_data, const OctetString *apn,
                                      esm_proc_pdn_type_t pdn_type);

#endif /* OPENAIR3_NAS_UE_ESM_PDNCONNECTIVITY_H_ */
