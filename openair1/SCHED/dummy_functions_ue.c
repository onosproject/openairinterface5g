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

#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "nfapi_interface.h"
#include "nfapi_vnf_interface.h"
#include "nfapi.h"
#include "vendor_ext.h"

#include "nfapi_vnf.h"


int oai_nfapi_hi_dci0_req(nfapi_hi_dci0_request_t *hi_dci0_req) {return(0);}
int oai_nfapi_ul_config_req(nfapi_ul_config_request_t *ul_config_req) {return(0);}
int oai_nfapi_tx_req(nfapi_tx_request_t *tx_req) {return(0);}
int oai_nfapi_dl_config_req(nfapi_dl_config_request_t *dl_config_req) {return(0);}

void init_eNB_afterRU(void) {;}

uint32_t from_earfcn(int eutra_bandP, uint32_t dl_earfcn) { return(0);}

int32_t get_uldl_offset(int eutra_bandP) { return(0);}

