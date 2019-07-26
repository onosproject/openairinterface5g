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

/*****************************************************************************
Source      RemoteUereport.c

Version

Date        2019/06/08

Product     NAS stack

Subsystem   EPS Session Management

Author

Description

*****************************************************************************/

#include <stdlib.h> // malloc, free
#include <string.h> // memset, memcpy, memcmp
#include <ctype.h>  // isprint

#include "esm_proc.h"
#include "commonDef.h"
#include "nas_log.h"

#include "esmData.h"
#include "esm_cause.h"
#include "esm_pt.h"
#include "PdnConnectivity.h"

#include "emm_sap.h"

#if defined(ENABLE_ITTI)
# include "assertions.h"
#endif

/****************************************************************************/
/****************  E X T E R N A L    D E F I N I T I O N S  ****************/
/****************************************************************************/

/****************************************************************************/
/*******************  L O C A L    D E F I N I T I O N S  *******************/
/****************************************************************************/

/*
 * --------------------------------------------------------------------------
 *  Internal data handled by the PDN connectivity procedure in the UE
 * --------------------------------------------------------------------------



/****************************************************************************/
/******************  E X P O R T E D    F U N C T I O N S  ******************/
/****************************************************************************/


//int _pdn_connectivity_set_pti(esm_data_t *esm_data, int pid, int pti);
int esm_proc_remote_ue_report(nas_user_t *user,int cid, unsigned int *pti)
{
    LOG_FUNC_IN;
    int rc = RETURNerror;
    int pid = cid - 1;
    esm_data_t *esm_data = user-> esm_data;
    esm_pt_data_t *esm_pt_data = user-> esm_pt_data;

    if (pti != NULL)
    {
  	LOG_TRACE(INFO, "ESM-PROC  - Assign new procedure transaction identity ");
   	/* Assign new procedure transaction identity */
   	*pti = esm_pt_assign(esm_pt_data);

   	if (*pti == ESM_PT_UNASSIGNED) {
	LOG_TRACE(WARNING, "ESM-PROC  - Failed to assign new procedure transaction identity");
	LOG_FUNC_RETURN (RETURNerror);
    	}
    	//static int _pdn_connectivity_set_pti(esm_data_t *esm_data, int pid, int pti);

   	/* Update the PDN connection data */

    	rc = _pdn_connectivity_set_pti(esm_data, pid, *pti);
    	if (rc != RETURNok) {
    		LOG_TRACE(WARNING, "ESM-PROC  - Failed to update PDN connection");
    	}
    }
    LOG_FUNC_RETURN(rc);
}
