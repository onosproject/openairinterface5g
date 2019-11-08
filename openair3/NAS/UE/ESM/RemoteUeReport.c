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
#include "esm_ebr.h"
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

 */

  //int _pdn_connectivity_delete(esm_data_t *esm_data, int pid);


/****************************************************************************/
/******************  E X P O R T E D    F U N C T I O N S  ******************/
/****************************************************************************/

int esm_proc_remote_ue_report(nas_user_t *user,int cid, int ebi, unsigned int *pti)
{
    LOG_FUNC_IN;
    int rc = RETURNerror;
    //int pid ;
    int pid = cid - 1;

    esm_data_t *esm_data = user-> esm_data;
    esm_pt_data_t *esm_pt_data = user-> esm_pt_data;

    if (pti != NULL) {

  	    LOG_TRACE(INFO, "ESM-PROC  - Assign new procedure transaction identity ""(cid=%d)", cid);

  	    /* Assign new procedure transaction identity */
  	    *pti = esm_pt_assign(esm_pt_data);
   	    if (*pti == ESM_PT_UNASSIGNED) {
   	    	LOG_TRACE(WARNING, "ESM-PROC  - Failed to assign new procedure transaction identity");
	        LOG_FUNC_RETURN (RETURNerror);
        }


   	    /* Update the PDN connection data */
        rc = _pdn_connectivity_set_pti(esm_data, pid, *pti);
        //ebi = esm_ebr_assign(user->esm_ebr_data, ebi, pid+1, FALSE);

    	if (rc != RETURNok) {
    		LOG_TRACE(WARNING, "ESM-PROC  - Failed to update PDN connection");
    	}
    }
    LOG_FUNC_RETURN(rc);
}

/****************************************************************************
 **                                                                        **
 ** Name:    esm_proc_remote_ue_report_low_layer                       **
 **                                                                        **
 ** Description: Initiates Remote UE Report procedure					**
 **                                                                        **
 **                                                                        **
 **              3GPP TS 24.301, section 6.5.1.2                           **
 **      The Relay UE requests send Remote UE Report message to inform
 **      the network about a new Off network UE.                         **
 **                                                            **
 **                                                                        **
 ** Inputs:  is_standalone: Indicates whether the Remote UE Report     **
 **             procedure is initiated as part of the at-  **
 **             tach procedure                             **
 **      pti:       Procedure transaction identity             **
 **      msg:       Encoded Remote UE Report message   **
 **                  to be sent                                 **
 **      sent_by_ue:    Not used - Always TRUE                     **
 **      Others:    None                                       **
 **                                                                        **
 ** Outputs:     None                                                      **
 **      Return:    RETURNok, RETURNerror                      **
 **      Others:    None                                       **
 **                                                                        **
 ***************************************************************************/
int esm_proc_remote_ue_report_low_layer(nas_user_t *user, int is_standalone, int pti,
                                      OctetString *msg)
{
    LOG_FUNC_IN;
    esm_pt_data_t *esm_pt_data = user->esm_pt_data;
    int rc = RETURNok;

    LOG_TRACE(INFO, "ESM-PROC  - Initiate Remote UE Report (pti=%d)", pti);

    if (is_standalone) {
        emm_sap_t emm_sap;
        emm_esm_data_t *emm_esm = &emm_sap.u.emm_esm.u.data;
        /*
         * Notity EMM that ESM PDU has to be forwarded to lower layers
         */
        emm_sap.primitive = EMMESM_UNITDATA_REQ;
        emm_sap.u.emm_esm.ueid = user->ueid;
        emm_esm->msg.length = msg->length;
        emm_esm->msg.value = msg->value;
        rc = emm_sap_send(user, &emm_sap);
    }
    LOG_FUNC_RETURN(rc);
}
/****************************************************************************
 **                                                                        **
 ** Name:    esm_proc_remote_ue_report_response                       **
 **                                                                        **
 ** Description: Initiates Remote UE Report response procedure					**
 **                                                                        **
 **                                                                        **
 **              3GPP TS 24.301, section 6.5.1.2                           **
 **      Network send this message to UE indicating the completion of
 **      remote ue report procedure.                               **
 **                                                            **
 **                                                                        **
 ** Inputs:  is_standalone: Indicates whether the Remote UE Report     **
 **             procedure is initiated as part of the at-  **
 **             tach procedure                             **
 **      pti:       Procedure transaction identity             **
 **      msg:       Encoded Remote UE Report response message   **
 **                  to be sent                                 **
 **      sent_by_ue:    Not used - Always TRUE                     **
 **      Others:    None                                       **
 **                                                                        **
 ** Outputs:     None                                                      **
 **      Return:    RETURNok, RETURNerror                      **
 **      Others:    None                                       **
 **                                                                        **
 ***************************************************************************/
int esm_proc_remote_ue_report_response(nas_user_t *user, int pti)
{
    LOG_FUNC_IN;
    esm_pt_data_t *esm_pt_data = user->esm_pt_data;
    int rc = RETURNerror;

    LOG_TRACE(INFO, "ESM-PROC  - Remote UE report response");

    /* Get the procedure transaction identity assigned to the PDN connection
     * entry which is still pending in the inactive state */
    rc = esm_pt_set_status(esm_pt_data, pti, ESM_PT_INACTIVE);

    if (pti != ESM_PT_UNASSIGNED) {
        /* Release the procedure transaction identity */
        rc = esm_pt_release(esm_pt_data, pti);
    }

    LOG_FUNC_RETURN(rc);
}

