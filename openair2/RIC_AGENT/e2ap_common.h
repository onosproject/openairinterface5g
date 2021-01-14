/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: LicenseRef-ONF-Member-1.0
 */

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


#ifndef _E2AP_COMMON_H
#define _E2AP_COMMON_H

#include "xer_support.h"

#ifndef E2AP_PORT
#define E2AP_PORT 36423
#endif

extern int e2ap_xer_print;
#define E2AP_XER_PRINT(stream,type,pdu) \
    do { if (e2ap_xer_print) { xer_fprint(stream,type,pdu); } } while (0);

#if defined(ENB_MODE)
#include "common/utils/LOG/log.h"
#define E2AP_ERROR(msg,args...) LOG_E(E2AP,msg,##args)
#define E2AP_INFO(msg,args...)  LOG_I(E2AP,msg,##args)
#define E2AP_WARN(msg,args...)  LOG_W(E2AP,msg,##args)
#define E2AP_DEBUG(msg,args...) LOG_D(E2AP,msg,##args)
#else
#define E2AP_ERROR(msg,args...) do { fprintf(stderr,"[E2AP][E] "msg,##args); } while (0)
#define E2AP_INFO(msg,args...)  do { fprintf(stderr,"[E2AP][I] "msg,##args); } while (0)
#define E2AP_WARN(msg,args...)  do { fprintf(stderr,"[E2AP][W] "msg,##args); } while (0)
#define E2AP_DEBUG(msg,args...) do { fprintf(stderr,"[E2AP][D] "msg,##args); } while (0)
#endif

#define E2AP_FIND_PROTOCOLIE_BY_ID(IE_TYPE, ie, container, IE_ID, mandatory) \
  do {\
    IE_TYPE **ptr; \
    ie = NULL; \
    for (ptr = container->protocolIEs.list.array; \
         ptr < &container->protocolIEs.list.array[container->protocolIEs.list.count]; \
         ptr++) { \
      if((*ptr)->id == IE_ID) { \
        ie = *ptr; \
        break; \
      } \
    } \
    if (mandatory) DevAssert(ie != NULL); \
  } while(0)

#endif /* _E2AP_COMMON_H */
