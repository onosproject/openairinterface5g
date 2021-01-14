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

#ifndef RIC_AGENT_COMMON_H
#define RIC_AGENT_COMMON_H

#include "common/utils/LOG/log.h"

#define RIC_AGENT_ERROR(msg,args...) LOG_E(RIC_AGENT,msg,##args)
#define RIC_AGENT_INFO(msg,args...)  LOG_I(RIC_AGENT,msg,##args)
#define RIC_AGENT_WARN(msg,args...)  LOG_W(RIC_AGENT,msg,##args)
#define RIC_AGENT_DEBUG(msg,args...) LOG_D(RIC_AGENT,msg,##args)

#endif
