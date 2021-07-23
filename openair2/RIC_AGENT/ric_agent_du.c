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

#include "ric_agent.h"
#include "flexran_agent_mac.h"
#include "openair2/LAYER2/MAC/mac.h"

#ifdef ENABLE_RAN_SLICING
int g_duSocket;
struct sockaddr_in g_RicAddr;
struct sockaddr_in g_duAddr;
socklen_t g_addr_size;

static void connectWithRic(void)
{
  /*Create UDP socket*/
  g_duSocket = socket(PF_INET, SOCK_DGRAM, 0);

  /*Configure settings in address struct*/
  g_duAddr.sin_family = AF_INET;
  g_duAddr.sin_port = htons(7891);
  g_duAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
  memset(g_duAddr.sin_zero, '\0', sizeof g_duAddr.sin_zero);

  /*Configure settings in address struct*/
  g_RicAddr.sin_family = AF_INET;
  g_RicAddr.sin_port = htons(7890);
  g_RicAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
  memset(g_RicAddr.sin_zero, '\0', sizeof g_RicAddr.sin_zero);

  printf("binding socket\n");
  /*Bind socket with address struct*/
  bind(g_duSocket, (struct sockaddr *) &g_duAddr, sizeof(g_duAddr));

  /*Initialize size variable to be used later on*/
  g_addr_size = sizeof g_duAddr;

  return;
}
#endif

#ifdef ENABLE_RAN_SLICING
void *du_ric_agent_task(void *args)
{
  int nBytes;
  apiMsg rxApi;
  connectWithRic();

  while(1)
  {
    /* Recv incoming pkts from RIC */
    nBytes = recvfrom(g_duSocket,&rxApi,sizeof(apiMsg),0,NULL, NULL);
    LOG_I(MAC,"Received %d bytes from RIC\n",nBytes);
    if (nBytes > 0)
    {
      handle_slicing_api_req(&rxApi);
    }
  }
  return NULL;
}
#endif
