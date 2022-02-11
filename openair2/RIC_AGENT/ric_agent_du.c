/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: Apache-2.0
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
#include "e2ap_generate_messages.h"
#include "e2ap_handler.h"

#ifdef ENABLE_RAN_SLICING
#include "e2sm_rsm.h"
#endif

du_ric_agent_info_t **du_ric_agent_info;
extern e2_conf_t **e2_conf;

#ifdef ENABLE_RAN_SLICING
#if 0
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

du_ric_agent_info_t *du_ric_agent_get_info(ranid_t ranid, int32_t assoc_id)
{   
    du_ric_agent_info_t *ric;

    ric = du_ric_agent_info[ranid];
    if ( (ric->du_assoc_id == assoc_id) ||
         (ric->du_data_conn_assoc_id == assoc_id) ) 
    {
        return ric;
    }

    return NULL; 
}

static void du_ric_agent_disconnect(du_ric_agent_info_t *ric)
{
    MessageDef *msg;
    sctp_close_association_t *sctp_close_association;

    msg = itti_alloc_new_message(TASK_RIC_AGENT_DU, SCTP_CLOSE_ASSOCIATION);
    sctp_close_association = &msg->ittiMsg.sctp_close_association;
    sctp_close_association->assoc_id = ric->du_assoc_id;

    itti_send_msg_to_task(TASK_SCTP, ric->ranid,msg);

    ric->du_assoc_id = -1; 
}

static int du_ric_agent_handle_sctp_new_association_resp(
        instance_t instance,
        sctp_new_association_resp_t *resp,
        uint8_t **outbuf,
        uint32_t *outlen,
        uint32_t *du_assoc_id)
{
    du_ric_agent_info_t *ric;
    int ret;

    DevAssert(resp != NULL);

    RIC_AGENT_INFO("new sctp assoc resp %d, sctp_state %d for nb %u\n", resp->assoc_id, resp->sctp_state, instance);

    if (resp->sctp_state != SCTP_STATE_ESTABLISHED) {
        if (du_ric_agent_info[instance] != NULL) {
            RIC_AGENT_INFO("resetting RIC connection %u\n", instance);
            //timer_remove(du_ric_agent_info[instance]->e2sm_kpm_timer_id);
            //ric_agent_info[instance]->e2sm_kpm_timer_id = 0;
            du_ric_agent_info[instance]->du_assoc_id = -1;
            timer_setup(5, 0, TASK_RIC_AGENT_DU, instance, TIMER_PERIODIC, NULL, &du_ric_agent_info[instance]->du_ric_connect_timer_id);
        } else {
            RIC_AGENT_ERROR("invalid nb/instance %u in sctp_new_association_resp\n", instance);
            return -1;
        }
        return 0;
    }

    /*
    else if (ric_agent_info[instance]->assoc_id != -1) {
    RIC_AGENT_ERROR("nb %u already associated (%d); ignoring new resp (%d)\n",
            instance,ric_agent_info[instance]->assoc_id,resp->assoc_id);
    }
    */

    RIC_AGENT_INFO("new sctp assoc resp %d for nb %u\n", resp->assoc_id, instance);

    ric = du_ric_agent_get_info(instance, -1);
    if (ric == NULL) {
        RIC_AGENT_ERROR("du_ric_agent_handle_sctp_new_association_resp: ric agent info not found %u\n", instance);
        return -1;
    }

    if (ric->du_assoc_id == -1)
    {
        ric->du_assoc_id = resp->assoc_id;

        timer_remove(ric->du_ric_connect_timer_id);

        /* Send an E2Setup request to RIC. */
        ret = e2ap_generate_e2_setup_request(ric->ranid, outbuf, outlen, e2_conf[0]->e2node_type);
        if (ret) {
            RIC_AGENT_ERROR("failed to generate E2setupRequest; disabling ranid %u!\n",
                ric->ranid);
            du_ric_agent_disconnect(ric);
            return 1;
        }
        *du_assoc_id = ric->du_assoc_id;
    }
    else
    {
        RIC_AGENT_INFO("Data Connection Assoc Id:%d updated\n",resp->assoc_id);
        ric->du_data_conn_assoc_id = resp->assoc_id;

        RIC_AGENT_INFO("e2ap_generate_e2_config_update\n");
        /*Send E2 Configuration Update to RIC */
        ret = e2ap_generate_e2_config_update(ric->ranid, outbuf, outlen, e2_conf[0]->e2node_type);
        if (ret) {
            RIC_AGENT_ERROR("failed to generate E2ConfigUpdate; disabling ranid %u!\n",
                ric->ranid);
            du_ric_agent_disconnect(ric);
            return 1;
        }

        *du_assoc_id = ric->du_data_conn_assoc_id;
    }
    return 0;
}

static int du_ric_agent_connect(ranid_t ranid)
{   
    MessageDef *msg;
    sctp_new_association_req_t *req;
    du_ric_agent_info_t *ric;

    ric = du_ric_agent_get_info(ranid, -1);
    if (ric == NULL) {
        RIC_AGENT_ERROR("ric_agent_connect: ric agent info not found %u\n", ranid);
        return -1;
    }

    msg = itti_alloc_new_message(TASK_RIC_AGENT_DU, SCTP_NEW_ASSOCIATION_REQ);
    req = &msg->ittiMsg.sctp_new_association_req;

    req->ppid = E2AP_SCTP_PPID;
    req->port = e2_conf[ranid]->remote_port;
    req->in_streams = 1;
    req->out_streams = 1;
    req->remote_address.ipv4 = 1;
    strncpy(req->remote_address.ipv4_address, e2_conf[ranid]->remote_ipv4_addr,
            sizeof(req->remote_address.ipv4_address));
    req->remote_address.ipv4_address[sizeof(req->remote_address.ipv4_address)-1] = '\0';
#if DISABLE_SCTP_MULTIHOMING
    // Comment out if testing with loopback
    req->local_address.ipv4 = 1;
    strncpy(req->local_address.ipv4_address, RC.rrc[0]->eth_params_s.my_addr,
            sizeof(req->local_address.ipv4_address));
    req->local_address.ipv4_address[sizeof(req->local_address.ipv4_address)-1] = '\0';
#endif
    req->ulp_cnx_id = 1;

    ric = du_ric_agent_info[ranid];

    RIC_AGENT_INFO("ranid %u connecting to RIC at %s:%u with IP %s\n",
            ranid,req->remote_address.ipv4_address, req->port, req->local_address.ipv4_address);
    itti_send_msg_to_task(TASK_SCTP, ranid, msg);

    return 0;
}

static void du_ric_agent_handle_timer_expiry(
        instance_t instance,
        long timer_id,
        void* arg,
        uint8_t **outbuf,
        uint32_t *outlen)
{
    du_ric_agent_info_t* ric;
    int ret = 0;

    ric = du_ric_agent_info[instance];

    if (timer_id == ric->du_ric_connect_timer_id) 
    {
        du_ric_agent_connect(instance);
    } else {
        RIC_AGENT_INFO("invalid timer expiry instance %u timer_id %ld", instance, timer_id);
    }
    DevAssert(ret == 0);
}

static void du_ric_agent_send_sctp_data(
        du_ric_agent_info_t *ric,
        uint16_t stream,
        uint8_t *buf,
        uint32_t len,
        uint32_t du_assoc_id)
{
    MessageDef *msg;
    sctp_data_req_t *sctp_data_req;

    msg = itti_alloc_new_message(TASK_RIC_AGENT_DU, SCTP_DATA_REQ);
    sctp_data_req = &msg->ittiMsg.sctp_data_req;

    sctp_data_req->assoc_id = du_assoc_id;
    sctp_data_req->stream = stream;
    sctp_data_req->buffer = buf;
    sctp_data_req->buffer_length = len;

    RIC_AGENT_INFO("Send SCTP data, ranid:%u, assoc_id:%d, len:%d\n", ric->ranid, du_assoc_id, len);

    itti_send_msg_to_task(TASK_SCTP, ric->ranid, msg);
}

static void du_ric_agent_handle_sctp_data_ind(
        instance_t instance,
        sctp_data_ind_t *ind,
        uint8_t **outbuf,
        uint32_t *outlen,
        uint32_t *du_assoc_id)
{
    int ret;
    du_ric_agent_info_t *ric;

    DevAssert(ind != NULL);

    ric = du_ric_agent_get_info(instance, ind->assoc_id);
    if (ric == NULL) {
        RIC_AGENT_ERROR("ric_agent_handle_sctp_data_ind: ric agent info not found %u\n", instance);
        return;
    }

    RIC_AGENT_DEBUG("sctp_data_ind instance %u assoc %d", instance, ind->assoc_id);

    du_e2ap_handle_message(ric, ind->stream, ind->buffer, ind->buffer_length, outbuf, outlen, du_assoc_id);

    ret = itti_free(TASK_UNKNOWN, ind->buffer);
    AssertFatal(ret == EXIT_SUCCESS, "failed to free sctp data buf (%d)\n",ret);
}

void *du_ric_agent_task(void *args)
{
  //int nBytes;
  //apiMsg rxApi;

  MessageDef *msg = NULL;
  int res;
  uint16_t i;
  uint8_t *outbuf = NULL;
  uint32_t outlen = 0;
  uint32_t du_assoc_id = 0;

  RIC_AGENT_INFO("starting DU E2 agent task\n");

  e2sm_rsm_init(e2_conf[0]->e2node_type);
  //connectWithRic();

  for (i = 0; i < RC.nb_inst; ++i) 
  {
    if (e2_conf[i]->enabled) 
    {
      timer_setup(5, 0, TASK_RIC_AGENT_DU, i, TIMER_PERIODIC, NULL, &du_ric_agent_info[i]->du_ric_connect_timer_id);
    }
  }

  while (1) 
  {
    itti_receive_msg(TASK_RIC_AGENT_DU, &msg);

    switch (ITTI_MSG_ID(msg)) 
    {
      case SCTP_NEW_ASSOCIATION_IND:
          RIC_AGENT_INFO("Received SCTP_NEW_ASSOCIATION_IND for instance %d\n",
                  ITTI_MESSAGE_GET_INSTANCE(msg));
          break;

      case SCTP_NEW_ASSOCIATION_RESP:
          du_ric_agent_handle_sctp_new_association_resp(
                  ITTI_MESSAGE_GET_INSTANCE(msg),
                  &msg->ittiMsg.sctp_new_association_resp,
                  &outbuf,
                  &outlen,
                  &du_assoc_id);
          break;

      case SCTP_DATA_IND:
          du_ric_agent_handle_sctp_data_ind(
                  ITTI_MESSAGE_GET_INSTANCE(msg),
                  &msg->ittiMsg.sctp_data_ind,
                  &outbuf,
                  &outlen,
                  &du_assoc_id);
          break;

      case TERMINATE_MESSAGE:
          RIC_AGENT_WARN("exiting RIC agent task\n");
          itti_exit_task();
          break;

      case SCTP_CLOSE_ASSOCIATION:
          RIC_AGENT_WARN("sctp connection to RIC closed\n");
          break;

      case TIMER_HAS_EXPIRED:
          du_ric_agent_handle_timer_expiry(
                  ITTI_MESSAGE_GET_INSTANCE(msg),
                  TIMER_HAS_EXPIRED(msg).timer_id,
                  TIMER_HAS_EXPIRED(msg).arg,
                  &outbuf,
                  &outlen);
          break;

      case DU_SLICE_API_RESP:
          RIC_AGENT_INFO("Received DU_SLICE_API_RESP for instance %d\n",
                  ITTI_MESSAGE_GET_INSTANCE(msg));
          du_e2ap_prepare_ric_control_response(
                  du_ric_agent_info[0],
                  &msg->ittiMsg.du_slice_api_resp, 
                  &outbuf, 
                  &outlen,
                  &du_assoc_id);
          break;
 
      default:
          RIC_AGENT_ERROR("unhandled message: %d:%s\n",
                  ITTI_MSG_ID(msg), ITTI_MSG_NAME(msg));
          break;
    }
    
    if (outlen) {
        instance_t instance = ITTI_MESSAGE_GET_INSTANCE(msg);
        du_ric_agent_info_t *ric = du_ric_agent_info[instance];
        //sctp_data_ind_t *ind = &msg->ittiMsg.sctp_data_ind;
        // ric_agent_info_t *ric = ric_agent_get_info(instance, ind->assoc_id);
        AssertFatal(ric != NULL, "ric agent info not found %u\n", instance);
        AssertFatal(du_assoc_id != 0, "Association ID not updated %u\n", du_assoc_id);
        du_ric_agent_send_sctp_data(ric, 0, outbuf, outlen, du_assoc_id);
        outlen = 0;
    }

    res = itti_free(ITTI_MSG_ORIGIN_ID(msg), msg);
    AssertFatal(res == EXIT_SUCCESS, "failed to free msg (%d)!\n",res);
    msg = NULL;
  }
#if 0
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
#endif

  return NULL;
}
#endif
