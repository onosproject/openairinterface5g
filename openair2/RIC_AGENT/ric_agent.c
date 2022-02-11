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

#include "common/ran_context.h"
#include "ric_agent.h"
#include "e2ap_generate_messages.h"
#include "e2ap_handler.h"
#include "e2sm_kpm.h"

#ifdef ENABLE_RAN_SLICING
#include "e2sm_rsm.h"
#endif
#define DISABLE_SCTP_MULTIHOMING 1

extern RAN_CONTEXT_t RC;

ric_agent_info_t **ric_agent_info;
e2_conf_t **e2_conf;

ric_ran_function_t **ran_functions = NULL;
unsigned int ran_functions_len = 0;
static unsigned int ran_functions_alloc_len = 0;


static void ric_agent_send_sctp_data(
        ric_agent_info_t *ric,
        uint16_t stream,
        uint8_t *buf,
        uint32_t len,
        uint32_t assoc_id);

int ric_agent_register_ran_function(ric_ran_function_t *func)
{
    ric_ran_function_t **tmp, **tmp2;
    ric_ran_function_id_t new_id;

    DevAssert(func != NULL);

    if (ran_functions == NULL || ran_functions_alloc_len == ran_functions_len) {
        tmp = (ric_ran_function_t **)realloc(ran_functions,sizeof(*ran_functions)*(ran_functions_alloc_len+8));
        if (tmp == NULL) {
            RIC_AGENT_ERROR("failed to allocate more memory for ran_function table");
            return -1;
        }
        if (ran_functions && ran_functions != tmp) {
            memcpy(tmp,ran_functions,ran_functions_alloc_len);
            tmp2 = ran_functions;
            ran_functions = tmp;
            if (tmp2) {
                free(tmp2);
            }
        } else {
            ran_functions = tmp;
        }
        ran_functions_alloc_len += 8;
    }

    new_id = ran_functions_len++;
    ran_functions[new_id] = func;
    func->function_id = new_id + 1;

    return 0;
}

ric_ran_function_t *ric_agent_lookup_ran_function(ric_ran_function_id_t function_id)
{
    int index = function_id - 1;
    if (index < 0 || index >= ran_functions_len)
        return NULL;

    return ran_functions[index];
}

ric_ran_function_t *ric_agent_lookup_ran_function_by_name(char *name)
{
    int i;

    for (i = 0; i < ran_functions_len; ++i) {
    if (strcmp(name,ran_functions[i]->name) == 0)
        return ran_functions[i];
    }

    return NULL;
}

ric_subscription_t *ric_agent_lookup_subscription(
        ric_agent_info_t *ric,
        long request_id,
        long instance_id,
        ric_ran_function_id_t function_id)
{
    ric_subscription_t *sub;

    LIST_FOREACH(sub, &ric->subscription_list, subscriptions) {
        if (sub->request_id == request_id
                && sub->instance_id == instance_id
                && sub->function_id == function_id)
            return sub;
    }

    return NULL;
}

ric_agent_info_t *ric_agent_get_info(ranid_t ranid, int32_t assoc_id)
{
    ric_agent_info_t *ric;

    ric = ric_agent_info[ranid];
    //if ( (ric->assoc_id != assoc_id) ||
    //     (ric->data_conn_assoc_id != assoc_id) ) 
    if ( (ric->assoc_id == assoc_id) ||
         (ric->data_conn_assoc_id == assoc_id) ) 
    {
        //return NULL;
        return ric;
    }

    //return ric;
    return NULL;
}

void ric_free_action(ric_action_t *action)
{
    if (action->def_buf)
        free(action->def_buf);
    free(action);
}


void ric_free_subscription(ric_subscription_t *sub)
{
    ric_action_t *action,*next;

    action = LIST_FIRST(&sub->action_list);
    while (action != NULL) {
        next = LIST_NEXT(action,actions);
        ric_free_action(action);
        action = next;
    }

    if (sub->event_trigger.buf)
        free(sub->event_trigger.buf);
    free(sub);
}

/*
 * This must not fail.  But if it must, the only way forward is to
 * terminate the current connection to its RIC, and reestablish.
 */
int ric_agent_reset(ric_agent_info_t *ric)
{
    ric_subscription_t *sub,*subnext;
    int ret;
    long cause,cause_detail;
    ric_ran_function_t *func;

    sub = LIST_FIRST(&ric->subscription_list);

    while (sub != NULL) {
        subnext = LIST_NEXT(sub,subscriptions);
        func = ric_agent_lookup_ran_function(sub->function_id);
        DevAssert(func);
        ret = func->model->handle_subscription_del(ric,sub,0,&cause,&cause_detail);
        if (ret) {
            RIC_AGENT_ERROR("subscription delete in reset failed (%ld/%ld); forcing!\n",
                    cause, cause_detail);
            func->model->handle_subscription_del(ric,sub,1,&cause,&cause_detail);
        }
        sub = subnext;
    }

    LIST_INIT(&ric->subscription_list);

    return 0;
}

static int ric_agent_connect(ranid_t ranid)
{
    MessageDef *msg;
    sctp_new_association_req_t *req;
    ric_agent_info_t *ric;

    ric = ric_agent_get_info(ranid, -1);
    if (ric == NULL) {
        RIC_AGENT_ERROR("ric_agent_connect: ric agent info not found %u\n", ranid);
        return -1;
    }

    msg = itti_alloc_new_message(TASK_RIC_AGENT, SCTP_NEW_ASSOCIATION_REQ);
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

    ric = ric_agent_info[ranid];

    RIC_AGENT_INFO("ranid %u connecting to RIC at %s:%u with IP %s\n",
            ranid,req->remote_address.ipv4_address, req->port, req->local_address.ipv4_address);
    itti_send_msg_to_task(TASK_SCTP, ranid, msg);

    return 0;
}

static void ric_agent_send_sctp_data(
        ric_agent_info_t *ric,
        uint16_t stream,
        uint8_t *buf,
        uint32_t len,
        uint32_t assoc_id)
{
    MessageDef *msg;
    sctp_data_req_t *sctp_data_req;

    msg = itti_alloc_new_message(TASK_RIC_AGENT, SCTP_DATA_REQ);
    sctp_data_req = &msg->ittiMsg.sctp_data_req;

    sctp_data_req->assoc_id = assoc_id;
    sctp_data_req->stream = stream;
    sctp_data_req->buffer = buf;
    sctp_data_req->buffer_length = len;

    RIC_AGENT_INFO("Send SCTP data, ranid:%u, assoc_id:%d, len:%d\n", ric->ranid, assoc_id, len);

    itti_send_msg_to_task(TASK_SCTP, ric->ranid, msg);
}

static void ric_agent_disconnect(ric_agent_info_t *ric)
{
    MessageDef *msg;
    sctp_close_association_t *sctp_close_association;

    msg = itti_alloc_new_message(TASK_RIC_AGENT, SCTP_CLOSE_ASSOCIATION);
    sctp_close_association = &msg->ittiMsg.sctp_close_association;
    sctp_close_association->assoc_id = ric->assoc_id;

    itti_send_msg_to_task(TASK_SCTP, ric->ranid,msg);

    ric->assoc_id = -1;
}

static int ric_agent_handle_sctp_new_association_resp(
        instance_t instance,
        sctp_new_association_resp_t *resp,
        uint8_t **outbuf,
        uint32_t *outlen,
        uint32_t *assoc_id)
{
    ric_agent_info_t *ric;
    int ret;

    DevAssert(resp != NULL);

    RIC_AGENT_INFO("new sctp assoc resp %d, sctp_state %d for nb %u\n", resp->assoc_id, resp->sctp_state, instance);

    if (resp->sctp_state != SCTP_STATE_ESTABLISHED) {
        if (ric_agent_info[instance] != NULL) {
            RIC_AGENT_INFO("resetting RIC connection %u\n", instance);
            timer_remove(ric_agent_info[instance]->e2sm_kpm_timer_id);
            ric_agent_info[instance]->e2sm_kpm_timer_id = 0;
            ric_agent_info[instance]->assoc_id = -1;
            timer_setup(5, 0, TASK_RIC_AGENT, instance, TIMER_PERIODIC, NULL, &ric_agent_info[instance]->ric_connect_timer_id);
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

    ric = ric_agent_get_info(instance, -1);
    if (ric == NULL) {
        RIC_AGENT_ERROR("[%s]: ric agent info not found %u\n", __func__, instance);
        return -1;
    }

    if (ric->assoc_id == -1)
    {
        ric->assoc_id = resp->assoc_id;

        timer_remove(ric->ric_connect_timer_id);

        /* Send an E2Setup request to RIC. */
        ret = e2ap_generate_e2_setup_request(ric->ranid, outbuf, outlen, e2_conf[0]->e2node_type);
        if (ret) {
            RIC_AGENT_ERROR("failed to generate E2setupRequest; disabling ranid %u!\n",
                ric->ranid);
            ric_agent_disconnect(ric);
            return 1;
        }
    
        *assoc_id = ric->assoc_id;
    }
    else
    {
        RIC_AGENT_INFO("Data Connection Assoc Id:%d updated\n",resp->assoc_id);
        ric->data_conn_assoc_id = resp->assoc_id;

        RIC_AGENT_INFO("e2ap_generate_e2_config_update\n");
        /*Send E2 Configuration Update to RIC */
        ret = e2ap_generate_e2_config_update(ric->ranid, outbuf, outlen, e2_conf[0]->e2node_type);
        if (ret) {
            RIC_AGENT_ERROR("failed to generate E2setupRequest; disabling ranid %u!\n",
                ric->ranid);
            ric_agent_disconnect(ric);
            return 1;
        }

        *assoc_id = ric->data_conn_assoc_id;
    }

    return 0;
}

static void ric_agent_handle_sctp_data_ind(
        instance_t instance,
        sctp_data_ind_t *ind,
        uint8_t **outbuf,
        uint32_t *outlen,
        uint32_t *assoc_id)
{
    int ret;
    ric_agent_info_t *ric;

    DevAssert(ind != NULL);

    ric = ric_agent_get_info(instance, ind->assoc_id);
    if (ric == NULL) {
        RIC_AGENT_ERROR("ric_agent_handle_sctp_data_ind: ric agent info not found %u\n", instance);
        return;
    }

    RIC_AGENT_DEBUG("sctp_data_ind instance %u assoc %d", instance, ind->assoc_id);

    e2ap_handle_message(ric, ind->stream, ind->buffer, ind->buffer_length, outbuf, outlen, assoc_id);

    ret = itti_free(TASK_UNKNOWN, ind->buffer);
    AssertFatal(ret == EXIT_SUCCESS, "failed to free sctp data buf (%d)\n",ret);
}

static void ric_agent_handle_timer_expiry(
        instance_t instance,
        long timer_id,
        void* arg,
        uint8_t **outbuf,
        uint32_t *outlen,
        uint32_t *assoc_id)
{
    ric_agent_info_t* ric;
    int ret = 0;

    ric = ric_agent_info[instance];

    if (timer_id == ric->ric_connect_timer_id) {
        ric_agent_connect(instance);
    } else if (timer_id == ric->e2sm_kpm_timer_id) {
        ret = e2ap_handle_timer_expiry(ric, timer_id, arg, outbuf, outlen);
        *assoc_id = ric->data_conn_assoc_id;
    } else if (timer_id == ric->gran_prd_timer_id) {
        ret = e2ap_handle_gp_timer_expiry(ric, timer_id, arg, outbuf, outlen);
    } else {
        RIC_AGENT_INFO("invalid timer expiry instance %u timer_id %ld", instance, timer_id);
    }
    DevAssert(ret == 0);
}

#ifdef ENABLE_RAN_SLICING
static void ric_agent_prepare_ric_ind(
        instance_t instance,
        eventTrigger *CUeventTrigger,
        uint8_t **outbuf,
        uint32_t *outlen,
        uint32_t *assoc_id)
{
    ric_agent_info_t* ric;
    ueStatusInd *ueAttachDetachEvTrigger;
    int ret = 0;

    ric = ric_agent_info[instance];
        
    ueAttachDetachEvTrigger = (ueStatusInd *)CUeventTrigger->eventTriggerBuff;

    ret = e2sm_rsm_ricInd(ric, 
                    ric->e2sm_rsm_function_id, 
                    ric->e2sm_rsm_request_id, 
                    ric->e2sm_rsm_instance_id,
                    CUeventTrigger->eventTriggerType,
                    ueAttachDetachEvTrigger,
                    outbuf,
                    outlen);

    DevAssert(ret == 0);
    *assoc_id = ric->data_conn_assoc_id;
    return;
}
#endif

void *ric_agent_task(void *args)
{
    MessageDef *msg = NULL;
    int res;
    uint16_t i;
    uint8_t *outbuf = NULL;
    uint32_t outlen = 0;
    uint32_t assoc_id = 0;

    RIC_AGENT_INFO("starting CU E2 agent task\n");

    e2sm_kpm_init();

#ifdef ENABLE_RAN_SLICING
    e2sm_rsm_init(e2_conf[0]->e2node_type);
#endif

    for (i = 0; i < RC.nb_inst; ++i) {
        if (e2_conf[i]->enabled) {
            timer_setup(5, 0, TASK_RIC_AGENT, i, TIMER_PERIODIC, NULL, &ric_agent_info[i]->ric_connect_timer_id);
        }
    }

    while (1) {
        itti_receive_msg(TASK_RIC_AGENT, &msg);

        switch (ITTI_MSG_ID(msg)) {
            case SCTP_NEW_ASSOCIATION_IND:
                RIC_AGENT_INFO("Received SCTP_NEW_ASSOCIATION_IND for instance %d\n",
                        ITTI_MESSAGE_GET_INSTANCE(msg));
                break;
            case SCTP_NEW_ASSOCIATION_RESP:
                ric_agent_handle_sctp_new_association_resp(
                        ITTI_MESSAGE_GET_INSTANCE(msg),
                        &msg->ittiMsg.sctp_new_association_resp,
                        &outbuf,
                        &outlen,
                        &assoc_id);
                break;
            case SCTP_DATA_IND:
                ric_agent_handle_sctp_data_ind(
                        ITTI_MESSAGE_GET_INSTANCE(msg),
                        &msg->ittiMsg.sctp_data_ind,
                        &outbuf,
                        &outlen,
                        &assoc_id);
                break;
            case TERMINATE_MESSAGE:
                RIC_AGENT_WARN("exiting RIC agent task\n");
                itti_exit_task();
                break;
            case SCTP_CLOSE_ASSOCIATION:
                RIC_AGENT_WARN("sctp connection to RIC closed\n");
                break;

            case TIMER_HAS_EXPIRED:
                ric_agent_handle_timer_expiry(
                        ITTI_MESSAGE_GET_INSTANCE(msg),
                        TIMER_HAS_EXPIRED(msg).timer_id,
                        TIMER_HAS_EXPIRED(msg).arg,
                        &outbuf,
                        &outlen, 
                        &assoc_id);
                break;
#ifdef ENABLE_RAN_SLICING
            case CU_EVENT_TRIGGER:
                RIC_AGENT_INFO("Received CU_EVENT_TRIGGER for instance %d\n",
                  ITTI_MESSAGE_GET_INSTANCE(msg));
                ric_agent_prepare_ric_ind(
                        ITTI_MESSAGE_GET_INSTANCE(msg),
                        &msg->ittiMsg.cu_event_trigger,
                        &outbuf,
                        &outlen,
                        &assoc_id);
                break;
#endif
            default:
                RIC_AGENT_ERROR("unhandled message: %d:%s\n",
                        ITTI_MSG_ID(msg), ITTI_MSG_NAME(msg));
                break;
        }
        if (outlen) {
            instance_t instance = ITTI_MESSAGE_GET_INSTANCE(msg);
            ric_agent_info_t *ric = ric_agent_info[instance];
            //sctp_data_ind_t *ind = &msg->ittiMsg.sctp_data_ind;
            // ric_agent_info_t *ric = ric_agent_get_info(instance, ind->assoc_id);
            AssertFatal(ric != NULL, "ric agent info not found %u\n", instance);
            AssertFatal(assoc_id != 0, "Association ID not updated %u\n", assoc_id);
            ric_agent_send_sctp_data(ric, 0, outbuf, outlen, assoc_id);
            outlen = 0;
        }

        res = itti_free(ITTI_MSG_ORIGIN_ID(msg), msg);
        AssertFatal(res == EXIT_SUCCESS, "failed to free msg (%d)!\n",res);
        msg = NULL;
    }

    return NULL;
}

#define RIC_CONFIG_STRING_ENABLED "enabled"
#define RIC_CONFIG_STRING_REMOTE_IPV4_ADDR "remote_ipv4_addr"
#define RIC_CONFIG_STRING_REMOTE_PORT "remote_port"

#define RIC_CONFIG_IDX_ENABLED          0
#define RIC_CONFIG_IDX_REMOTE_IPV4_ADDR 1
#define RIC_CONFIG_IDX_REMOTE_PORT      2
#define RIC_CONFIG_IDX_FUNCTIONS_ENABLED 3

#define RIC_PORT 36421

#define RICPARAMS_DESC { \
    { RIC_CONFIG_STRING_ENABLED, \
        "yes/no", 0, strptr:NULL, defstrval:"no", TYPE_STRING, 0 }, \
    { RIC_CONFIG_STRING_REMOTE_IPV4_ADDR, \
        NULL, 0, strptr:NULL, defstrval: "127.0.0.1", TYPE_STRING, 0 }, \
    { RIC_CONFIG_STRING_REMOTE_PORT, \
        NULL, 0, uptr:NULL, defintval:RIC_PORT, TYPE_UINT, 0 }  \
}

void RCconfig_ric_agent(void) 
{
    uint16_t i;
    char buf[16];
    paramdef_t ric_params[] = RICPARAMS_DESC;

    e2_conf = (e2_conf_t **)calloc(256, sizeof(e2_conf_t));

    if (NODE_IS_CU(RC.rrc[0]->node_type))
    {
        ric_agent_info = (ric_agent_info_t **)calloc(250, sizeof(ric_agent_info_t));
    }
    else if (NODE_IS_DU(RC.rrc[0]->node_type))
    {
        du_ric_agent_info = (du_ric_agent_info_t **)calloc(250, sizeof(du_ric_agent_info_t));
    }

    for (i = 0; i < RC.nb_inst; ++i) 
    {
/*      if (!NODE_IS_CU(RC.rrc[i]->node_type)) {
            continue;
        }
*/
        /* Get RIC configuration. */
        snprintf(buf, sizeof(buf), "%s.[%u].RIC", ENB_CONFIG_STRING_ENB_LIST, i);
        config_get(ric_params, sizeof(ric_params)/sizeof(paramdef_t), buf);
        
        if (ric_params[RIC_CONFIG_IDX_ENABLED].strptr != NULL
                && strcmp(*ric_params[RIC_CONFIG_IDX_ENABLED].strptr, "yes") == 0) 
        {
            RIC_AGENT_INFO("NODE[%d] enabled for NB %u\n",RC.rrc[i]->node_type, i);

            e2_conf[i] = (e2_conf_t *)calloc(1,sizeof(e2_conf_t));
            if (NODE_IS_CU(RC.rrc[i]->node_type))
            {
                ric_agent_info[i] = (ric_agent_info_t *)calloc(1, sizeof(ric_agent_info_t));
                ric_agent_info[i]->assoc_id = -1;
                ric_agent_info[i]->data_conn_assoc_id = -1;
            } 
            else if (NODE_IS_DU(RC.rrc[i]->node_type))
            {
                du_ric_agent_info[i] = (du_ric_agent_info_t *)calloc(1, sizeof(du_ric_agent_info_t));
                du_ric_agent_info[i]->du_assoc_id = -1;
                du_ric_agent_info[i]->du_data_conn_assoc_id = -1;
            }

            e2_conf[i]->enabled = 1;
            e2_conf[i]->node_name = strdup(RC.rrc[i]->node_name);
            e2_conf[i]->cell_identity = RC.rrc[i]->configuration.cell_identity;
            e2_conf[i]->mcc = RC.rrc[i]->configuration.mcc[0];
            e2_conf[i]->mnc = RC.rrc[i]->configuration.mnc[0];
            e2_conf[i]->mnc_digit_length = RC.rrc[i]->configuration.mnc_digit_length[0];
            switch (RC.rrc[i]->node_type) {
                case ngran_eNB_CU:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_ENB_CU;
                    break;
                case ngran_ng_eNB_CU:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_NG_ENB_CU;
                    break;
                case ngran_gNB_CU:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_GNB_CU;
                    break;
                case ngran_eNB_DU:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_ENB_DU;
                    break;
                default:
                    break;
            }
            e2_conf[i]->remote_ipv4_addr = strdup(*ric_params[RIC_CONFIG_IDX_REMOTE_IPV4_ADDR].strptr);
            e2_conf[i]->remote_port = *ric_params[RIC_CONFIG_IDX_REMOTE_PORT].uptr;
        }
    }
}
