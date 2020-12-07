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

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <arpa/inet.h>

#include "queue.h"
#include "tree.h"
#include "assertions.h"
#include "intertask_interface.h"
#include "sctp_eNB_defs.h"
#include "common/config/config_userapi.h"
#include "common/ran_context.h"

#include "ric_agent.h"
#include "ric_agent_common.h"
#include "ric_agent_config.h"
#include "ric_agent_defs.h"
#include "e2ap_generate_messages.h"
#include "e2ap_handler.h"
#include "e2sm_common.h"

extern RAN_CONTEXT_t RC;

ric_ran_function_t **ran_functions = NULL;
unsigned int ran_functions_len = 0;
static unsigned int ran_functions_alloc_len = 0;

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

    DevAssert(ranid < RC.nb_inst);
    ric = RC.ric[ranid];
    if (ric->assoc_id != assoc_id) {
        return NULL;
    }

    return ric;
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
            E2AP_ERROR("subscription delete in reset failed (%ld/%ld); forcing!\n",
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
    char *cc,*tmp = NULL,*tmp2,*tok;
    ric_ran_function_t *func;
    int j;

    ric = ric_agent_get_info(ranid, -1);
    if (ric == NULL) {
        RIC_AGENT_ERROR("ric_agent_connect: ric agent info not found %u\n", ranid);
        return -1;
    }

    if (!ric->functions_enabled) {
        ric->functions_enabled_len = 0;
        ric->functions_enabled = (ric_ran_function_id_t *) \
                                 calloc(ran_functions_len, sizeof(*ric->functions_enabled));
        if (ric->functions_enabled_str && strlen(ric->functions_enabled_str) > 0) {
            cc = strdup(ric->functions_enabled_str);
            tmp2 = cc;
            while ((tok = strtok_r(tmp2, " ", &tmp)) != NULL) {
                tmp2 = NULL;
                func = ric_agent_lookup_ran_function_by_name(tok);
                if (!func) {
                    RIC_AGENT_ERROR("unknown RIC RAN function '%s'; ignoring!\n",tok);
                } else if (!func->enabled) {
                    RIC_AGENT_WARN("RIC RAN function '%s' globally disabled; ignoring\n", tok);
                }
                else {
                    /* Check if already enabled for this NB. */
                    for (j = 0; j < ric->functions_enabled_len; ++j) {
                        if (ric->functions_enabled[j] == func->function_id)
                            break;
                    }
                    if (j == ric->functions_enabled_len) {
                        DevAssert(ric->functions_enabled_len < ran_functions_len);
                        ric->functions_enabled[ric->functions_enabled_len++] = func->function_id;
                    }
                }
            }
            free(cc);
        } else {
            /* Just enable everything. */
            ric->functions_enabled_len = ran_functions_len;
            for (j = 0; j < ran_functions_len; ++j) {
                ric->functions_enabled[j] = ran_functions[j]->function_id;
            }
        }
    }

    msg = itti_alloc_new_message(TASK_RIC_AGENT, SCTP_NEW_ASSOCIATION_REQ);
    req = &msg->ittiMsg.sctp_new_association_req;

    req->ppid = 0;
    req->port = RC.ric[ranid]->remote_port;
    req->in_streams = 1;
    req->out_streams = 1;
    req->remote_address.ipv4 = 1;
    strncpy(req->remote_address.ipv4_address,RC.ric[ranid]->remote_ipv4_addr,
            sizeof(req->remote_address.ipv4_address));
    req->remote_address.ipv4_address[sizeof(req->remote_address.ipv4_address)-1] = '\0';
    req->local_address.ipv4 = 1;
    strncpy(req->local_address.ipv4_address, RC.rrc[0]->eth_params_s.my_addr,
            sizeof(req->local_address.ipv4_address));
    req->local_address.ipv4_address[sizeof(req->local_address.ipv4_address)-1] = '\0';
    req->ulp_cnx_id = 1;

    ric = RC.ric[ranid];
    ric->state = RIC_CONNECTING;

    RIC_AGENT_INFO("ranid %u connecting to RIC at %s:%u with IP %s\n",
            ranid,req->remote_address.ipv4_address, req->port, req->local_address.ipv4_address);
    itti_send_msg_to_task(TASK_SCTP, ranid,msg);

    return 0;
}

void ric_agent_send_sctp_data(
        ric_agent_info_t *ric,
        uint16_t stream,
        uint8_t *buf,
        uint32_t len)
{
    MessageDef *msg;
    sctp_data_req_t *sctp_data_req;

    msg = itti_alloc_new_message(TASK_RIC_AGENT, SCTP_DATA_REQ);
    sctp_data_req = &msg->ittiMsg.sctp_data_req;

    sctp_data_req->assoc_id = ric->assoc_id;
    sctp_data_req->stream = stream;
    sctp_data_req->buffer = buf;
    sctp_data_req->buffer_length = len;

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

    ric->state = RIC_DISCONNECTED;
    ric->assoc_id = -1;
}

static int ric_agent_handle_sctp_new_association_resp(
        instance_t instance,
        sctp_new_association_resp_t *resp)
{
    ric_agent_info_t *ric;
    int ret;
    uint8_t *buf = NULL;
    uint32_t len;

    DevAssert(resp != NULL);

    RIC_AGENT_INFO("new sctp assoc resp %d, sctp_state %d for nb %u\n", resp->assoc_id, resp->sctp_state, instance);

    if (instance >= RC.nb_inst) {
        RIC_AGENT_ERROR("invalid nb/instance %u in sctp_new_association_resp\n", instance);
        return -1;
    } else if (resp->sctp_state != SCTP_STATE_ESTABLISHED) {
        if (RC.ric[instance] != NULL) {
            RC.ric[instance]->assoc_id = -1;
            RIC_AGENT_INFO("resetting RIC connection %u\n", instance);
            timer_remove(RC.ric[instance]->e2sm_kpm_timer_id);
            timer_setup(5, 0, TASK_RIC_AGENT, instance, TIMER_PERIODIC, NULL, &RC.ric[instance]->ric_connect_timer_id);
        } else {
            RIC_AGENT_ERROR("invalid nb/instance %u in sctp_new_association_resp\n", instance);
            return -1;
        }
        return 0;
    }

    /*
    else if (RC.ric[instance]->assoc_id != -1) {
    RIC_AGENT_ERROR("nb %u already associated (%d); ignoring new resp (%d)\n",
            instance,RC.ric[instance]->assoc_id,resp->assoc_id);
    }
    */

    RIC_AGENT_INFO("new sctp assoc resp %d for nb %u\n", resp->assoc_id, instance);

    ric = ric_agent_get_info(instance, -1);
    if (ric == NULL) {
        RIC_AGENT_ERROR("ric_agent_handle_sctp_new_association_resp: ric agent info not found %u\n", instance);
        return -1;
    }
    ric->assoc_id = resp->assoc_id;
    ric->state = RIC_CONNECTED;

    timer_remove(ric->ric_connect_timer_id);

    /* Send an E2Setup request to RIC. */
    ret = e2ap_generate_e2_setup_request(ric, &buf, &len);
    if (ret) {
        RIC_AGENT_ERROR("failed to generate E2setupRequest; disabling ranid %u!\n",
                ric->ranid);
        ric_agent_disconnect(ric);
        ric->state = RIC_DISABLED;
        if (buf)
            free(buf);
        return 1;
    }

    ric_agent_send_sctp_data(ric, 0, buf,len);

    return 0;
}

static void ric_agent_handle_sctp_data_ind(
        instance_t instance,
        sctp_data_ind_t *ind)
{
    int ret;
    ric_agent_info_t *ric;

    DevAssert(ind != NULL);
    DevAssert(instance < RC.nb_inst);

    ric = ric_agent_get_info(instance, ind->assoc_id);
    if (ric == NULL) {
        RIC_AGENT_ERROR("ric_agent_handle_sctp_data_ind: ric agent info not found %u\n", instance);
        return;
    }

    RIC_AGENT_DEBUG("sctp_data_ind instance %u assoc %d", instance, ind->assoc_id);

    e2ap_handle_message(ric, ind->stream, ind->buffer, ind->buffer_length);

    ret = itti_free(TASK_UNKNOWN, ind->buffer);
    AssertFatal(ret == EXIT_SUCCESS, "failed to free sctp data buf (%d)\n",ret);
}

static void ric_agent_handle_timer_expiry(instance_t instance, long timer_id, void* arg) {
    ric_agent_info_t* ric;
    int ret = 0;

    DevAssert(instance < RC.nb_inst);

    ric = RC.ric[instance];

    if (timer_id == ric->ric_connect_timer_id) {
        ric_agent_connect(instance);
    } else {
        ret = e2ap_handle_timer_expiry(ric, timer_id, arg);
    }
    DevAssert(ret == 0);
}

void *ric_agent_task(void *args)
{
    MessageDef *msg = NULL;
    int res;
    uint16_t i;

    if (!ric_agent_is_enabled()) {
        RIC_AGENT_INFO(" *** RIC agent not enabled for any NB; exiting task\n");
        itti_exit_task();
    }

    e2sm_kpm_init();

    RIC_AGENT_INFO("starting RIC agent task\n");
    itti_mark_task_ready(TASK_RIC_AGENT);

    for (i = 0; i < RC.nb_inst; ++i) {
        timer_setup(5, 0, TASK_RIC_AGENT, i, TIMER_PERIODIC, NULL, &RC.ric[i]->ric_connect_timer_id);
    }

    while (1) {
        itti_receive_msg(TASK_RIC_AGENT, &msg);

        switch (ITTI_MSG_ID(msg)) {
            case TERMINATE_MESSAGE:
                RIC_AGENT_WARN("exiting RIC agent task\n");
                itti_exit_task();
                break;
            case SCTP_NEW_ASSOCIATION_RESP:
                ric_agent_handle_sctp_new_association_resp(
                        ITTI_MESSAGE_GET_INSTANCE(msg),
                        &msg->ittiMsg.sctp_new_association_resp);
                break;
            case SCTP_DATA_IND:
                ric_agent_handle_sctp_data_ind(
                        ITTI_MESSAGE_GET_INSTANCE(msg),
                        &msg->ittiMsg.sctp_data_ind);
                break;
            case SCTP_CLOSE_ASSOCIATION:
                RIC_AGENT_WARN("sctp connection to RIC closed\n");
                break;

            case TIMER_HAS_EXPIRED:
                ric_agent_handle_timer_expiry(
                        ITTI_MESSAGE_GET_INSTANCE(msg),
                        TIMER_HAS_EXPIRED(msg).timer_id,
                        TIMER_HAS_EXPIRED(msg).arg);
                break;
            default:
                RIC_AGENT_ERROR("unhandled message: %d:%s\n",
                        ITTI_MSG_ID(msg), ITTI_MSG_NAME(msg));
                break;
        }

        res = itti_free(ITTI_MSG_ORIGIN_ID(msg),msg);
        AssertFatal(res == EXIT_SUCCESS,"failed to free msg (%d)!\n",res);
        msg = NULL;
    }

    return NULL;
}
