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

#include "pdcp_ue_manager.h"

/* from OAI */
#include "pdcp.h"
#include "targets/RT/USER/lte-softmodem.h"

#define TODO do { \
    printf("%s:%d:%s: todo\n", __FILE__, __LINE__, __FUNCTION__); \
    exit(1); \
  } while (0)

static pdcp_ue_manager_t *pdcp_ue_manager;

/* necessary globals for OAI, not used internally */
hash_table_t  *pdcp_coll_p;
static uint64_t pdcp_optmask;

/****************************************************************************/
/* rlc_data_req queue - begin                                               */
/****************************************************************************/

#include <pthread.h>

/* New PDCP and RLC both use "big locks". In some cases a thread may do
 * lock(rlc) followed by lock(pdcp) (typically when running 'rx_sdu').
 * Another thread may first do lock(pdcp) and then lock(rlc) (typically
 * the GTP module calls 'pdcp_data_req' that, in a previous implementation
 * was indirectly calling 'rlc_data_req' which does lock(rlc)).
 * To avoid the resulting deadlock it is enough to ensure that a call
 * to lock(pdcp) will never be followed by a call to lock(rlc). So,
 * here we chose to have a separate thread that deals with rlc_data_req,
 * out of the PDCP lock. Other solutions may be possible.
 * So instead of calling 'rlc_data_req' directly we have a queue and a
 * separate thread emptying it.
 */

typedef struct {
  protocol_ctxt_t ctxt_pP;
  srb_flag_t      srb_flagP;
  MBMS_flag_t     MBMS_flagP;
  rb_id_t         rb_idP;
  mui_t           muiP;
  confirm_t       confirmP;
  sdu_size_t      sdu_sizeP;
  mem_block_t     *sdu_pP;
} rlc_data_req_queue_item;

#define RLC_DATA_REQ_QUEUE_SIZE 10000

typedef struct {
  rlc_data_req_queue_item q[RLC_DATA_REQ_QUEUE_SIZE];
  volatile int start;
  volatile int length;
  pthread_mutex_t m;
  pthread_cond_t c;
} rlc_data_req_queue;

static rlc_data_req_queue q;

static void *rlc_data_req_thread(void *_)
{
  int i;

  while (1) {
    if (pthread_mutex_lock(&q.m) != 0) abort();
    while (q.length == 0)
      if (pthread_cond_wait(&q.c, &q.m) != 0) abort();
    i = q.start;
    if (pthread_mutex_unlock(&q.m) != 0) abort();

    rlc_data_req(&q.q[i].ctxt_pP,
                 q.q[i].srb_flagP,
                 q.q[i].MBMS_flagP,
                 q.q[i].rb_idP,
                 q.q[i].muiP,
                 q.q[i].confirmP,
                 q.q[i].sdu_sizeP,
                 q.q[i].sdu_pP,
                 NULL,
                 NULL);

    if (pthread_mutex_lock(&q.m) != 0) abort();

    q.length--;
    q.start = (q.start + 1) % RLC_DATA_REQ_QUEUE_SIZE;

    if (pthread_cond_signal(&q.c) != 0) abort();
    if (pthread_mutex_unlock(&q.m) != 0) abort();
  }
}

static void init_rlc_data_req_queue(void)
{
  pthread_t t;

  pthread_mutex_init(&q.m, NULL);
  pthread_cond_init(&q.c, NULL);

  if (pthread_create(&t, NULL, rlc_data_req_thread, NULL) != 0) {
    LOG_E(PDCP, "%s:%d:%s: fatal\n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }
}

static void enqueue_rlc_data_req(const protocol_ctxt_t *const ctxt_pP,
                                 const srb_flag_t   srb_flagP,
                                 const MBMS_flag_t  MBMS_flagP,
                                 const rb_id_t      rb_idP,
                                 const mui_t        muiP,
                                 confirm_t    confirmP,
                                 sdu_size_t   sdu_sizeP,
                                 mem_block_t *sdu_pP,
                                 void *_unused1, void *_unused2)
{
  int i;
  int logged = 0;

  if (pthread_mutex_lock(&q.m) != 0) abort();
  while (q.length == RLC_DATA_REQ_QUEUE_SIZE) {
    if (!logged) {
      logged = 1;
      LOG_W(PDCP, "%s: rlc_data_req queue is full\n", __FUNCTION__);
    }
    if (pthread_cond_wait(&q.c, &q.m) != 0) abort();
  }

  i = (q.start + q.length) % RLC_DATA_REQ_QUEUE_SIZE;
  q.length++;

  q.q[i].ctxt_pP    = *ctxt_pP;
  q.q[i].srb_flagP  = srb_flagP;
  q.q[i].MBMS_flagP = MBMS_flagP;
  q.q[i].rb_idP     = rb_idP;
  q.q[i].muiP       = muiP;
  q.q[i].confirmP   = confirmP;
  q.q[i].sdu_sizeP  = sdu_sizeP;
  q.q[i].sdu_pP     = sdu_pP;

  if (pthread_cond_signal(&q.c) != 0) abort();
  if (pthread_mutex_unlock(&q.m) != 0) abort();
}

/****************************************************************************/
/* rlc_data_req queue - end                                                 */
/****************************************************************************/

void pdcp_layer_init(void)
{
  /* be sure to initialize only once */
  static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
  static int initialized = 0;
  if (pthread_mutex_lock(&m) != 0) abort();
  if (initialized) {
    if (pthread_mutex_unlock(&m) != 0) abort();
    return;
  }
  initialized = 1;
  if (pthread_mutex_unlock(&m) != 0) abort();

  pdcp_ue_manager = new_pdcp_ue_manager(1);
  init_rlc_data_req_queue();
}

uint64_t pdcp_module_init(uint64_t _pdcp_optmask)
{
  pdcp_optmask = _pdcp_optmask;
  return pdcp_optmask;
}

static void deliver_sdu_drb(void *_ue, pdcp_entity_t *entity,
                            char *buf, int size)
{
  pdcp_ue_t *ue = _ue;
  MessageDef  *message_p;
  uint8_t     *gtpu_buffer_p;
  int rb_id;
  int i;

  for (i = 0; i < 5; i++) {
    if (entity == ue->drb[i]) {
      rb_id = i+1;
      goto rb_found;
    }
  }

  LOG_E(PDCP, "%s:%d:%s: fatal, no RB found for ue %d\n",
        __FILE__, __LINE__, __FUNCTION__, ue->rnti);
  exit(1);

rb_found:
  gtpu_buffer_p = itti_malloc(TASK_PDCP_ENB, TASK_GTPV1_U,
                              size + GTPU_HEADER_OVERHEAD_MAX);
  AssertFatal(gtpu_buffer_p != NULL, "OUT OF MEMORY");
  memcpy(&gtpu_buffer_p[GTPU_HEADER_OVERHEAD_MAX], buf, size);
  message_p = itti_alloc_new_message(TASK_PDCP_ENB, GTPV1U_ENB_TUNNEL_DATA_REQ);
  AssertFatal(message_p != NULL, "OUT OF MEMORY");
  GTPV1U_ENB_TUNNEL_DATA_REQ(message_p).buffer       = gtpu_buffer_p;
  GTPV1U_ENB_TUNNEL_DATA_REQ(message_p).length       = size;
  GTPV1U_ENB_TUNNEL_DATA_REQ(message_p).offset       = GTPU_HEADER_OVERHEAD_MAX;
  GTPV1U_ENB_TUNNEL_DATA_REQ(message_p).rnti         = ue->rnti;
  GTPV1U_ENB_TUNNEL_DATA_REQ(message_p).rab_id       = rb_id + 4;
printf("!!!!!!! deliver_sdu_drb (drb %d) sending message to gtp size %d: ", rb_id, size);
//for (i = 0; i < size; i++) printf(" %2.2x", (unsigned char)buf[i]);
printf("\n");
  itti_send_msg_to_task(TASK_GTPV1_U, INSTANCE_DEFAULT, message_p);
}

static void deliver_pdu_drb(void *_ue, pdcp_entity_t *entity,
                            char *buf, int size, int sdu_id)
{
  pdcp_ue_t *ue = _ue;
  int rb_id;
  protocol_ctxt_t ctxt;
  int i;
  mem_block_t *memblock;

  for (i = 0; i < 5; i++) {
    if (entity == ue->drb[i]) {
      rb_id = i+1;
      goto rb_found;
    }
  }

  LOG_E(PDCP, "%s:%d:%s: fatal, no RB found for ue %d\n",
        __FILE__, __LINE__, __FUNCTION__, ue->rnti);
  exit(1);

rb_found:
  ctxt.module_id = 0;
  ctxt.enb_flag = 1;
  ctxt.instance = 0;
  ctxt.frame = 0;
  ctxt.subframe = 0;
  ctxt.eNB_index = 0;
  ctxt.configured = 1;
  ctxt.brOption = 0;

  ctxt.rnti = ue->rnti;

  memblock = get_free_mem_block(size, __FUNCTION__);
  memcpy(memblock->data, buf, size);

printf("!!!!!!! deliver_pdu_drb (srb %d) calling rlc_data_req size %d: ", rb_id, size);
//for (i = 0; i < size; i++) printf(" %2.2x", (unsigned char)memblock->data[i]);
printf("\n");
  enqueue_rlc_data_req(&ctxt, 0, MBMS_FLAG_NO, rb_id, sdu_id, 0, size, memblock, NULL, NULL);
}

static void deliver_sdu_srb(void *_ue, pdcp_entity_t *entity,
                            char *buf, int size)
{
  pdcp_ue_t *ue = _ue;
  int rb_id;
  protocol_ctxt_t ctxt;
  int i;

  for (i = 0; i < 2; i++) {
    if (entity == ue->srb[i]) {
      rb_id = i+1;
      goto rb_found;
    }
  }

  LOG_E(PDCP, "%s:%d:%s: fatal, no RB found for ue %d\n",
        __FILE__, __LINE__, __FUNCTION__, ue->rnti);
  exit(1);

rb_found:
  ctxt.module_id = 0;
  ctxt.enb_flag = 1;
  ctxt.instance = 0;
  ctxt.frame = 0;
  ctxt.subframe = 0;
  ctxt.eNB_index = 0;
  ctxt.configured = 1;
  ctxt.brOption = 0;

  ctxt.rnti = ue->rnti;

printf("!!!!!!! deliver_sdu_srb (srb %d) calling rrc_data_ind size %d: ", rb_id, size);
//for (i = 0; i < size; i++) printf(" %2.2x", (unsigned char)buf[i]);
printf("\n");
  rrc_data_ind(&ctxt, rb_id, size, (unsigned char *)buf);
}

static void deliver_pdu_srb(void *_ue, pdcp_entity_t *entity,
                            char *buf, int size, int sdu_id)
{
  pdcp_ue_t *ue = _ue;
  int rb_id;
  protocol_ctxt_t ctxt;
  int i;
  mem_block_t *memblock;

  for (i = 0; i < 2; i++) {
    if (entity == ue->srb[i]) {
      rb_id = i+1;
      goto rb_found;
    }
  }

  LOG_E(PDCP, "%s:%d:%s: fatal, no RB found for ue %d\n",
        __FILE__, __LINE__, __FUNCTION__, ue->rnti);
  exit(1);

rb_found:
  ctxt.module_id = 0;
  ctxt.enb_flag = 1;
  ctxt.instance = 0;
  ctxt.frame = 0;
  ctxt.subframe = 0;
  ctxt.eNB_index = 0;
  ctxt.configured = 1;
  ctxt.brOption = 0;

  ctxt.rnti = ue->rnti;

  memblock = get_free_mem_block(size, __FUNCTION__);
  memcpy(memblock->data, buf, size);

printf("!!!!!!! deliver_pdu_srb (srb %d) calling rlc_data_req size %d: ", rb_id, size);
//for (i = 0; i < size; i++) printf(" %2.2x", (unsigned char)memblock->data[i]);
printf("\n");
  enqueue_rlc_data_req(&ctxt, 1, MBMS_FLAG_NO, rb_id, sdu_id, 0, size, memblock, NULL, NULL);
}

boolean_t pdcp_data_ind(
  const protocol_ctxt_t *const  ctxt_pP,
  const srb_flag_t srb_flagP,
  const MBMS_flag_t MBMS_flagP,
  const rb_id_t rb_id,
  const sdu_size_t sdu_buffer_size,
  mem_block_t *const sdu_buffer)
{
  pdcp_ue_t *ue;
  pdcp_entity_t *rb;
  int rnti = ctxt_pP->rnti;

  if (ctxt_pP->module_id != 0 ||
      ctxt_pP->enb_flag != 1 ||
      ctxt_pP->instance != 0 ||
      ctxt_pP->eNB_index != 0 ||
      ctxt_pP->configured != 1 ||
      ctxt_pP->brOption != 0) {
    LOG_E(PDCP, "%s:%d:%s: fatal\n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }

  if (ctxt_pP->enb_flag)
    T(T_ENB_PDCP_UL, T_INT(ctxt_pP->module_id), T_INT(rnti),
      T_INT(rb_id), T_INT(sdu_buffer_size));

  pdcp_manager_lock(pdcp_ue_manager);
  ue = pdcp_manager_get_ue(pdcp_ue_manager, rnti);

  if (srb_flagP == 1) {
    if (rb_id < 1 || rb_id > 2)
      rb = NULL;
    else
      rb = ue->srb[rb_id - 1];
  } else {
    if (rb_id < 1 || rb_id > 5)
      rb = NULL;
    else
      rb = ue->drb[rb_id - 1];
  }

  if (rb != NULL) {
    rb->recv_pdu(rb, (char *)sdu_buffer->data, sdu_buffer_size);
  } else {
    LOG_E(PDCP, "%s:%d:%s: fatal: no RB found (rb_id %d, srb_flag %d)\n",
          __FILE__, __LINE__, __FUNCTION__, rb_id, srb_flagP);
    exit(1);
  }

  pdcp_manager_unlock(pdcp_ue_manager);

  free_mem_block(sdu_buffer, __FUNCTION__);

  return 1;
}

void pdcp_run(const protocol_ctxt_t *const  ctxt_pP)
{
  MessageDef      *msg_p;
  int             result;
  protocol_ctxt_t ctxt;

  while (1) {
    itti_poll_msg(ctxt_pP->enb_flag ? TASK_PDCP_ENB : TASK_PDCP_UE, &msg_p);
    if (msg_p == NULL)
      break;
    switch (ITTI_MSG_ID(msg_p)) {
    case RRC_DCCH_DATA_REQ:
      PROTOCOL_CTXT_SET_BY_MODULE_ID(
          &ctxt,
          RRC_DCCH_DATA_REQ(msg_p).module_id,
          RRC_DCCH_DATA_REQ(msg_p).enb_flag,
          RRC_DCCH_DATA_REQ(msg_p).rnti,
          RRC_DCCH_DATA_REQ(msg_p).frame,
          0,
          RRC_DCCH_DATA_REQ(msg_p).eNB_index);
      result = pdcp_data_req(&ctxt,
                             SRB_FLAG_YES,
                             RRC_DCCH_DATA_REQ(msg_p).rb_id,
                             RRC_DCCH_DATA_REQ(msg_p).muip,
                             RRC_DCCH_DATA_REQ(msg_p).confirmp,
                             RRC_DCCH_DATA_REQ(msg_p).sdu_size,
                             RRC_DCCH_DATA_REQ(msg_p).sdu_p,
                             RRC_DCCH_DATA_REQ(msg_p).mode,
                             NULL, NULL);

      if (result != TRUE)
        LOG_E(PDCP, "PDCP data request failed!\n");
      result = itti_free(ITTI_MSG_ORIGIN_ID(msg_p), RRC_DCCH_DATA_REQ(msg_p).sdu_p);
      AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
      break;
    default:
      LOG_E(PDCP, "Received unexpected message %s\n", ITTI_MSG_NAME(msg_p));
      break;
    }
  }
}

static void add_srb(int rnti, struct LTE_SRB_ToAddMod *s)
{
  pdcp_entity_t *pdcp_srb;
  pdcp_ue_t *ue;

  int srb_id = s->srb_Identity;

printf("\n\n################# add srb %d\n\n\n", srb_id);

  if (srb_id != 1 && srb_id != 2) {
    LOG_E(PDCP, "%s:%d:%s: fatal, bad srb id %d\n",
          __FILE__, __LINE__, __FUNCTION__, srb_id);
    exit(1);
  }

  pdcp_manager_lock(pdcp_ue_manager);
  ue = pdcp_manager_get_ue(pdcp_ue_manager, rnti);
  if (ue->srb[srb_id-1] != NULL) {
    LOG_D(PDCP, "%s:%d:%s: warning SRB %d already exist for ue %d, do nothing\n",
          __FILE__, __LINE__, __FUNCTION__, srb_id, rnti);
  } else {
    pdcp_srb = new_pdcp_entity_srb(srb_id, deliver_sdu_srb, ue, deliver_pdu_srb, ue);
    pdcp_ue_add_srb_pdcp_entity(ue, srb_id, pdcp_srb);

    LOG_D(PDCP, "%s:%d:%s: added srb %d to ue %d\n",
          __FILE__, __LINE__, __FUNCTION__, srb_id, rnti);
  }
  pdcp_manager_unlock(pdcp_ue_manager);
}

static void add_drb_am(int rnti, struct LTE_DRB_ToAddMod *s)
{
  pdcp_entity_t *pdcp_drb;
  pdcp_ue_t *ue;

  int drb_id = s->drb_Identity;

printf("\n\n################# add drb %d\n\n\n", drb_id);

  if (drb_id != 1) {
    LOG_E(PDCP, "%s:%d:%s: fatal, bad drb id %d\n",
          __FILE__, __LINE__, __FUNCTION__, drb_id);
    exit(1);
  }

  pdcp_manager_lock(pdcp_ue_manager);
  ue = pdcp_manager_get_ue(pdcp_ue_manager, rnti);
  if (ue->drb[drb_id-1] != NULL) {
    LOG_D(PDCP, "%s:%d:%s: warning SRB %d already exist for ue %d, do nothing\n",
          __FILE__, __LINE__, __FUNCTION__, drb_id, rnti);
  } else {
    pdcp_drb = new_pdcp_entity_drb_am(drb_id, deliver_sdu_drb, ue, deliver_pdu_drb, ue);
    pdcp_ue_add_drb_pdcp_entity(ue, drb_id, pdcp_drb);

    LOG_D(PDCP, "%s:%d:%s: added drb %d to ue %d\n",
          __FILE__, __LINE__, __FUNCTION__, drb_id, rnti);
  }
  pdcp_manager_unlock(pdcp_ue_manager);
}

static void add_drb(int rnti, struct LTE_DRB_ToAddMod *s)
{
  switch (s->rlc_Config->present) {
  case LTE_RLC_Config_PR_am:
    add_drb_am(rnti, s);
    break;
  case LTE_RLC_Config_PR_um_Bi_Directional:
    //add_drb_um(rnti, s);
    TODO;
    break;
  default:
    LOG_E(PDCP, "%s:%d:%s: fatal: unhandled DRB type\n",
          __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }
}

boolean_t rrc_pdcp_config_asn1_req(
  const protocol_ctxt_t *const  ctxt_pP,
  LTE_SRB_ToAddModList_t  *const srb2add_list,
  LTE_DRB_ToAddModList_t  *const drb2add_list,
  LTE_DRB_ToReleaseList_t *const drb2release_list,
  const uint8_t                   security_modeP,
  uint8_t                  *const kRRCenc,
  uint8_t                  *const kRRCint,
  uint8_t                  *const kUPenc
#if (LTE_RRC_VERSION >= MAKE_VERSION(9, 0, 0))
  ,LTE_PMCH_InfoList_r9_t  *pmch_InfoList_r9
#endif
  ,rb_id_t                 *const defaultDRB)
{
  int rnti = ctxt_pP->rnti;
  int i;

  if (ctxt_pP->enb_flag != 1 ||
      ctxt_pP->module_id != 0 ||
      ctxt_pP->instance != 0 ||
      ctxt_pP->eNB_index != 0 ||
      //ctxt_pP->configured != 2 ||
      //srb2add_list == NULL ||
      //drb2add_list != NULL ||
      drb2release_list != NULL ||
      security_modeP != 255 ||
      //kRRCenc != NULL ||
      //kRRCint != NULL ||
      //kUPenc != NULL ||
      pmch_InfoList_r9 != NULL ||
      defaultDRB != NULL) {
    TODO;
  }

  if (srb2add_list != NULL) {
    for (i = 0; i < srb2add_list->list.count; i++) {
      add_srb(rnti, srb2add_list->list.array[i]);
    }
  }

  if (drb2add_list != NULL) {
    for (i = 0; i < drb2add_list->list.count; i++) {
      add_drb(rnti, drb2add_list->list.array[i]);
    }
  }

  /* update security */
  if (kRRCint != NULL) {
    /* todo */
  }

  free(kRRCenc);
  free(kRRCint);
  free(kUPenc);

  return 0;
}

uint64_t get_pdcp_optmask(void)
{
  TODO;
}

boolean_t pdcp_remove_UE(
  const protocol_ctxt_t *const  ctxt_pP)
{
  LOG_D(RLC, "%s:%d:%s: remove UE %d\n", __FILE__, __LINE__, __FUNCTION__, ctxt_pP->rnti);
  pdcp_manager_lock(pdcp_ue_manager);
  pdcp_manager_remove_ue(pdcp_ue_manager, ctxt_pP->rnti);
  pdcp_manager_unlock(pdcp_ue_manager);

  return 1;
}

void pdcp_config_set_security(const protocol_ctxt_t* const  ctxt_pP, pdcp_t *pdcp_pP, rb_id_t rb_id,
                              uint16_t lc_idP, uint8_t security_modeP, uint8_t *kRRCenc_pP, uint8_t *kRRCint_pP, uint8_t *kUPenc_pP)
{
  pdcp_ue_t *ue;
  pdcp_entity_t *rb;
  int rnti = ctxt_pP->rnti;

  if (ctxt_pP->module_id != 0 ||
      ctxt_pP->enb_flag != 1 ||
      ctxt_pP->instance != 0 ||
      ctxt_pP->eNB_index != 0) {
    LOG_E(PDCP, "%s:%d:%s: fatal\n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }

  pdcp_manager_lock(pdcp_ue_manager);

  ue = pdcp_manager_get_ue(pdcp_ue_manager, rnti);

  if (rb_id < 1 || rb_id > 2)
    rb = NULL;
  else
    rb = ue->srb[rb_id - 1];

  if (rb == NULL) {
    LOG_E(PDCP, "%s:%d:%s: fatal: no SRB found (rb_id %d)\n",
          __FILE__, __LINE__, __FUNCTION__, rb_id);
    exit(1);
  }

  if (kRRCint_pP != NULL)
    rb->set_integrity_key(rb, (char *)kRRCint_pP);

  pdcp_manager_unlock(pdcp_ue_manager);

  free(kRRCenc_pP);
  free(kRRCint_pP);
  free(kUPenc_pP);
}

static boolean_t pdcp_data_req_srb(
  protocol_ctxt_t  *ctxt_pP,
  const rb_id_t rb_id,
  const mui_t muiP,
  const confirm_t confirmP,
  const sdu_size_t sdu_buffer_size,
  unsigned char *const sdu_buffer)
{
  pdcp_ue_t *ue;
  pdcp_entity_t *rb;
  int rnti = ctxt_pP->rnti;

  if (ctxt_pP->module_id != 0 ||
      ctxt_pP->enb_flag != 1 ||
      ctxt_pP->instance != 0 ||
      ctxt_pP->eNB_index != 0 /*||
      ctxt_pP->configured != 1 ||
      ctxt_pP->brOption != 0*/) {
    LOG_E(PDCP, "%s:%d:%s: fatal\n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }

  pdcp_manager_lock(pdcp_ue_manager);

  ue = pdcp_manager_get_ue(pdcp_ue_manager, rnti);

  if (rb_id < 1 || rb_id > 2)
    rb = NULL;
  else
    rb = ue->srb[rb_id - 1];

  if (rb == NULL) {
    LOG_E(PDCP, "%s:%d:%s: fatal: no SRB found (rb_id %d)\n",
          __FILE__, __LINE__, __FUNCTION__, rb_id);
    exit(1);
  }

  rb->recv_sdu(rb, (char *)sdu_buffer, sdu_buffer_size, muiP);

  pdcp_manager_unlock(pdcp_ue_manager);

  return 1;
}

static boolean_t pdcp_data_req_drb(
  protocol_ctxt_t  *ctxt_pP,
  const rb_id_t rb_id,
  const mui_t muiP,
  const confirm_t confirmP,
  const sdu_size_t sdu_buffer_size,
  unsigned char *const sdu_buffer)
{
  pdcp_ue_t *ue;
  pdcp_entity_t *rb;
  int rnti = ctxt_pP->rnti;

  if (ctxt_pP->module_id != 0 ||
      ctxt_pP->enb_flag != 1 ||
      ctxt_pP->instance != 0 ||
      ctxt_pP->eNB_index != 0 /*||
      ctxt_pP->configured != 1 ||
      ctxt_pP->brOption != 0*/) {
    LOG_E(PDCP, "%s:%d:%s: fatal\n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }

  pdcp_manager_lock(pdcp_ue_manager);

  ue = pdcp_manager_get_ue(pdcp_ue_manager, rnti);

  if (rb_id < 1 || rb_id > 5)
    rb = NULL;
  else
    rb = ue->drb[rb_id - 1];

  if (rb == NULL) {
    LOG_E(PDCP, "%s:%d:%s: fatal: no DRB found (rb_id %d)\n",
          __FILE__, __LINE__, __FUNCTION__, rb_id);
    exit(1);
  }

  rb->recv_sdu(rb, (char *)sdu_buffer, sdu_buffer_size, muiP);

  pdcp_manager_unlock(pdcp_ue_manager);

  return 1;
}

boolean_t pdcp_data_req(
  protocol_ctxt_t  *ctxt_pP,
  const srb_flag_t srb_flagP,
  const rb_id_t rb_id,
  const mui_t muiP,
  const confirm_t confirmP,
  const sdu_size_t sdu_buffer_size,
  unsigned char *const sdu_buffer,
  const pdcp_transmission_mode_t mode
#if (LTE_RRC_VERSION >= MAKE_VERSION(14, 0, 0))
  ,const uint32_t *const sourceL2Id
  ,const uint32_t *const destinationL2Id
#endif
  )
{
  if (srb_flagP)
    return pdcp_data_req_srb(ctxt_pP, rb_id, muiP, confirmP, sdu_buffer_size,
                             sdu_buffer);
  return pdcp_data_req_drb(ctxt_pP, rb_id, muiP, confirmP, sdu_buffer_size,
                           sdu_buffer);
}

void pdcp_set_pdcp_data_ind_func(pdcp_data_ind_func_t pdcp_data_ind)
{
  /* nothing to do */
}

void pdcp_set_rlc_data_req_func(send_rlc_data_req_func_t send_rlc_data_req)
{
  /* nothing to do */
}
