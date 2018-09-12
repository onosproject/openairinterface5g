
#include <stdio.h>
#include <errno.h>
#include "mme_config.h"
#include "assertions.h"
#include "intertask_interface.h"
#include "msc.h"
#include "gtpv1u.h"
#include "gtpv1u_eNB_defs.h"
#include "gtpv1_u_messages_types.h"
#include "udp_eNB_task.h"
#include "common/utils/LOG/log.h"
#include "COMMON/platform_types.h"
#include "COMMON/platform_constants.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/ran_context.h"
#include "gtpv1u_eNB_defs.h"

int noS1_create_s1u_tunnel( const instance_t instanceP,
                            const gtpv1u_enb_create_tunnel_req_t *create_tunnel_req,
                            gtpv1u_enb_create_tunnel_resp_t *create_tunnel_resp_pP) {
  LOG_D(GTPU, "Start create tunnels for RNTI %x, num_tunnels %d, sgw_S1u_teid %d, eps_bearer_id %d \n",
        create_tunnel_req->rnti,
        create_tunnel_req->num_tunnels,
        create_tunnel_req->sgw_S1u_teid[0],
        create_tunnel_req->eps_bearer_id[0]);
  /*
    on a pas l'adresse IP du UE, mais on l'aura dans le premier paquet sortant
    Sinon, il faut la passer dans le create tunnel
  */
  return 0;
}

int noS1_update_s1u_tunnel( const instance_t instanceP,
                            const gtpv1u_enb_create_tunnel_req_t *create_tunnel_req,
                            const rnti_t prior_rnti) {
  LOG_D(GTPU, "Start update tunnels for old RNTI %x, new RNTI %x, num_tunnels %d, sgw_S1u_teid %d, eps_bearer_id %d\n",
        prior_rnti,
        create_tunnel_req->rnti,
        create_tunnel_req->num_tunnels,
        create_tunnel_req->sgw_S1u_teid[0],
        create_tunnel_req->eps_bearer_id[0]);
  return 0;
}

int noS1_delete_s1u_tunnel(const instance_t instance,
                           gtpv1u_enb_delete_tunnel_req_t *req_pP) {
  LOG_D(GTPU, "Start delete tunnels for RNTI %x, num_erab %d, eps_bearer_id %d \n",
        req_pP->rnti,
        req_pP->num_erab,
        req_pP->eps_bearer_id[0]);
  return 0;
}

int noS1_send(const instance_t instance,
              gtpv1u_enb_tunnel_data_req_t *req) {
  uint8_t *buffer=req->buffer+req->offset;;
  size_t length=req->length;
  uint64_t rnti=req->rnti;
  int  rab_id=req->rab_id;
  /*
    il faut lire l'adresse source du UE (si on l'a pas rajoutée dans le create tunnel
    la ranger dans la table des IP->teid
    et envoyer le paquet (probablement en raw car l'entête ip est déja faite)
  */
  return 0;
}

int noS1_init(int a) {
  return -1;
}

int noS1_receiver(int fd) {
  /*
    il faut trouver le teid à partir de l'adresse destination (l'adresse du UE)
    puis adapter le code ci dessous
    PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, gtpv1u_teid_data_p->enb_id, ENB_FLAG_YES,  gtpv1u_teid_data_p->ue_id, 0, 0,gtpv1u_teid_data_p->enb_id);
    result = pdcp_data_req(
       &ctxt,
       SRB_FLAG_NO,
       (gtpv1u_teid_data_p->eps_bearer_id) ? gtpv1u_teid_data_p->eps_bearer_id - 4: 5-4,
       0, // mui
       SDU_CONFIRM_NO, // confirm
       buffer_len,
       buffer,
       PDCP_TRANSMISSION_MODE_DATA
    #if (RRC_VERSION >= MAKE_VERSION(14, 0, 0))
       ,NULL, NULL
    #endif
       );
  */
  return 0;
}

void *noS1_eNB_task(void *args) {
  int sd;
  AssertFatal((sd=noS1_init(0))>=0,"");
  itti_subscribe_event_fd(TASK_GTPV1_U, sd);

  while(1) {
    MessageDef *received_message_p = NULL;
    itti_receive_msg(TASK_GTPV1_U, &received_message_p);

    if (received_message_p != NULL) {
      instance_t instance = ITTI_MSG_INSTANCE(received_message_p);

      switch (ITTI_MSG_ID(received_message_p)) {
        case GTPV1U_ENB_DELETE_TUNNEL_REQ: {
          noS1_delete_s1u_tunnel(instance,
                                 &received_message_p->ittiMsg.Gtpv1uDeleteTunnelReq);
        }
        break;

        // DATA TO BE SENT TO UDP
        case GTPV1U_ENB_TUNNEL_DATA_REQ: {
          gtpv1u_enb_tunnel_data_req_t *data_req = NULL;
          data_req = &GTPV1U_ENB_TUNNEL_DATA_REQ(received_message_p);
          noS1_send(instance,
                    data_req);
        }
        break;

        case TERMINATE_MESSAGE: {
          itti_exit_task();
        }
        break;

        case TIMER_HAS_EXPIRED:
          LOG_W(GTPU,"Timer not devlopped\n");
          break;

        default: {
          LOG_E(GTPU, "Unkwnon message ID %d:%s\n",
                ITTI_MSG_ID(received_message_p),
                ITTI_MSG_NAME(received_message_p));
        }
        break;
      }

      AssertFatal( EXIT_SUCCESS == itti_free(ITTI_MSG_ORIGIN_ID(received_message_p), received_message_p),
                   "Failed to free memory !\n");
      received_message_p = NULL;
    }

    struct epoll_event *events;

    int nb_events = itti_get_events(TASK_GTPV1_U, &events);

    if (nb_events > 0 && events!= NULL )
      for (int i = 0; i < nb_events; i++)
        if (events[i].data.fd==sd)
          noS1_receiver(events[i].data.fd);
  }

  return NULL;
}
