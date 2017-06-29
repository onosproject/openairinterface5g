#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>
#include <netinet/ether.h>
#include <unistd.h>
#include <errno.h>
#include <linux/sysctl.h>
#include <sys/sysctl.h>

#include "common_lib.h"
#include "ethernet_lib.h"

#include "mobipass.h"
#include "queues.h"

struct mobipass_header {
  uint16_t flags;
  uint16_t fifo_status;
  unsigned char seqno;
  unsigned char ack;
  uint32_t word0;
  uint32_t timestamp;
} __attribute__((__packed__));

int trx_eth_start(openair0_device *device) { return 0;}
int trx_eth_request(openair0_device *device, void *msg, ssize_t msg_len) { abort(); return 0;}
int trx_eth_reply(openair0_device *device, void *msg, ssize_t msg_len) { abort(); return 0;}
int trx_eth_get_stats(openair0_device* device) { return(0); }
int trx_eth_reset_stats(openair0_device* device) { return(0); }
void trx_eth_end(openair0_device *device) {}
int trx_eth_stop(openair0_device *device) { return(0); }
int trx_eth_set_freq(openair0_device* device, openair0_config_t *openair0_cfg,int exmimo_dump_config) { return(0); }
int trx_eth_set_gains(openair0_device* device, openair0_config_t *openair0_cfg) { return(0); }

int da__write(openair0_device *device, openair0_timestamp timestamp, void **buff, int nsamps, int cc, int flags) {
  struct mobipass_header *mh = (struct mobipass_header *)(((char *)buff[0]) + 14);
  static uint32_t last_timestamp = 4*7680*2-640;
  last_timestamp += 640;
  if (last_timestamp != ntohl(mh->timestamp)) { printf("bad timestamp wanted %d got %d\n", last_timestamp, ntohl(mh->timestamp)); exit(1); }
//printf("__write nsamps %d timestamps %ld seqno %d (packet timestamp %d)\n", nsamps, timestamp, mh->seqno, ntohl(mh->timestamp));
  if (nsamps != 640) abort();
  mobipass_send(buff[0]);
  return nsamps;
}

int da__read(openair0_device *device, openair0_timestamp *timestamp, void **buff, int nsamps, int cc) {
  static uint32_t ts = 0;
  static unsigned char seqno = 0;
//printf("__read nsamps %d return timestamp %d\n", nsamps, ts);
  *timestamp = htonl(ts);
  ts += nsamps;
  if (nsamps != 640) { printf("bad nsamps %d, should be 640\n", nsamps); fflush(stdout); abort(); }

  dequeue_from_mobipass(ntohl(*timestamp), buff[0]);

#if 1
  struct mobipass_header *mh = (struct mobipass_header *)(((char *)buff[0]) + 14);
  mh->flags = 0;
  mh->fifo_status = 0;
  mh->seqno = seqno++;
  mh->ack = 0;
  mh->word0 = 0;
  mh->timestamp = htonl(ts);
#endif

  return nsamps;
}

int transport_init(openair0_device *device, openair0_config_t *openair0_cfg,
        eth_params_t * eth_params )
{
  init_mobipass();

  eth_state_t *eth = (eth_state_t*)malloc(sizeof(eth_state_t));
  memset(eth, 0, sizeof(eth_state_t));

  if (eth_params->transp_preference != 4) goto err;
  if (eth_params->if_compress != 0) goto err;

  eth->flags = ETH_RAW_IF5_MOBIPASS;
  eth->compression = NO_COMPRESS;
  device->Mod_id           = 0;//num_devices_eth++;
  device->transp_type      = ETHERNET_TP;

  device->trx_start_func   = trx_eth_start;
  device->trx_request_func = trx_eth_request;
  device->trx_reply_func   = trx_eth_reply;
  device->trx_get_stats_func   = trx_eth_get_stats;
  device->trx_reset_stats_func = trx_eth_reset_stats;
  device->trx_end_func         = trx_eth_end;
  device->trx_stop_func        = trx_eth_stop;
  device->trx_set_freq_func = trx_eth_set_freq;
  device->trx_set_gains_func = trx_eth_set_gains;
  device->trx_write_func   = da__write;
  device->trx_read_func    = da__read;

  eth->if_name = eth_params->local_if_name;
  device->priv = eth;
        
  /* device specific */
  openair0_cfg[0].iq_rxrescale = 15;//rescale iqs
  openair0_cfg[0].iq_txshift = eth_params->iq_txshift;// shift
  openair0_cfg[0].tx_sample_advance = eth_params->tx_sample_advance;

  /* this is useless, I think */
  if (device->host_type == BBU_HOST) {
    /*Note scheduling advance values valid only for case 7680000 */    
    switch ((int)openair0_cfg[0].sample_rate) {
    case 30720000:
      openair0_cfg[0].samples_per_packet    = 3840;     
      break;
    case 23040000:     
      openair0_cfg[0].samples_per_packet    = 2880;
      break;
    case 15360000:
      openair0_cfg[0].samples_per_packet    = 1920;      
      break;
    case 7680000:
      openair0_cfg[0].samples_per_packet    = 960;     
      break;
    case 1920000:
      openair0_cfg[0].samples_per_packet    = 240;     
      break;
    default:
      printf("Error: unknown sampling rate %f\n",openair0_cfg[0].sample_rate);
      exit(-1);
      break;
    }
  }

  device->openair0_cfg=&openair0_cfg[0];

  return 0;

err:
  printf("MOBIPASS: bad config file?\n");
  exit(1);
}
