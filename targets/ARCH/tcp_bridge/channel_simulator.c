#include "common_lib.h"
#include "channel_simulator/utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <inttypes.h>

typedef struct buffer_s {
  uint64_t timestamp;
  int n_samples;
  int used_samples;             /* for TX to check correct use of buffer */
  void *b;
  struct buffer_s *next;
} buffer_t;

typedef struct {
  int sock;
  int samples_per_subframe;
  uint64_t rx_frequency;
  uint64_t tx_frequency;
  uint32_t rx_sample_advance;
  uint32_t tx_sample_advance;
  lock_t lock;
  buffer_t * volatile rx_head;
  buffer_t *rx_last;
  buffer_t *tx_head;
  buffer_t *tx_last;
  volatile int first_rx_ok;
  uint64_t rx_next_timestamp;
  uint64_t first_tx_timestamp;
  lock_t tx_lock;
  lock_t tx_buf_lock;
  /* data to pass from rx thread to tx thread */
  uint64_t first_timestamp;
  int samples_per_packet;
} channel_simulator_state_t;

void add_rx_buffer(channel_simulator_state_t *c, buffer_t *b)
{
  lock(&c->lock);
  if (c->rx_head == NULL) {
    c->rx_head = b;
  } else {
    c->rx_last->next = b;
  }
  c->rx_last = b;
  lock_signal(&c->lock);
  unlock(&c->lock);
}

void add_tx_buffer(channel_simulator_state_t *c, buffer_t *b)
{
  lock(&c->tx_buf_lock);
  if (c->tx_head == NULL) {
    c->tx_head = b;
  } else {
    c->tx_last->next = b;
  }
  c->tx_last = b;
  lock_signal(&c->tx_buf_lock);
  unlock(&c->tx_buf_lock);
}

/* called with c->lock acquired */
uint32_t get_rx_sample(channel_simulator_state_t *c, uint64_t timestamp)
{
  uint32_t *b;
  uint32_t ret;
again:
  while (c->rx_head == NULL) lock_wait(&c->lock);
  if (c->rx_head->timestamp + c->rx_head->n_samples <= timestamp) {
    buffer_t *cur = c->rx_head;
    c->rx_head = cur->next;
    if (c->rx_head == NULL) {
      c->rx_last = NULL;
    }
    free(cur->b);
    free(cur);
    goto again;
  }
  if (timestamp < c->rx_head->timestamp) {
    printf("ERROR: channel_simulator: get_rx_sample failure\n");
    exit(1);
  }
  b = c->rx_head->b;
  ret = b[timestamp - c->rx_head->timestamp];
  return ret;
}

uint32_t get_tx_sample(channel_simulator_state_t *c, uint64_t timestamp)
{
  buffer_t *cur;
  uint32_t ret;
again:
  if (timestamp < c->first_tx_timestamp) { ret = 0; goto done; }
  while (c->tx_head == NULL) lock_wait(&c->tx_buf_lock);
  if (timestamp < c->tx_head->timestamp) { ret = 0; goto done; }
  if (c->tx_head->timestamp + c->tx_head->n_samples <= timestamp) {
    cur = c->tx_head;
    c->tx_head = cur->next;
    if (c->tx_head == NULL) c->tx_last = NULL;
    if (cur->used_samples != cur->n_samples) {
      printf("ERROR: channel_simulator: get_tx_sample: late packet "
             "timestamp %"PRIu64" n_samples %d used_samples %d\n",
             cur->timestamp, cur->n_samples, cur->used_samples);
    }
    free(cur->b);
    free(cur);
    goto again;
  }
  cur = c->tx_head;
  ret = 0;
  while (cur) {
    if (timestamp >= cur->timestamp &&
        timestamp < cur->timestamp + cur->n_samples) {
      uint32_t *b = cur->b;
      ret = b[timestamp - cur->timestamp];
      cur->used_samples++;
      goto done;
    }
    cur = cur->next;
  }
done:
  return ret;
}

void *tx_thread(void *_c)
{
  channel_simulator_state_t *c = _c;
  unsigned char b[8+4];
  uint32_t *out;
  uint64_t timestamp;
  int n_samples;
  int i;

  lock(&c->lock);
  while (c->first_rx_ok != 1) lock_wait(&c->lock);
  unlock(&c->lock);

  timestamp = c->first_timestamp;
  n_samples = c->samples_per_packet;

  out = malloc(n_samples * 4);
  if (out == NULL) goto err;

  while (1) {
    lock(&c->tx_buf_lock);
    for (i = 0; i < c->samples_per_packet; i++)
      out[i] = get_tx_sample(c, timestamp - c->rx_sample_advance
                                          + c->tx_sample_advance
                                          + n_samples + i);
    unlock(&c->tx_buf_lock);
    pu64(b, timestamp - c->rx_sample_advance
                      + c->tx_sample_advance + n_samples);
    pu32(b+8, n_samples);
    lock(&c->tx_lock);
    if (fullwrite(c->sock, b, 8+4) != 8+4 ||
        fullwrite(c->sock, out, n_samples * 4) != n_samples * 4) {
      unlock(&c->tx_lock);
      goto err;
    }
    unlock(&c->tx_lock);
    timestamp += n_samples;
  }

  return NULL;

err:
  printf("ERROR: channel_simulator: tx_thread failed\n");
  exit(1);
}

void *rx_thread(void *_c)
{
  channel_simulator_state_t *c = _c;
  unsigned char b[8+4];
  buffer_t *rx;
  uint64_t timestamp;
  int n_samples;

  while (1) {
    if (fullread(c->sock, b, 8+4) != 8+4) goto err;
    rx = calloc(1, sizeof(buffer_t)); if (rx == NULL) goto err;
    timestamp = rx->timestamp = gu64(b);
    n_samples = rx->n_samples = gu32(b+8);
    rx->b = malloc(rx->n_samples * 4); if (rx->b == NULL) goto err;
    rx->next = NULL;
    if (fullread(c->sock, rx->b, rx->n_samples * 4) != rx->n_samples * 4)
      goto err;
    add_rx_buffer(c, rx);
    if (c->first_rx_ok == 0) {
      lock(&c->lock);
      c->rx_next_timestamp = timestamp;
      /* first_tx_timestamp is first RX timestamp + samples per frame * 4
       * because the eNB works as follows: it receives subframe N and
       * generates subframe N+4. This could become configurable for if
       * the eNB works differently in the future.
       */
      c->first_tx_timestamp = timestamp + c->samples_per_subframe * 4;
      c->first_timestamp = timestamp;
      c->samples_per_packet = n_samples;
      c->first_rx_ok = 1;
      lock_signal(&c->lock);
      unlock(&c->lock);
    }
  }

  return NULL;

err:
  printf("ERROR: channel_simulator: rx_thread failed\n");
  exit(1);
}

void init_connection(channel_simulator_state_t *c)
{
  unsigned char b[8*2+4*3];
  char *env;

  env = getenv("CHANNEL_SIMULATOR_RX_SAMPLE_ADVANCE");
  if (env != NULL) {
    c->rx_sample_advance = atoi(env);
    printf("channel_simulator: setting rx_sample_advance to %"PRIu32"\n",
           c->rx_sample_advance);
  } else {
    c->rx_sample_advance = 0;
    printf("channel_simulator: environment variable"
           " CHANNEL_SIMULATOR_RX_SAMPLE_ADVANCE"
           " not found, using 0\n");
  }

  env = getenv("CHANNEL_SIMULATOR_TX_SAMPLE_ADVANCE");
  if (env != NULL) {
    c->tx_sample_advance = atoi(env);
    printf("channel_simulator: setting tx_sample_advance to %"PRIu32"\n",
           c->tx_sample_advance);
  } else {
    c->tx_sample_advance = 0;
    printf("channel_simulator: environment variable"
           " CHANNEL_SIMULATOR_TX_SAMPLE_ADVANCE"
           " not found, using 0\n");
  }

  pu64(b, c->rx_frequency);
  pu64(b+8, c->tx_frequency);
  pu32(b+8*2, c->rx_sample_advance);
  pu32(b+8*2+4, c->tx_sample_advance);
  pu32(b+8*2+4*2, c->samples_per_subframe * 1000);

  lock(&c->tx_lock);
  if (fullwrite(c->sock, b, 8*2+4*3) != 8*2+4*3) {
    printf("ERROR: channel_simulator: init_connection failed\n");
    exit(1);
  }
  unlock(&c->tx_lock);
}

int channel_simulator_start(openair0_device *device)
{
  int port = 4024;
  char *ip = "127.0.0.1";
  char *env;
  channel_simulator_state_t *channel_simulator = device->priv;
  struct sockaddr_in addr;
  int v;
  int sock;

  env = getenv("CHANNEL_SIMULATOR_IP");
  if (env != NULL)
    ip = env;
  else
    printf("channel_simulator: environment variable CHANNEL_SIMULATOR_IP"
           " not found, using default IP %s\n", ip);

  env = getenv("CHANNEL_SIMULATOR_PORT");
  if (env != NULL)
    port = atoi(env);
  else
    printf("channel_simulator: environment variable CHANNEL_SIMULATOR_PORT"
           " not found, using default port %d\n", port);

  sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock == -1) { perror("channel_simulator: socket"); exit(1); }

  v = 1;
  if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &v, sizeof(v)) == -1)
    { perror("channel_simulator: setsockopt(NODELAY)"); exit(1); }

  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = inet_addr(ip);

  while (1) {
    printf("channel_simulator: trying to connect to %s:%d\n", ip, port);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
      printf("channel_simulator: connection established\n");
      channel_simulator->sock = sock;
      init_connection(channel_simulator);
      new_thread(rx_thread, channel_simulator);
      new_thread(tx_thread, channel_simulator);
      return 0;
    }

    perror("channel_simulator");
    sleep(1);
  }
}

int channel_simulator_request(openair0_device *device, void *msg, ssize_t msg_len) { abort(); return 0; }
int channel_simulator_reply(openair0_device *device, void *msg, ssize_t msg_len) { abort(); return 0; }
int channel_simulator_get_stats(openair0_device* device) { return 0; }
int channel_simulator_reset_stats(openair0_device* device) { return 0; }
void channel_simulator_end(openair0_device *device) {}
int channel_simulator_stop(openair0_device *device) { return 0; }
int channel_simulator_set_gains(openair0_device* device, openair0_config_t *openair0_cfg) { return 0; }

int channel_simulator_set_freq(openair0_device* device, openair0_config_t *openair0_cfg,int exmimo_dump_config)
{
  channel_simulator_state_t *c = device->priv;
  unsigned char b[8+4+8*2];

  if (getenv("CHANNEL_SIMULATOR_RX_FREQUENCY") != NULL ||
      getenv("CHANNEL_SIMULATOR_TX_FREQUENCY") != NULL)
    return 0;

  pu64(b, 0);
  pu32(b+8, 8*2);
  pu64(b+8+4, openair0_cfg[0].rx_freq[0]);
  pu64(b+8+4+8, openair0_cfg[0].tx_freq[0]);

  lock(&c->tx_lock);
  if (fullwrite(c->sock, b, 8+4+8*2) != 8+4+8*2) {
    printf("ERROR: channel_simulator: channel_simulator_set_freq failed\n");
    exit(1);
  }
  unlock(&c->tx_lock);

  return 0;
}

int channel_simulator_write(openair0_device *device, openair0_timestamp timestamp, void **buff, int nsamps, int cc, int flags)
{
  channel_simulator_state_t *c = device->priv;
  buffer_t *tx;

//printf("channel_simulator_write start ts %"PRIu64" nsamps %d\n", (uint64_t)timestamp, nsamps);
  if (cc != 1)
    { printf("ERROR: channel_simulator: only works with 1 CC\n"); exit(1); }

  tx = calloc(1, sizeof(buffer_t)); if (tx == NULL) goto err;
  tx->timestamp = timestamp;
  tx->n_samples = nsamps;
  tx->used_samples = 0;
  tx->b = malloc(nsamps * 4); if (tx->b == NULL) goto err;
  memcpy(tx->b, buff[0], nsamps * 4);
  tx->next = NULL;

  add_tx_buffer(c, tx);

//printf("channel_simulator_write done\n");
  return nsamps;

err:
  printf("ERROR: channel_simulator: channel_simulator_write failed\n");
  exit(1);
}

uint64_t get_rx_timestamp(channel_simulator_state_t *c)
{
  uint64_t ret;
  lock(&c->lock);
  while (!c->first_rx_ok) lock_wait(&c->lock);
  ret = c->rx_next_timestamp;
  unlock(&c->lock);
  return ret;
}

int channel_simulator_read(openair0_device *device, openair0_timestamp *timestamp, void **buff, int nsamps, int cc)
{
//printf("channel_simulator_read start nsamps %d\n", nsamps);
  channel_simulator_state_t *c = device->priv;
  uint32_t *r = (uint32_t *)(buff[0]);
  uint64_t ts = get_rx_timestamp(c);
  int i;
  lock(&c->lock);
  for (i = 0; i < nsamps; i++) r[i] = get_rx_sample(c, ts + i);
  c->rx_next_timestamp += nsamps;
  unlock(&c->lock);
  *timestamp = ts;
//printf("channel_simulator_read done ts %"PRIu64"\n", ts);
  return nsamps;
}

__attribute__((__visibility__("default")))
int device_init(openair0_device* device, openair0_config_t *openair0_cfg)
{
  channel_simulator_state_t *channel_simulator = (channel_simulator_state_t*)malloc(sizeof(channel_simulator_state_t));
  char *env;

  memset(channel_simulator, 0, sizeof(channel_simulator_state_t));

  /* only 25, 50 or 100 PRBs handled for the moment */
  if (openair0_cfg[0].sample_rate != 30720000 &&
      openair0_cfg[0].sample_rate != 15360000 &&
      openair0_cfg[0].sample_rate !=  7680000) {
    printf("channel_simulator: ERROR: only 25, 50 or 100 PRBs supported\n");
    exit(1);
  }

  device->trx_start_func       = channel_simulator_start;
  device->trx_get_stats_func   = channel_simulator_get_stats;
  device->trx_reset_stats_func = channel_simulator_reset_stats;
  device->trx_end_func         = channel_simulator_end;
  device->trx_stop_func        = channel_simulator_stop;
  device->trx_set_freq_func    = channel_simulator_set_freq;
  device->trx_set_gains_func   = channel_simulator_set_gains;
  device->trx_write_func       = channel_simulator_write;
  device->trx_read_func        = channel_simulator_read;

  device->priv = channel_simulator;

  switch ((int)openair0_cfg[0].sample_rate) {
  case 30720000: channel_simulator->samples_per_subframe = 30720; break;
  case 15360000: channel_simulator->samples_per_subframe = 15360; break;
  case 7680000:  channel_simulator->samples_per_subframe = 7680; break;
  }

  init_lock(&channel_simulator->lock);
  init_lock(&channel_simulator->tx_lock);
  init_lock(&channel_simulator->tx_buf_lock);

  env = getenv("CHANNEL_SIMULATOR_RX_FREQUENCY");
  if (env == NULL) {
    printf("channel_simulator: CHANNEL_SIMULATOR_RX_FREQUENCY not set, used provided value %g\n",
           openair0_cfg[0].rx_freq[0]);
    channel_simulator->rx_frequency = openair0_cfg[0].rx_freq[0];
  } else {
    channel_simulator->rx_frequency = atoll(env);
    printf("channel_simulator: set RX frequency to %"PRIu64"\n", channel_simulator->rx_frequency);
  }

  env = getenv("CHANNEL_SIMULATOR_TX_FREQUENCY");
  if (env == NULL) {
    printf("channel_simulator: CHANNEL_SIMULATOR_TX_FREQUENCY not set, used provided value %g\n",
           openair0_cfg[0].tx_freq[0]);
    channel_simulator->tx_frequency = openair0_cfg[0].tx_freq[0];
  } else {
    channel_simulator->tx_frequency = atoll(env);
    printf("channel_simulator: set TX frequency to %"PRIu64"\n", channel_simulator->tx_frequency);
  }

  /* let's pretend to be a b2x0 */
  device->type = USRP_B200_DEV;

  device->openair0_cfg=&openair0_cfg[0];

  return 0;
}
