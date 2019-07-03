#define _GNU_SOURCE
#include "usrp.h"
#include "../utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <pthread.h>
#include <immintrin.h>

#include <inttypes.h>

typedef struct {
  char *data;
  uint32_t n_samples;
  uint64_t timestamp;
} buffer_t;

void receive_from_channel_simulator(int sock, buffer_t *buf)
{
  unsigned char b[8+4];
  uint32_t n_samples;

  if (fullread(sock, b, 8+4) != 8+4) goto err;

  buf->timestamp = gu64(b);
  n_samples = gu32(b+8);
  if (n_samples != buf->n_samples) {
    free(buf->data);
    buf->n_samples = n_samples;
    if (posix_memalign((void **)&buf->data, 32, buf->n_samples * 4) != 0) goto err;
  }
  if (fullread(sock, buf->data, buf->n_samples * 4) != buf->n_samples * 4)
    goto err;

  return;

err:
  printf("ERROR: receive_from_channel_simulator failed\n");
  exit(1);
}

void send_to_channel_simulator(int sock, char *usrp_data, int n_samples,
                               uint64_t timestamp)
{
  unsigned char b[8+4];

  pu64(b, timestamp);
  pu32(b+8, n_samples);
  if (fullwrite(sock, b, 8+4) != 8+4 ||
      fullwrite(sock, usrp_data, n_samples * 4) != n_samples * 4) {
    printf("ERROR: send_to_channel_simulator failed\n");
    exit(1);
  }
}

int connect_to_channel_simulator(void)
{
  int port = 4024;
  char *ip = "127.0.0.1";
  struct sockaddr_in addr;
  int sock;
  int v;

  sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock == -1) { perror("socket"); exit(1); }

  v = 1;
  if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &v, sizeof(v)) == -1)
    { perror("channel_simulator: setsockopt(NODELAY)"); exit(1); }

  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = inet_addr(ip);

  while (1) {
    printf("trying to connect to %s:%d\n", ip, port);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
      printf("connection established\n");
      return sock;
    }

    perror("channel_simulator");
    sleep(1);
  }
}

void init_connection(int sock, uint64_t rx_frequency, uint64_t tx_frequency,
                     int samples_per_subframe)
{
  unsigned char b[8*2+4*3];

  pu64(b, rx_frequency);
  pu64(b+8, tx_frequency);
  pu32(b+8*2, samples_per_subframe * 2);         /* RX sample advance */
  pu32(b+8*2+4, 0);                              /* TX sample advance */
  pu32(b+8*2+4*2, samples_per_subframe * 1000);

  if (fullwrite(sock, b, 8*2+4*3) != 8*2+4*3) {
    printf("ERROR: init_connection failed\n");
    exit(1);
  }
}

void go_realtime(void)
{
  struct sched_param sparam;
  int policy;
  memset(&sparam, 0, sizeof(sparam));
  sparam.sched_priority = sched_get_priority_max(SCHED_FIFO)-1;
  policy = SCHED_FIFO ;
  if (pthread_setschedparam(pthread_self(), policy, &sparam) != 0) {
    printf("ERROR: Failed to set pthread priority\n");
    exit(1);
  }
}

int main(void)
{
  int sock;
  buffer_t buf = { data:NULL, n_samples:0, timestamp:0 };
  uint64_t sim_timestamp;
  uint64_t usrp_timestamp;
  int samples_per_subframe = 7680;
  int tx_sample_advance = 0; //40;
  char *usrp_data;

  go_realtime();

  usrp_init_connection(2560000000, 2680000000);

  sock = connect_to_channel_simulator();
  init_connection(sock, 1, 2, samples_per_subframe);

  receive_from_channel_simulator(sock, &buf);
  sim_timestamp = buf.timestamp - 2 * samples_per_subframe + buf.n_samples;

  if (posix_memalign((void **)&usrp_data, 32, buf.n_samples * 4) != 0) {
    printf("ERROR: out of memory\n");
    exit(1);
  }

  send_to_channel_simulator(sock, usrp_data, buf.n_samples, sim_timestamp);
  sim_timestamp += buf.n_samples;

  usrp_start();
  usrp_timestamp = usrp_read(usrp_data, buf.n_samples);

  while (1) {
    int i;
    for (i = 0; i < buf.n_samples * 2; i += 16) {
      __m256i *a = (__m256i *)&((int16_t *)buf.data)[i];
      *a = _mm256_slli_epi16(*a, 4);
    }
    usrp_write(buf.data, buf.n_samples,
               usrp_timestamp - buf.n_samples + 2 * samples_per_subframe - tx_sample_advance);
    usrp_timestamp = usrp_read(usrp_data, buf.n_samples);
    for (i = 0; i < buf.n_samples * 2; i += 16) {
      __m256i *a = (__m256i *)&((int16_t *)usrp_data)[i];
      *a = _mm256_srai_epi16(*a, 4);
    }
    send_to_channel_simulator(sock, usrp_data, buf.n_samples, sim_timestamp);
    sim_timestamp += buf.n_samples;
    receive_from_channel_simulator(sock, &buf);
  }

  return 0;
}
