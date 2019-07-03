#include "channel_simulator.h"

#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/socket.h>
#include <immintrin.h>

void init_channel_simulator(channel_simulator *c,
    uint32_t samplerate, int n_samples)
{
  c->timestamp = 1024;         /* timestamps [0..1023] used as commands */
  c->samplerate = samplerate;
  c->n_samples = n_samples;
  c->channels = NULL;
  c->channels_count = 0;
  c->connections = NULL;
  c->connections_count = 0;
}

void cleanup_connections(channel_simulator *c)
{
  connection *con;
  int i;

  /* remove dead connections */
  i = 0;
  while (i < c->connections_count) {
    if (c->connections[i].socket != -1) { i++; continue; }
    con = &c->connections[i];
    c->channels[con->rx_channel_index].connection_count--;
    c->channels[con->tx_channel_index].connection_count--;
    free(con->iq_buffer);
    if (c->connections_count == 1) {
      free(c->connections);
      c->connections = NULL;
      c->connections_count = 0;
      break;
    }
    c->connections_count--;
    memmove(&c->connections[i], &c->connections[i+1],
            (c->connections_count-i) * sizeof(connection));
    c->connections = realloc(c->connections,
                             c->connections_count * sizeof(connection));
    if (c->connections == NULL) goto oom;
  }

  /* remove channels with no more connections */
  i = 0;
  while (i < c->channels_count) {
    if (c->channels[i].connection_count) { i++; continue; }
    free(c->channels[i].data);
    if (c->channels_count == 1) {
      free(c->channels);
      c->channels = NULL;
      c->channels_count = 0;
      break;
    }
    c->channels_count--;
    memmove(&c->channels[i], &c->channels[i+1],
            (c->channels_count-i) * sizeof(channel));
    c->channels = realloc(c->channels, c->channels_count * sizeof(channel));
    if (c->channels == NULL) goto oom;
  }

  return;

oom:
  printf("ERROR: cleanup_connections: out of memory\n");
  exit(1);
}

static int find_or_create_channel(channel_simulator *c, uint64_t freq,
                                  uint32_t sample_advance)
{
  channel *chan;
  int i;
  for (i = 0; i < c->channels_count; i++) {
    if (c->channels[i].frequency != freq) continue;
    if (c->channels[i].sample_advance != sample_advance) {
      printf("ERROR: bad sample_advance\n");
      exit(1);
    }
    return i;
  }

  c->channels_count++;
  c->channels = realloc(c->channels, c->channels_count * sizeof(channel));
  if (c->channels == NULL) goto oom;

  chan = &c->channels[i];
  chan->frequency = freq;
  chan->sample_advance = sample_advance;
  if (posix_memalign((void **)&chan->data, 32, c->n_samples * 4) != 0)
    goto oom;
  chan->connection_count = 0;

  return i;

oom:
  printf("ERROR: find_or_create_channel: out of memory\n");
  exit(1);
}

void channel_simulator_add_connection(channel_simulator *c,
            int socket, uint64_t rx_frequency, uint64_t tx_frequency,
            uint32_t rx_sample_advance, uint32_t tx_sample_advance)
{
  connection *con;

  printf("INFO: new connection RX %"PRIu64" TX %"PRIu64"\n",
         rx_frequency, tx_frequency);

  c->connections_count++;
  c->connections = realloc(c->connections,
                           c->connections_count * sizeof(connection));
  if (c->connections == NULL) goto oom;
  con = &c->connections[c->connections_count-1];

  con->socket = socket;
  if (posix_memalign((void **)&con->iq_buffer, 32, c->n_samples * 4) != 0)
    goto oom;
  con->rx_frequency = rx_frequency;
  con->tx_frequency = tx_frequency;
  con->rx_channel_index = find_or_create_channel(c, rx_frequency,
                                                 rx_sample_advance);
  con->tx_channel_index = find_or_create_channel(c, tx_frequency,
                                                 tx_sample_advance);

  c->channels[con->rx_channel_index].connection_count++;
  c->channels[con->tx_channel_index].connection_count++;

  if (posix_memalign(&con->gain, 32, 32) != 0) {
    printf("posix_memalign failed\n");
    exit(1);
  }

  /* default gain = 1 (actually 1 - 1/2^15) */
  *(__m256i *)con->gain = _mm256_set1_epi16(0x7fff);

  return;

oom:
  printf("ERROR: channel_simulator_add_connection: out of memory\n");
  exit(1);
}

void connection_send_rx(connection *c, uint64_t timestamp,
    uint32_t *data, int n_samples)
{
  unsigned char b[8+4];
  int k;
  int16_t *from;
  int16_t *to;
  __m256i *gain;

  if (c->socket == -1) return;

  /* apply gain on data to send */
  from = (int16_t *)data;
  to = (int16_t *)c->iq_buffer;
  gain = (__m256i *)c->gain;

  /* does: to[] = from[] * gain[] */
  for (k = 0; k < n_samples * 2; k+=16) {
    __m256i *a, *b;
    a = (__m256i *)&to[k];
    b = (__m256i *)&from[k];
    *a = _mm256_mulhrs_epi16(*b, *gain);
  }

  pu64(b, timestamp);
  pu32(b+8, n_samples);
  if (fullwrite(c->socket, b, 8+4) != 8+4 ||
      fullwrite(c->socket, c->iq_buffer, n_samples * 4) != n_samples * 4) {
    printf("ERROR: connection_send_rx failed, dropping\n");
    shutdown(c->socket, SHUT_RDWR);
    close(c->socket);
    c->socket = -1;
  }
}

void command_set_frequency(channel_simulator *cs, connection *con, int n)
{
  unsigned char b[8*2];
  uint64_t rx_frequency;
  uint64_t tx_frequency;
  uint32_t rx_sample_advance;
  uint32_t tx_sample_advance;

  if (n != 8*2) goto err;
  if (fullread(con->socket, b, 8*2) != 8*2) goto err;
  rx_frequency = gu64(b);
  tx_frequency = gu64(b+8);

  printf("INFO: setting new frequencies RX %"PRIu64" and TX %"PRIu64"\n",
         rx_frequency, tx_frequency);

  cs->channels[con->rx_channel_index].connection_count--;
  cs->channels[con->tx_channel_index].connection_count--;

  rx_sample_advance = cs->channels[con->rx_channel_index].sample_advance;
  tx_sample_advance = cs->channels[con->tx_channel_index].sample_advance;

  con->rx_frequency = rx_frequency;
  con->tx_frequency = tx_frequency;
  con->rx_channel_index = find_or_create_channel(cs, rx_frequency,
                                                 rx_sample_advance);
  con->tx_channel_index = find_or_create_channel(cs, tx_frequency,
                                                 tx_sample_advance);
  cs->channels[con->rx_channel_index].connection_count++;
  cs->channels[con->tx_channel_index].connection_count++;

  return;

err:
  printf("ERROR: command_set_frequency failed, dropping\n");
  shutdown(con->socket, SHUT_RDWR);
  close(con->socket);
  con->socket = -1;
}

void command_set_gain(channel_simulator *cs, connection *con, int n)
{
  unsigned char b[4];
  int v;
  if (n != 4) goto err;
  if (fullread(con->socket, b, 4) != 4) goto err;
  v = gu32(b);

  *(__m256i *)con->gain = _mm256_set1_epi16(v);

  printf("INFO: setting gain to %d\n", v);

  return;

err:
  printf("ERROR: command_set_gain failed, dropping\n");
  shutdown(con->socket, SHUT_RDWR);
  close(con->socket);
  con->socket = -1;
}

void do_command(channel_simulator *cs, connection *c, uint64_t command, int n)
{
  switch (command) {
  case 0: return command_set_frequency(cs, c, n);
  case 1: return command_set_gain(cs, c, n);
  default:
    printf("ERROR: bad command %"PRIu64", dropping\n", command);
    shutdown(c->socket, SHUT_RDWR);
    close(c->socket);
    c->socket = -1;
  }
}

void connection_receive_tx(channel_simulator *cs,
    connection *c, uint64_t timestamp, int n_samples)
{
  unsigned char b[8+4];
  uint64_t recv_timestamp;
  uint32_t recv_n_samples;

again:
  if (c->socket == -1) return;

  if (fullread(c->socket, b, 8+4) != 8+4) goto err;
  recv_timestamp = gu64(b);
  recv_n_samples = gu32(b+8);

  if (recv_timestamp < 1024) {
    do_command(cs, c, recv_timestamp, recv_n_samples);
    goto again;
  }

  if (timestamp != recv_timestamp) {
    printf("ERROR: bad timestamp, got %"PRIu64" expected %"PRIu64"\n",
           recv_timestamp, timestamp);
    goto err;
  }
  if (n_samples != recv_n_samples) {
    printf("ERROR, bad n_samples, got %d expected %d\n",
           recv_n_samples, n_samples);
    goto err;
  }
  if (fullread(c->socket, c->iq_buffer, n_samples * 4) != n_samples * 4)
    goto err;

  return;

err:
  printf("ERROR: connection_receive_tx failed, dropping\n");
  shutdown(c->socket, SHUT_RDWR);
  close(c->socket);
  c->socket = -1;
}

void channel_simulate(channel_simulator *c)
{
  int i;
  int k;
  connection *con;
  channel *chan;
  int16_t *to;
  int16_t *from;
  int16_t mix[c->channels_count][c->n_samples*2] __attribute__ ((aligned (32)));

  memset(mix, 0, c->channels_count * c->n_samples*2 * 2);

  /* clear channels */
  for (i = 0; i < c->channels_count; i++)
    memset(c->channels[i].data, 0, c->n_samples * 4);

  /* basic mixing */
  for (i = 0; i < c->connections_count; i++) {
    __m256i *gain;
    con = &c->connections[i];
    from = (int16_t *)con->iq_buffer;
    gain = (__m256i *)con->gain;

    /* does: mix[] = mix[] + from[] * gain[] */
    for (k = 0; k < c->n_samples * 2; k+=16) {
      __m256i *a, *b, v;
      a = (__m256i *)&mix[con->tx_channel_index][k];
      b = (__m256i *)&from[k];
      v = _mm256_mulhrs_epi16(*b, *gain);
      *a = _mm256_adds_epi16(*a, v);
    }
  }

  for (i = 0; i < c->channels_count; i++) {
    chan = &c->channels[i];
    to = (int16_t *)chan->data;
    memcpy(to, mix[i], c->n_samples * 2 * 2);
  }
}
