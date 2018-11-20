#ifndef _CHANNEL_SIMULATOR_H_
#define _CHANNEL_SIMULATOR_H_

#include <stdint.h>

typedef struct {
  uint64_t frequency;         /* unit: Hz */
  uint32_t *data;
  int      connection_count;
} channel;

typedef struct {
  int      socket;
  uint32_t *tx_buffer;
  uint64_t rx_frequency;
  uint64_t tx_frequency;
  int      rx_channel_index;
  int      tx_channel_index;
} connection;

typedef struct {
  uint64_t   timestamp;
  uint32_t   samplerate;
  int        n_samples;                /* handle n_samples at a time */
  channel    *channels;
  int        channels_count;
  connection *connections;
  int        connections_count;
} channel_simulator;

void init_channel_simulator(channel_simulator *c,
    uint32_t samplerate, int n_samples);
void cleanup_connections(channel_simulator *c);
void channel_simulator_add_connection(channel_simulator *c,
    int socket, uint64_t rx_frequency, uint64_t tx_frequency);

void connection_send_rx(connection *c, uint64_t timestamp,
    uint32_t *data, int n_samples);
void connection_receive_tx(channel_simulator *cs,
    connection *c, uint64_t timestamp, int n_samples);

void channel_simulate(channel_simulator *c);

#endif /* _CHANNEL_SIMULATOR_H_ */
