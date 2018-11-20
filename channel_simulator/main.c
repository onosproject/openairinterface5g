#include "channel_simulator.h"
#include "connection_manager.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>

int main(void)
{
  channel_simulator c;
  connection_manager cm;
  int i;
  int samplerate = 7680000;

  init_connection_manager(&cm, "0.0.0.0", 4024);
  init_channel_simulator(&c, samplerate, 512);

  while (1) {
    connection_manager_lock(&cm);
    if (c.connections_count == 0) connection_manager_wait_connection(&cm);
    if (cm.new_connections) {
      for (i = 0; i < cm.new_connections; i++) {
        if (cm.c[i].samplerate != samplerate) {
          printf("ERROR: new connection has bad samplerate %d, dropping\n",
                 cm.c[i].samplerate);
          shutdown(cm.c[i].socket, SHUT_RDWR);
          close(cm.c[i].socket);
          continue;
        }
        channel_simulator_add_connection(&c,
            cm.c[i].socket, cm.c[i].rx_frequency, cm.c[i].tx_frequency);
      }
      connection_manager_clear(&cm);
    }
    connection_manager_unlock(&cm);

    for (i = 0 ; i < c.connections_count; i++) {
      connection *con = &c.connections[i];
      channel    *ch  = &c.channels[con->rx_channel_index];
      connection_send_rx(con, c.timestamp, ch->data, c.n_samples);
      connection_receive_tx(&c, con, c.timestamp + c.n_samples, c.n_samples);
    }

    cleanup_connections(&c);

    c.timestamp += c.n_samples;
    channel_simulate(&c);

    static int processed = 0;
    processed += c.n_samples;
    if (processed >= samplerate/1000) {
      processed -= samplerate/1000;
      usleep(2000);
    }
  }

  return 0;
}
