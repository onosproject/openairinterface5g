#include "connection_manager.h"

#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <unistd.h>

typedef struct {
  uint64_t rx_frequency;
  uint64_t tx_frequency;
  uint32_t samplerate;
} init_message;

static int get_init_message(init_message *m, int s)
{
  unsigned char b[8*2+4];
  if (fullread(s, b, 8*2+4) != 8*2+4) return -1;
  m->rx_frequency = gu64(b);
  m->tx_frequency = gu64(b+8);
  m->samplerate   = gu32(b+8*2);
  return 0;
}

static void *connection_manager_thread(void *_cm)
{
  connection_manager *cm = _cm;
  init_message m;
  int s;
  int t;

  s = create_listen_socket(cm->ip, cm->port);

  while (1) {
    t = socket_accept(s);
    if (t == -1) {
      printf("ERROR: socket_accept failed (%s)\n", strerror(errno));
      continue;
    }
    if (get_init_message(&m, t) == -1) {
      printf("ERROR: get_init_message failed\n");
      close(t);
      continue;
    }

    connection_manager_lock(cm);
    cm->new_connections++;
    cm->c = realloc(cm->c, cm->new_connections * sizeof(new_connection));
    if (cm->c == NULL) {
      printf("ERROR: get_init_message: out of memory\n");
      exit(1);
    }
    cm->c[cm->new_connections-1].socket       = t;
    cm->c[cm->new_connections-1].rx_frequency = m.rx_frequency;
    cm->c[cm->new_connections-1].tx_frequency = m.tx_frequency;
    cm->c[cm->new_connections-1].samplerate   = m.samplerate;
    lock_signal(&cm->l);
    connection_manager_unlock(cm);
  }

  return NULL;
}

void init_connection_manager(connection_manager *cm, char *listen_ip, int port)
{
  cm->ip = strdup(listen_ip);
  if (cm->ip == NULL) {
    printf("ERROR: init_connection_manager: out of memory\n");
    exit(1);
  }
  cm->port = port;
  cm->new_connections = 0;
  cm->c = NULL;
  init_lock(&cm->l);
  new_thread(connection_manager_thread, cm);
}

void connection_manager_lock(connection_manager *cm)
{
  lock(&cm->l);
}

void connection_manager_unlock(connection_manager *cm)
{
  unlock(&cm->l);
}

/* this function must be called with lock acquired */
void connection_manager_wait_connection(connection_manager *cm)
{
  while (cm->new_connections == 0) lock_wait(&cm->l);
}

/* this function must be called with lock acquired */
void connection_manager_clear(connection_manager *cm)
{
  free(cm->c);
  cm->c = NULL;
  cm->new_connections = 0;
}
