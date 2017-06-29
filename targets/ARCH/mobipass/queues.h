#ifndef _QUEUES_H_
#define _QUEUES_H_

#include <stdint.h>

void enqueue_to_mobipass(void *data);
void dequeue_to_mobipass(uint32_t timestamp, void *data);

void enqueue_from_mobipass(void *receive_packet);
void dequeue_from_mobipass(uint32_t timestamp, void *data);

void init_queues(void);

#endif /* _QUEUES_H_ */
