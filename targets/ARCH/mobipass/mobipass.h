#ifndef _MOBIPASS_H_
#define _MOBIPASS_H_

void mobipass_send(void *data);
void init_mobipass(void);

/* TODO: following variable only works for 10MHz */
#define SAMPLES_PER_1024_FRAMES (7680*2*10*1024)

#endif /* _MOBIPASS_H_ */
