#ifndef FF_MAC_H
#define FF_MAC_H

/* this file contains OAI related FAPI definitions */

/* this is the public view of the FAPI's OAI interface */
typedef struct {
  void *sched;     /* this is the pointer returned by SchedInit */
                   /* to be used when calling FAPI functions */
} fapi_interface_t;

/* this function initializes OAI's FAPI interfacing
 * it returns the opaque pointer given by SchedInit
 */
fapi_interface_t *init_fapi(void);

#endif /* FF_MAC_H */
