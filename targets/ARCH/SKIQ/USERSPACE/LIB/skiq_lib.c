/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/** skiq_lib.cpp
 *
 * \author: Raymond Knopp : raymond.knopp@eurecom.fr
 */


#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "math.h"
#include "common_lib.h"
#include "sidekiq_api.h"
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#include "PHY/TOOLS/time_meas.h"

/** @addtogroup _SIDEKIQ_PHY_RF_INTERFACE_
 * @{
 */


#ifdef __SSE4_1__
#  include <smmintrin.h>
#endif
 
#ifdef __AVX2__
#  include <immintrin.h>
#endif

//#define DEBUG_SKIQ_TX 1
//#define DEBUG_SKIQ_RX 1

#define SKIQ_MAX_TX_ELM 10
#define SKIQ_MAX_NUM_TX_PACKETS (30720/1020)
#define SKIQ_BLOCK_SIZE_IN_WORDS        (1024-4) /* OAI choice for block size */

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

//#define SKIQ_ASYNCH 1

/* helper MACROs (pulled from Linux kernel) */
#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

#define container_of(ptr, _type, member) ({          \
    const typeof(((_type *)0)->member)*__mptr = (ptr);    \
             (_type *)((char *)__mptr - offsetof(_type, member)); })
typedef struct {
  uint32_t *dataptr;
  uint32_t length;
  uint64_t timestamp;
  uint32_t active;
} TX_input_packet_q_elm_t;

typedef struct {
  int head;
  int tail;
  TX_input_packet_q_elm_t elm[SKIQ_MAX_TX_ELM-1];
} TX_input_packet_q_t;

typedef struct {
    uint64_t meta, ts;
    int32_t iq[SKIQ_BLOCK_SIZE_IN_WORDS];
} skiq_tx_packet_t;

typedef struct {
    void *priv;
    int32_t idx;
    skiq_tx_packet_t skiq_pkt;
} tx_packet_t;

/*! \brief Sidekiq specific data structure */ 
typedef struct {

  //! Number of cards
  uint8_t number_of_cards;
  //! List of card ids
  uint8_t card_list[SKIQ_MAX_NUM_CARDS];
  //! Number of buffers
  unsigned int num_buffers;
  //! Buffer size 
  unsigned int buffer_size;
  //! Number of transfers
  unsigned int num_transfers;
  //! RX timeout
  unsigned int rx_timeout_ms;
  //! TX timeout
  unsigned int tx_timeout_ms;
  //! Sample rate
  unsigned int sample_rate;
  //! time offset between transmiter timestamp and receiver timestamp;
  double tdiff;
  //! TX number of forward samples use brf_time_offset to get this value
  int tx_sample_offset; //166 for 20Mhz

  //! RX residual samples from last read (flushed upon next read)
  uint32_t residual_read_buffer[SKIQ_MAX_NUM_CARDS][1018];
  //! RX residual timestamp from last read (used upon next read)
  uint64_t residual_ts[SKIQ_MAX_NUM_CARDS];
  //! RX residual samples counter
  uint32_t residual_read_size[SKIQ_MAX_NUM_CARDS];

  
  // --------------------------------
  // Debug and output control
  // --------------------------------
  //! Number of underflows
  int num_underflows;
  //! Number of overflows
  int num_overflows;
  //! number of sequential errors
  int num_seq_errors;
  //! number of RX errors
  int num_rx_errors;
  //! Number of TX errors
  int num_tx_errors;

  //! timestamp of current TX
  uint64_t tx_current_ts;
  //! timestamp of current RX
  uint64_t rx_current_ts;
  //! number of actual samples transmitted
  uint64_t tx_actual_nsamps;
  //! number of actual samples received
  uint64_t rx_actual_nsamps;
  //! number of TX samples
  uint64_t tx_nsamps;
  //! number of RX samples
  uint64_t rx_nsamps;
  //! number of TX count
  uint64_t tx_count;
  //! number of RX count
  uint64_t rx_count;
  //! timestamp of RX packet
  openair0_timestamp rx_timestamp;
  //! size of TX packet blocks;
  uint32_t block_size_in_words;
  //! closest integer number of blocks
  uint32_t num_blocks_per_subframe;  
  //! Queue for TX requests
  TX_input_packet_q_t txq;
  //! mutex to protect txq modifications
  pthread_mutex_t tx_buffer_mutex;
  //! mutex for callback function
  pthread_mutex_t space_avail_mutex;
  //! condition variable for callback function
  pthread_cond_t space_avail_cond;
  //! pointers to Sidekiq TX packets
  tx_packet_t *tx_packet[SKIQ_MAX_NUM_TX_PACKETS];
  //! active status of SKIQ TX packet
  int txp_active[SKIQ_MAX_NUM_TX_PACKETS];
  //! TX activity indicator
  int tx_active;
  //! mutex for TX
  pthread_mutex_t tx_mutex;
  //! pthread structure for TX
  pthread_t tx_thread;
} skiq_state_t;

//! Number of Sidekiq devices 
int num_devices=0;


/*These items configure the underlying asynch stream used by the the sync interface. 
 */

/*! \brief Sidekiq Init function (not used at the moment)
 * \param device RF frontend parameters set by application
 * \returns 0 on success
 */
int trx_skiq_init(openair0_device *device) {
   return 0;
}

/*! \brief get current timestamp
 *\param device the hardware to use 
 *\returns timestamp of Sidekiq
 */
 
openair0_timestamp trx_get_timestamp(openair0_device *device) {
  return 0;
}

void skiq_dump_txpacket(void *txp,int len,FILE *fp) {

  printf("txp.idx %d\n",((tx_packet_t*)txp)->idx);
  printf("txp.skiq_pkt.meta 0x%" PRIx64 "\n",((tx_packet_t*)txp)->skiq_pkt.meta);
  printf("txp.skiq_pkt.ts %llu\n",((tx_packet_t*)txp)->skiq_pkt.ts);
  if (fp!=NULL) fwrite(txp,sizeof(skiq_tx_packet_t),1,fp);


  /*
  for (int i=0;i<len;i++) {
    if (i%10 == 0) printf("\n%d :",i);
    printf("%x.",((tx_packet_t*)txp)->skiq_pkt.iq[i]);
    
  }
  printf("\n");*/
}

#define DUMP_TX_FILE 1

void *skiq_tx_thread(void *arg) {

  skiq_state_t *skiq = (skiq_state_t *)arg;
  TX_input_packet_q_t *txq = &skiq->txq;
  tx_packet_t *txp_i;
  int i=0,j;
  int next;
  int len;
  int32_t res;
  struct sched_param sparam;
  int s;
  long long in,out;
  int tx_drop_cnt=0;
  int tx_cnt=0;
  uint32_t late;
#ifdef DUMP_TX_FILE
  int dump_cnt=0;
  FILE *fp=fopen("/tmp/skiq_txdebug.dat","w");
#endif
  memset(&sparam, 0, sizeof(sparam));
  sparam.sched_priority = sched_get_priority_max(SCHED_FIFO);

  printf("skiq_tx_thread: starting tx_thread (tx_active %d)\n",skiq->tx_active);
  s = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sparam);
  if (s !=0) {
    printf("skiq_tx_thraed: cannot set thread priority\n");
    skiq->tx_active=0;
    return((void*)NULL);
  }

  
  mlockall(MCL_CURRENT | MCL_FUTURE);

    // enable the Tx streaming
  if( skiq_start_tx_streaming(skiq->card_list[0], skiq_tx_hdl_A1) != 0 ){
      printf("Error: unable to start tx streaming\r\n");
      return((void*)NULL);
  }
  else skiq->tx_active=1;

  while (skiq->tx_active == 1) {

    if (tx_cnt > 1000) {
      tx_cnt = 0;
      skiq_read_tx_num_late_timestamps(skiq->card_list[0],skiq_tx_hdl_A1, 
				       &late);
      printf("skiq: num_late_timestamps %u\n",late);
    }
#ifdef DEBUG_SKIQ_TX
    printf(ANSI_COLOR_RED "skiq_tx_thread: locking mutex (time %llu) txq->elm[%d].active %d (len %d),skiq->txp_active[%d] %d,txq->elm[txq->head].timestamp %llun" ANSI_COLOR_RESET,rdtsc_oai(),
	   txq->head,
	   txq->elm[txq->head].active,
	   txq->elm[txq->head].length,
	   i,
	   skiq->txp_active[i],
	   txq->elm[txq->head].timestamp
	   );
    
#endif
    pthread_mutex_lock(&skiq->tx_mutex);

    uint64_t txts;
    skiq_read_curr_tx_timestamp(skiq->card_list[0],skiq_tx_hdl_A1, 
				&txts);
    in = rdtsc_oai();
#ifdef DEBUG_SKIQ_TX
    printf(ANSI_COLOR_RED "skiq_tx_thread: got mutex (time %llu) txq->elm[%d].active %d (len %d),skiq->txp_active[%d] %d,txq->elm[txq->head].timestamp %llu txts %llu\n" ANSI_COLOR_RESET,rdtsc_oai(),
	   txq->head,
	   txq->elm[txq->head].active,
	   txq->elm[txq->head].length,
	   i,
	   skiq->txp_active[i],
	   txq->elm[txq->head].timestamp,
	   txts);
#endif
    
    if ((txq->elm[txq->head].active==1)&&
	(skiq->txp_active[i]==0)) {
    //&& ((txq->elm[txq->head].timestamp-txts)<4*(skiq->block_size_in_words))) { // queue is not empty

      
#ifdef DEBUG_SKIQ_TX
      printf(ANSI_COLOR_RED "skiq_tx_thread: inner condition\n" ANSI_COLOR_RESET);
#endif	
      txp_i = skiq->tx_packet[i];
      
      if (txq->elm[txq->head].length >= skiq->block_size_in_words) {
	// there are enough samples in the head of the queue to fill the buffer and we're not too far in advance either
#ifdef DEBUG_SKIQ_TX	  
	printf(ANSI_COLOR_RED            "skiq_tx_thread: (head_length >= %d): Putting elm %d (tail %d) to TX packet %d/%d (length %d, TS %llu)\n"    ANSI_COLOR_RESET,
	       skiq->block_size_in_words,
	       txq->head,txq->tail,
	       i,
	       skiq->num_blocks_per_subframe,
	       txq->elm[txq->head].length,
	       txq->elm[txq->head].timestamp);
#endif	  
	skiq->txp_active[i]=1;
	//timestamp
        txp_i->skiq_pkt.meta = 0;
        txp_i->skiq_pkt.ts = txq->elm[txq->head].timestamp;
	//IQ data
#ifdef DEBUG_SKIQ_TX	  
	printf(ANSI_COLOR_RED            "skiq_tx_thread: txp_i %p => dataptr %p\n"    ANSI_COLOR_RESET,
	       txp_i,txq->elm[txq->head].dataptr);
#endif	  
	memcpy(txp_i->skiq_pkt.iq,
	       (void *)txq->elm[txq->head].dataptr,
	       skiq->block_size_in_words<<2);
	// update tx queue
	txq->elm[txq->head].dataptr   += skiq->block_size_in_words;
	txq->elm[txq->head].timestamp += skiq->block_size_in_words;
	txq->elm[txq->head].length    -= skiq->block_size_in_words;
#ifdef DEBUG_SKIQ_TX
	  printf("elm %d: writing %d words/ left %d to tx @%llu -> %p\n",txq->head,skiq->block_size_in_words,txq->elm[txq->head].length,txq->elm[txq->head].timestamp-skiq->block_size_in_words,
		 txq->elm[txq->head].dataptr-skiq->block_size_in_words);
#endif
	// copy skiq pointer in front of TX packet buffer
        txp_i->priv = (void *)skiq;
	// copy tx_packet index in front of TX packet buffer
        txp_i->idx = i;

	if (txq->elm[txq->head].length==0) {
	  // disactivate head element
	  txq->elm[txq->head].active=0;
	  // point head to next element in the queue
	  txq->head=(txq->head+1)%SKIQ_MAX_TX_ELM;
	}
	  
#ifdef DEBUG_SKIQ_TX
	printf(ANSI_COLOR_RED "skiq_tx_thread: unlocking mutex, time %llu\n" ANSI_COLOR_RESET,rdtsc_oai());
#endif	
	pthread_mutex_unlock(&skiq->tx_mutex);

#ifdef SKIQ_ASYNCH	
	if ((res=skiq_transmit(skiq->card_list[0], skiq_tx_hdl_A1,(int32_t*)&(txp_i->skiq_pkt))) == SKIQ_TX_ASYNC_SEND_QUEUE_FULL ) {
#ifdef DEBUG_SKIQ_TX
	  printf("skiq_tx_thread: send queue full, sleeping\n");
#endif
	  pthread_mutex_lock( &skiq->space_avail_mutex );
	  pthread_cond_wait( &skiq->space_avail_cond, &skiq->space_avail_mutex );
	  pthread_mutex_unlock( &skiq->space_avail_mutex );
	  // send packet again now that there is room
	  if ((res=skiq_transmit(skiq->card_list[0], skiq_tx_hdl_A1,(int32_t*)&(txp_i->skiq_pkt))) == SKIQ_TX_ASYNC_SEND_QUEUE_FULL )
	    printf("skiq_tx_thread: error, send queue still full after cond_signal, packet will be dropped\n");
	}
#else
	if ((res=skiq_transmit(skiq->card_list[0], skiq_tx_hdl_A1,(int32_t*)&(txp_i->skiq_pkt))) < 0 ) {

	  printf("skiq_tx_thread: skiq_transmit error, exiting\n");
	  skiq->tx_active=0;
	}
#ifdef DUMP_TX_FILE	
	if (dump_cnt<1+(153600/skiq->block_size_in_words)) {
	  printf("skiq_tx_thread: Dumping packet %d/%d\n",dump_cnt,153600/skiq->block_size_in_words);
	  skiq_dump_txpacket((void*)txp_i,skiq->block_size_in_words,fp);
	  dump_cnt++;
	}
#endif	
	skiq->txp_active[i]=0;
#endif
	tx_cnt++;
	
	i=(i+1)%skiq->num_blocks_per_subframe;
	
	out = rdtsc_oai();
#ifdef DEBUG_SKIQ_TX
	printf(ANSI_COLOR_RED           "skiq_tx_thread: received res %d from skiq_transmit(), time %d (in %llu)\n"  ANSI_COLOR_RESET,
	       res,(int) (out-in),in);
#endif
      }
      else {
	// empty head and continue with next element if its there

	
	// index of next element in the queue
	next =(txq->head+1)%SKIQ_MAX_TX_ELM;
	
	if (txq->elm[next].active>0) {
#ifdef DEBUG_SKIQ_TX
	  printf(ANSI_COLOR_RED              "skiq_tx_thread: (head_length < %d): Putting elm %d (tail %d) to TX packet %d/%d (length %d, TS %llu)\n"             ANSI_COLOR_RESET,
		 skiq->block_size_in_words,
		 txq->head,
		 txq->tail,
		 i,
		 skiq->num_blocks_per_subframe,
		 txq->elm[txq->head].length,
		 txq->elm[txq->head].timestamp);
#endif
	  // more than one active element in the queue
	  
	  
	  // length of residual amount in head of queue
	  len = txq->elm[txq->head].length;
	  
	  skiq->txp_active[i]=1;
	  
	  // timestamp of head elemen in queue
          txp_i->skiq_pkt.meta = 0;
          txp_i->skiq_pkt.ts = txq->elm[txq->head].timestamp;
	  // IQ data date of head element
	  memcpy(txp_i->skiq_pkt.iq,
		 (void *)txq->elm[txq->head].dataptr,
		 len<<2);
#ifdef DEBUG_SKIQ_TX
	  printf("elm %d: writing %d words/ left %d to tx @%llu -> %p\n",txq->head,len,len-txq->elm[txq->head].length,
		 txq->elm[txq->head].timestamp,
		 txq->elm[txq->head].dataptr);
#endif	  
	  // disactivate head element
	  txq->elm[txq->head].active=0;
	  // point head to next element in the queue
	  txq->head=next;
#ifdef DEBUG_SKIQ_TX
	  printf(ANSI_COLOR_RED          "skiq_tx_thread: (head_length < %d): Putting elm %d (tail %d) to TX packet %d/%d (length %d, TS %llu)\n"               ANSI_COLOR_RESET,
		 skiq->block_size_in_words,
		 txq->head,txq->tail,
		 i,
		 skiq->num_blocks_per_subframe,
		 skiq->block_size_in_words-len,
		 txq->elm[txq->head].timestamp);
#endif
	  // copy IQ data from new element
	  memcpy(&(txp_i->skiq_pkt.iq[len]),
		 (void*)txq->elm[txq->head].dataptr,
		 (skiq->block_size_in_words-len)<<2);
	  // update queue
	  txq->elm[txq->head].dataptr   += (skiq->block_size_in_words-len);
	  txq->elm[txq->head].timestamp += (skiq->block_size_in_words-len);
	  txq->elm[txq->head].length    -= (skiq->block_size_in_words-len);
#ifdef DEBUG_SKIQ_TX
	  printf("elm %d : writing %d words/ left %d to tx @%llu -> %p\n",txq->head,skiq->block_size_in_words-len,txq->elm[txq->head].length,
		 txq->elm[txq->head].timestamp-(skiq->block_size_in_words-len),
		 txq->elm[txq->head].dataptr-(skiq->block_size_in_words-len));
#endif
	  // copy skiq pointer in front of TX packet buffer
          txp_i->priv = (void *)skiq;
	  // copy tx_packet index in front of TX packet buffer
          txp_i->idx = i;
	  
#ifdef DEBUG_SKIQ_TX
	  printf(ANSI_COLOR_RED                   "skiq_tx_thread: Unlocking tx_mutex\n"               ANSI_COLOR_RESET);
#endif	  
	  pthread_mutex_unlock(&skiq->tx_mutex);

#ifdef SKIQ_ASYNCH

	  if ((res=skiq_transmit(skiq->card_list[0], skiq_tx_hdl_A1,(int32_t*)&(txp_i->skiq_pkt))) == SKIQ_TX_ASYNC_SEND_QUEUE_FULL ) {
#ifdef DEBUG_SKIQ_TX
	    printf("skiq_tx_thread: send queue full, sleeping\n");
#endif
	    pthread_mutex_lock( &skiq->space_avail_mutex );
	    pthread_cond_wait( &skiq->space_avail_cond, &skiq->space_avail_mutex );
	    pthread_mutex_unlock( &skiq->space_avail_mutex );
	  }
#else
	  if ((res=skiq_transmit(skiq->card_list[0], skiq_tx_hdl_A1,(int32_t*)&(txp_i->skiq_pkt))) < 0 ) {
	    
	    printf("skiq_tx_thread: skiq_transmit error, exiting\n");
	    skiq->tx_active=0;
	  }
	  skiq->txp_active[i]=0;
#endif	  
	  i=(i+1)%skiq->num_blocks_per_subframe;
#ifdef DEBUG_SKIQ_TX
	  printf(ANSI_COLOR_RED  "skiq_tx_thread: received res %d from skiq_transmit(), time now %llu\n" ANSI_COLOR_RESET,
		 res,rdtsc_oai());
#endif
	}
#ifdef DEBUG_SKIQ_TX
	//	printf(ANSI_COLOR_RED                   "Unlocking tx_mutex + sleeping, time %llu\n"               ANSI_COLOR_RESET,rdtsc_oai());
#endif	  
	pthread_mutex_unlock(&skiq->tx_mutex);
	usleep(100);
#ifdef DEBUG_SKIQ_TX
	//	printf(ANSI_COLOR_RED                   "skiq_tx_thread: waking up, time %llu\n"               ANSI_COLOR_RESET,rdtsc_oai());
#endif	  
      }
    } // txp_active[i]=0
    else {
#ifdef DEBUG_SKIQ_TX
      //      printf(ANSI_COLOR_RED                   "skiq_tx_thread: Unlocking tx_mutex + sleeping, time %llu\n"               ANSI_COLOR_RESET,rdtsc_oai());
#endif	  
      pthread_mutex_unlock(&skiq->tx_mutex);
      usleep(100);
#ifdef DEBUG_SKIQ_TX
      //      printf(ANSI_COLOR_RED                   "skiq_tx_thread: waking up, time %llu\n"               ANSI_COLOR_RESET,rdtsc_oai());
#endif	        
    }
  }
  printf(ANSI_COLOR_RED           "skiq_tx_thread: returning\n"            ANSI_COLOR_RESET);
#ifdef DUMP_TX_FILE	
  fclose(fp);
#endif
  return((void*)NULL);
}


/*! \brief Start Sidekiq
 * \param device the hardware to use 
 * \returns 0 on success
 */
int trx_skiq_start(openair0_device *device) {

  skiq_state_t *skiq = (skiq_state_t*)device->priv;
  
  skiq_write_chan_mode(skiq->card_list[0], skiq_chan_mode_single);
  // set 5ms timeout on read
  //  skiq_set_rx_transfer_timeout(skiq->card_list[0], 5000);

  skiq->rx_current_ts=0;
  skiq->rx_actual_nsamps=0;
  skiq->rx_nsamps=0;
  skiq->rx_count=0;
  skiq->residual_read_size[skiq->card_list[0]] = 0;

  if(skiq_reset_timestamps(skiq->card_list[0]) != 0 ) {
      printf("Error: unable to reset the timestamps\r\n");
      return (-1);
  }

  // create the tx thread
  pthread_create(&skiq->tx_thread,NULL,skiq_tx_thread,(void*)skiq);
  

  if ( skiq_start_rx_streaming(skiq->card_list[0],skiq_rx_hdl_A1) != 0 ){
      printf("Error: unable to start rx streaming\r\n");
      return (-1);
  }
  
  return 0;
}



int skiq_add_tx_el(skiq_state_t *skiq, openair0_timestamp ptimestamp,void **buff,int nsamps) {

  TX_input_packet_q_t *txq = &skiq->txq;
  int res=0;

#ifdef DEBUG_SKIQ_TX
  printf(ANSI_COLOR_BLUE "skiq_add_tx_el: Locking TX mutex, time %llu\n" ANSI_COLOR_RESET,rdtsc_oai());
#endif
  pthread_mutex_lock(&skiq->tx_mutex);
#ifdef DEBUG_SKIQ_TX
  printf(ANSI_COLOR_BLUE "skiq_add_tx_el: Unlocked TX mutex, time %llu\n" ANSI_COLOR_RESET,rdtsc_oai());
#endif
  
  if (((txq->tail+1) % SKIQ_MAX_TX_ELM) != txq->head) { // queue is not full

#ifdef DEBUG_SKIQ_TX    
    printf(ANSI_COLOR_BLUE  "skiq_add_tx_el: Adding element at time %lu and size %d to txq (head %d, tail %d, SKIQ_MAX_TX_ELM %d), buff %p\n" ANSI_COLOR_RESET,
	   (uint64_t)ptimestamp,nsamps,txq->head,txq->tail,SKIQ_MAX_TX_ELM,buff[0]);
#endif
    txq->elm[txq->tail].dataptr   = buff[0];
    txq->elm[txq->tail].timestamp = ptimestamp;
    txq->elm[txq->tail].length    = nsamps;
    txq->elm[txq->tail].active    = 1;
    
    txq->tail = (txq->tail+1) % SKIQ_MAX_TX_ELM;
  }
  else {
    printf("TX queue is full, dropping element\n");
    res=-1;
  }

  /*
  int diff = (SKIQ_MAX_TX_ELM + txq->tail - txq->head)%SKIQ_MAX_TX_ELM;
  if (diff > 2) {
    txq->head = (txq->head+1)%SKIQ_MAX_TX_ELM;
    printf("dropping TX queue head\n");
    }*/
  
  pthread_mutex_unlock(&skiq->tx_mutex);
#ifdef DEBUG_SKIQ_TX
  printf(ANSI_COLOR_BLUE "skiq_add_tx_el: unlocked TX mutex, time %llu\n" ANSI_COLOR_RESET,rdtsc_oai());
#endif
  return(res);
}

/*! \brief Called to send samples to the Sidekiq RF target
      \param device pointer to the device structure specific to the RF hardware target
      \param timestamp The timestamp at whicch the first sample MUST be sent 
      \param buff Buffer which holds the samples
      \param nsamps number of samples to be sent
      \param cc index of the component carrier
      \param flags Ignored for the moment
      \returns 0 on success
*/

static int trx_skiq_write(openair0_device *device,openair0_timestamp ptimestamp, void **buff, int nsamps, int cc, int flags) {    

  skiq_state_t *skiq = (skiq_state_t*)device->priv;
  /* SKIQ has only 1 rx/tx chaine : is it correct? */
  
#ifdef DEBUG_SKIQ_TX
  uint32_t late;
  uint64_t txts;

  /*
  skiq_read_tx_num_late_timestamps(skiq->card_list[0],skiq_tx_hdl_A1, 
				   &late);
  skiq_read_curr_tx_timestamp(skiq->card_list[0],skiq_tx_hdl_A1, 
			      &txts);
  */
  
  printf(ANSI_COLOR_BLUE     "trx_skiq_write: Writing buff %p (%p) @ %llu (time now %llu, SKIQ TS %llu, tx_active %d), late %d\n" ANSI_COLOR_RESET,
	 buff,buff[0],ptimestamp,rdtsc_oai(),
	 txts,
	 late,
	 skiq->tx_active);
#endif

  if (skiq->tx_active == 0) {
    printf("TX not active yet, dropping TX packet\n");
    return(nsamps);
  }
  // add write to queue
  if (skiq_add_tx_el(skiq,ptimestamp,buff,nsamps) < 0) {
    printf("TX buffer full, exiting\n");
    skiq->tx_active=0;
    sleep(5);
    return(0);
  }


  
  //  skiq_send_tx(skiq);
  
  skiq->tx_current_ts=0;
  skiq->tx_actual_nsamps+=0;
  skiq->tx_nsamps+=nsamps;
  skiq->tx_count++;
  

  return(nsamps);
}

typedef struct {

  uint64_t rf_timestamp;
  uint64_t sys_timestamp;
  uint32_t sys_word;
  uint32_t user_word;
} skiq_read_header_t;

/*! \brief Receive samples from hardware.
 * Read \ref nsamps samples from each channel to buffers. buff[0] is the array for
 * the first channel. *ptimestamp is the time at which the first sample
 * was received.
 * \param device the hardware to use
 * \param[out] ptimestamp the time at which the first sample was received.
 * \param[out] buff An array of pointers to buffers for received samples. The buffers must be large enough to hold the number of samples \ref nsamps.
 * \param nsamps Number of samples. One sample is 2 byte I + 2 byte Q => 4 byte.
 * \param cc  Index of component carrier
 * \returns number of samples read
*/
static int trx_skiq_read(openair0_device *device, openair0_timestamp *ptimestamp, void **buff, int nsamps, int cc) {

  skiq_rx_status_t status;
  skiq_state_t *skiq = (skiq_state_t*)device->priv;
  
  uint8_t *buf;
  uint64_t curr_ts;
  skiq_rx_hdl_t hdl = skiq_rx_hdl_A1;
  uint32_t total_len = 0;
  uint32_t len;
  uint32_t nsamps_block=0,nsamps_used=0;
  int res; 
  int gap;
  
  while (total_len < nsamps) {

    res = skiq->residual_read_size[skiq->card_list[0]];
    // handle residual samples stored from last read
    if (res > 0) {
      if (res <= nsamps)
	total_len = res;
      else
	total_len = nsamps;
      
      memcpy(buff[0],
	     (void*)skiq->residual_read_buffer[skiq->card_list[0]],
	     total_len<<2);
      *ptimestamp = skiq->residual_ts[skiq->card_list[0]];
      skiq->rx_current_ts = skiq->residual_ts[skiq->card_list[0]] + total_len;

#ifdef DEBUG_SKIQ_RX      
      printf(ANSI_COLOR_BLUE  "trx_skiq_read: requested (residual) %d samps, res %d (total so far %d), ts %lu, next_ts %lu\n"  ANSI_COLOR_RESET ,nsamps,res,total_len,
	     *ptimestamp,skiq->rx_current_ts);
#endif	
      if (res >= nsamps) { // the read was less than what was left so update residual buffer and return
	skiq->residual_read_size[skiq->card_list[0]]-=nsamps;
	memmove((void*)skiq->residual_read_buffer[skiq->card_list[0]],
		(void*)skiq->residual_read_buffer[skiq->card_list[0]]+(total_len<<2),
		skiq->residual_read_size[skiq->card_list[0]]<<2);
	return(nsamps);
      }
      else {
	skiq->residual_read_size[skiq->card_list[0]]=0;
      }
    }

    gap          = 0;
    status       = skiq_receive(skiq->card_list[0], &hdl, &buf, &len);
    nsamps_block = (len-sizeof(skiq_read_header_t))>>2;
    
  
    if( status == skiq_rx_status_success ) {
      if( buf != NULL ) {

	curr_ts = *((uint64_t*)(buf));
#ifdef DEBUG_SKIQ_RX      
	printf("trx_skiq_read: requested %d samps, got %d (total so far %d), curr_ts %llu, expected %llu\n",nsamps,nsamps_block,total_len,
	       curr_ts,skiq->rx_current_ts);
#endif	
	if ((skiq->rx_count > 0) &&
	    (curr_ts != skiq->rx_current_ts)) { // handel gap in received timestamp
	  printf("SKIQ Error: timestamp error expected 0x%016lx but got 0x%016lx.\r\n", 
		 skiq->rx_current_ts, curr_ts);
	  gap = (int)(curr_ts-skiq->rx_current_ts);
	  printf("SKIQ: gap of %d samples in timestamp, adjusting\n",gap);
	  printf("SKIQ: total samples received %lu\n",  skiq->rx_nsamps);
	  printf("SKIQ: system word %x\n", ((skiq_read_header_t*)buf)->sys_word);

	  //	  return (-1);
        }
      }
      else {
	printf("Error: skiq_received returned NULL pointer\r\n");
	return (-1);
      }

      gap=0;
      // copy buffer
      if (total_len + nsamps_block + gap <= nsamps)
	nsamps_used = nsamps_block;
      else
	nsamps_used = nsamps - total_len;

      uint32_t tmp,i;
      uint32_t *in = (uint32_t*)(((void*)buf)+sizeof(skiq_read_header_t));
      uint32_t *out = (uint32_t*)(((void*)buff[0])+(total_len<<2));
      //flip I/Q positions
      for (i=0;i<nsamps_used;i++) {
	tmp = in[i];
	out[i] = (tmp<<16)|(tmp>>16);
      }
	/*
      memcpy(buff[0]+((gap+total_len)<<2),
	     ((void*)buf)+sizeof(skiq_read_header_t),
	     nsamps_used<<2);*/
	
      if (gap>0)
	memset(buff[0]+(total_len<<2),0,gap<<2);
      
      if (total_len == 0) { // this is the first read so copy timestamp
	*ptimestamp = (openair0_timestamp)curr_ts;
      }
      total_len += nsamps_used+gap;
      
      skiq->rx_current_ts = curr_ts+nsamps_used+gap;
      skiq->rx_count++;
      skiq->rx_actual_nsamps+=(nsamps_used+gap);
	
      if (status == skiq_rx_status_error_overrun)
	printf("status : overrun, exiting ...\n");
      else if (status == skiq_rx_status_error_generic)
	printf("status : generic error, exiting ...\n");
    } // skiq_rx_status_success
  } // total_len < nsamps

#ifdef DEBUG_SKIQ_RX      
  printf("trx_skiq_read: (residual update): nsamps_used %d, nsamps_block %d\n",nsamps_used,nsamps_block);
#endif
  
  if ((nsamps_used+gap) < nsamps_block) { //update residual counters
    skiq->residual_read_size[skiq->card_list[0]] = nsamps_block-nsamps_used-gap;
    skiq->residual_ts[skiq->card_list[0]] = skiq->rx_current_ts;
    memcpy((void*)skiq->residual_read_buffer[skiq->card_list[0]],
	   ((void*)buf)+sizeof(skiq_read_header_t),
	   (nsamps_block-nsamps_used)<<2);
  }
  
  skiq->rx_nsamps+=nsamps;

#ifdef DEBUG_SKIQ_TX
  printf(ANSI_COLOR_BLUE "trx_skiq_read: returning %d samples @ %llu\n" ANSI_COLOR_RESET,nsamps,*ptimestamp);
#endif
  return nsamps;

}

/*! \brief print the Sidekiq statistics  
* \param device the hardware to use
* \returns  0 on success
*/
int trx_skiq_get_stats(openair0_device* device) {

  return(0);

}

/*! \brief Reset the Sidekiq statistics  
* \param device the hardware to use
* \returns  0 on success
*/
int trx_skiq_reset_stats(openair0_device* device) {

  return(0);

}


/*! \brief Stop Sidekiq
 * \param card the hardware to use
 * \returns 0 in success 
 */
int trx_skiq_stop(openair0_device *device) {
  
  skiq_state_t *skiq = (skiq_state_t*)device->priv;

  printf("SKIQ: Stopping RX streaming now\n");
  
  skiq_stop_rx_streaming(skiq->card_list[0],skiq_rx_hdl_A1);

  printf("SKIQ: Stopping TX streaming now\n");
  skiq->tx_active=0;
  skiq_stop_tx_streaming(skiq->card_list[0],skiq_tx_hdl_A1);  
  return(0);

}

/*! \brief Terminate operation of the Sidekiq transceiver -- free all associated resources 
 * \param device the hardware to use
 */
void trx_skiq_end(openair0_device *device) {

  skiq_state_t *skiq = (skiq_state_t*)device->priv;
  // Disable RX module, shutting down our underlying RX stream
  int i;

  if (skiq->tx_active==1) {
    printf("SKIQ: skiq_end, stopping device\n");
    trx_skiq_stop(device);
    sleep(1);
  }
  
  for (i=0;i<skiq->num_blocks_per_subframe;i++) {
    free(skiq->tx_packet[i]);
  }

  pthread_mutex_destroy(&skiq->tx_mutex);
  pthread_mutex_destroy(&skiq->space_avail_mutex);
  pthread_cond_destroy(&skiq->space_avail_cond);
  skiq_exit();
  sleep(1);
  free((void*)device->priv);
}

/*! \brief Set frequencies (TX/RX)
 * \param device the hardware to use
 * \param openair0_cfg1 openair0 Config structure (ignored. It is there to comply with RF common API)
 * \param exmimo_dump_config (ignored)
 * \returns 0 in success 
 */
int trx_skiq_set_freq(openair0_device* device, openair0_config_t *openair0_cfg1,int exmimo_dump_config) {

  skiq_state_t *skiq = (skiq_state_t *)device->priv;
  openair0_config_t *openair0_cfg = (openair0_config_t *)device->openair0_cfg;
  int result;
  int cardid=0;
  
  if ((result=skiq_write_rx_LO_freq(skiq->card_list[cardid],
				    skiq_rx_hdl_A1,
				    (uint64_t)openair0_cfg->rx_freq[0]) < 0)) 
    { 
      printf("Error: failed to set Rx LO freq to %llu Hz\n",(unsigned long long)openair0_cfg->rx_freq[0]); 
      return(-1); 
    }
    
  return(0);

}

/*! \brief Set Gains (TX/RX)
 * \param device the hardware to use
 * \param openair0_cfg openair0 Config structure
 * \returns 0 in success 
 */
int trx_skiq_set_gains(openair0_device* device, openair0_config_t *openair0_cfg) {

  return(0);

}


void skiq_tx_complete(int32_t status,uint32_t *p_data) {

  tx_packet_t *txp_i;
  skiq_tx_packet_t *stp = (skiq_tx_packet_t *)p_data;
  skiq_state_t *skiq;
  
  if (p_data) {
    txp_i = (tx_packet_t*) container_of(stp,tx_packet_t,skiq_pkt);
    skiq = (skiq_state_t *)(txp_i->priv);
    if (skiq) {
#ifdef DEBUG_SKIQ_TX
      printf("skiq_tx_complete: packet %d (%p) received with status %x, clearing active flag (%d => 0)\n",
	     txp_i->idx,p_data,status,skiq->txp_active[txp_i->idx]);
#endif
      skiq->txp_active[txp_i->idx]=0;
      
      pthread_cond_signal(&skiq->space_avail_cond);
    }
  }
  else {
    printf("skiq_tx_complete: received NULL p_data (%p)\n",p_data);
  }
  
}


/*! \brief calibration table for Sidekiq */
rx_gain_calib_table_t calib_table_skiq[] = {
  {2300000000.0,49.5},
  {1880000000.0,47.5},
  {816000000.0,57.5},
  {-1,0}};

/*! \brief set RX gain offset from calibration table
 * \param openair0_cfg RF frontend parameters set by application
 * \param chain_index RF chain ID
 */
void set_rx_gain_offset(openair0_config_t *openair0_cfg, int chain_index) {

  int i=0;
  // loop through calibration table to find best adjustment factor for RX frequency
  double min_diff = 6e9,diff;
  
  while (openair0_cfg->rx_gain_calib_table[i].freq>0) {
    diff = fabs(openair0_cfg->rx_freq[chain_index] - openair0_cfg->rx_gain_calib_table[i].freq);
    printf("cal %d: freq %f, offset %f, diff %f\n",
	   i,
	   openair0_cfg->rx_gain_calib_table[i].freq,
	   openair0_cfg->rx_gain_calib_table[i].offset,diff);
    if (min_diff > diff) {
      min_diff = diff;
      openair0_cfg->rx_gain_offset[chain_index] = openair0_cfg->rx_gain_calib_table[i].offset;
    }
    i++;
  }
  
}

/*! \brief Initialize Openair Sidekiq target. It returns 0 if OK 
 * \param device the hardware to use
 * \param openair0_cfg RF frontend parameters set by application
 * \returns 0 on success
 */
int device_init(openair0_device *device, openair0_config_t *openair0_cfg) {

  skiq_state_t *skiq = (skiq_state_t*)malloc(sizeof(skiq_state_t));
  
  memset(skiq, 0, sizeof(skiq_state_t));
  pthread_mutex_init(&skiq->tx_mutex,NULL);
  pthread_mutex_init(&skiq->space_avail_mutex,NULL);
  pthread_cond_init(&skiq->space_avail_cond,NULL);
  /* device specific */
  //openair0_cfg->txlaunch_wait = 1;//manage when TX processing is triggered
  //openair0_cfg->txlaunch_wait_slotcount = 1; //manage when TX processing is triggered
  openair0_cfg->iq_txshift = 0;// shift
  openair0_cfg->iq_rxrescale = 15;//rescale iqs
  
  // init required params
  switch ((int)openair0_cfg->sample_rate) {
  case 30720000:
    openair0_cfg->tx_sample_advance           = 0;
    skiq->block_size_in_words                 = SKIQ_BLOCK_SIZE_IN_WORDS;
    skiq->num_blocks_per_subframe             = 30720/(skiq->block_size_in_words);
    break;
  case 23040000:
    openair0_cfg->tx_sample_advance           = 0;
    skiq->block_size_in_words                 = SKIQ_BLOCK_SIZE_IN_WORDS;
    skiq->num_blocks_per_subframe             = 23040/(skiq->block_size_in_words);
    break;    
  case 15360000:
    openair0_cfg->tx_sample_advance           = 0;
    skiq->block_size_in_words                 = SKIQ_BLOCK_SIZE_IN_WORDS;
    skiq->num_blocks_per_subframe             = 15360/(skiq->block_size_in_words);
    break;
  case 7680000:
    openair0_cfg->tx_sample_advance           = 0;
    skiq->block_size_in_words                 = SKIQ_BLOCK_SIZE_IN_WORDS;
    skiq->num_blocks_per_subframe             = 7680/(skiq->block_size_in_words);
    break;
  case 1920000:
    openair0_cfg->tx_sample_advance           = 0;
    skiq->block_size_in_words                 = 256-4;
    skiq->num_blocks_per_subframe             = 1920/(skiq->block_size_in_words);
    break;
  default:
    printf("Error: unsupported sampling rate %f\n",openair0_cfg->sample_rate);
    exit(-1);
    break;
  }
  openair0_cfg->iq_txshift= 0;
  openair0_cfg->iq_rxrescale = 15; /*not sure*/
  openair0_cfg->rx_gain_calib_table = calib_table_skiq;


  // open device

  printf("[SKIQ] init dev \n");

  /* probe the host system for installed Sidekiq cards */
  skiq_probe(true, false);
  /* query the list of all Sidekiq cards */
  skiq_get_avail_cards( &skiq->number_of_cards, skiq->card_list );

  if (skiq->number_of_cards>0)
    printf("Initializing %d Sidekiq cards\n",skiq->number_of_cards);
  else {
    printf("Found no Sidekiq cards, exiting\n");
    return(-1);
  }
  
  skiq_init(skiq_pcie_init_level_2, skiq_usb_init_level_0,skiq->card_list,skiq->number_of_cards);

  int32_t result;
  uint32_t p_git_hash,p_build_date;
  uint8_t p_major,p_minor,p_patch;
  skiq_fpga_tx_fifo_size_t p_tx_fifo_size;
  
  for (int cardid=0;cardid<skiq->number_of_cards;cardid++) {
    if ((result=skiq_read_fpga_version(skiq->card_list[cardid],
				       &p_git_hash,
				       &p_build_date,
				       &p_major,
				       &p_minor,
				       &p_tx_fifo_size))<0)
      {
	printf("SKIQ Error: failed to get FPGA version information\n");
	return(-1); 
      }
    else {
      printf("Sidekiq FPGA (git hash %x, build data %d, major %d, minor %d, fifo size %d kwords\n",
	     p_git_hash,p_build_date,p_major,p_minor,p_tx_fifo_size == 0 ? -1 : 2<<p_tx_fifo_size);
    }

    if ((result=skiq_read_libsidekiq_version(&p_major,
					     &p_minor,
					     &p_patch))<0)
      {
	printf("SKIQ Error: failed to get libsidekiq version information\n");
	return(-1); 
      }
    else {
      printf("lisidekiq info (major %d, minor %d, patch %d\n",
	     p_major,p_minor,p_patch);
    }
    
    if((result=skiq_write_rx_sample_rate_and_bandwidth(skiq->card_list[cardid],
						       skiq_rx_hdl_A1,
						       (uint32_t)openair0_cfg->sample_rate,
						       (uint32_t)openair0_cfg->rx_bw) < 0)) 
      { 
	printf("SKIQ Error: failed to set rx sample rate to %d Hz\n",(uint32_t)openair0_cfg->sample_rate); 
	return(-1); 
      }
    else {
	printf("SKIQ set rx sample rate to %d Hz\n",(uint32_t)openair0_cfg->sample_rate); 
    }
    if((result=skiq_write_tx_sample_rate_and_bandwidth(skiq->card_list[cardid],
						       skiq_tx_hdl_A1,
						       (uint32_t)openair0_cfg->sample_rate,
						       (uint32_t)openair0_cfg->tx_bw) < 0)) 
      { 
	printf("SKIQ Error: failed to set tx sample rate to %d Hz\n",(uint32_t)openair0_cfg->sample_rate); 
	return(-1); 
      }
    else {
	printf("SKIQ set tx sample rate to %d Hz\n",(uint32_t)openair0_cfg->sample_rate); 
    }
    
    if ((result=skiq_write_rx_LO_freq(skiq->card_list[cardid],
				      skiq_rx_hdl_A1,
				      (uint64_t)openair0_cfg->rx_freq[0]) < 0)) 
      { 
	printf("SKIQ Error: failed to set Rx LO freq to %llu Hz\n",(unsigned long long)openair0_cfg->rx_freq[0]); 
	return(-1); 
      }
    else {
      printf("SKIQ: set Rx LO freq to %llu Hz\n",(unsigned long long)openair0_cfg->rx_freq[0]); 
    }
    if ((result=skiq_write_tx_LO_freq(skiq->card_list[cardid],
				      skiq_tx_hdl_A1,
				      (uint64_t)openair0_cfg->tx_freq[0]) < 0)) 
      { 
	printf("SKIQ Error: failed to set tx LO freq to %llu Hz\n",(unsigned long long)openair0_cfg->tx_freq[0]); 
	return(-1); 
      }
    else {
	printf("SKIQ: set tx LO freq to %llu Hz\n",(unsigned long long)openair0_cfg->tx_freq[0]); 
    }
    if( (result=skiq_write_tx_data_flow_mode(skiq->card_list[cardid], skiq_tx_hdl_A1, 
                                                   skiq_tx_with_timestamps_data_flow_mode)) != 0 )
    {
        printf("Error: unable to configure Tx data flow mode to with_timestatmps\r\n");
        return (-1);
    }
    if( (result=skiq_write_tx_block_size(skiq->card_list[cardid], skiq_tx_hdl_A1, 
                                         skiq->block_size_in_words)) != 0 )
    {
        printf("Error: unable to configure Tx block size\r\n");
        return (-1);
    }

    if( (result=skiq_disable_tx_tone(skiq->card_list[cardid], skiq_tx_hdl_A1)) != 0)
      {
        printf("Error: unable to disable tx_tone\r\n");
        return (-1);
      }
#ifdef SKIQ_ASYNCH
    // set the transfer mode to async
    if( skiq_write_tx_transfer_mode(skiq->card_list[cardid], skiq_tx_hdl_A1, skiq_tx_transfer_mode_async) != 0 )
    {
        printf("Error: unable to set transfer mode to async\r\n");
        return (-1);
    }
    
    // register the callback
    if( skiq_register_tx_complete_callback(skiq->card_list[cardid], &skiq_tx_complete ) != 0 )
    {
        printf("Error: unable to register callback\r\n");
        return (-1);
    }
    if( skiq_write_num_tx_threads( skiq->card_list[cardid],
				   2) != 0)
    {
      printf("Error: unable to set number of TX threads to 2\r\n");
      return (-1);
    }    
#else
    // set the transfer mode to sync
    if( skiq_write_tx_transfer_mode(skiq->card_list[cardid], skiq_tx_hdl_A1, skiq_tx_transfer_mode_sync) != 0 )
    {
        printf("Error: unable to set transfer mode to sync\r\n");
        return (-1);
    }
#endif    

    if ((result=skiq_write_rx_gain_mode(skiq->card_list[cardid],
					skiq_rx_hdl_A1,
					skiq_rx_gain_manual) < 0)) 
      { 
	printf("SKIQ Error: failed to set Rx gain mode to manual\n"); 
	return(-1); 
      }
    else {
      printf("SKIQ: Set RX gain mode to manual\n");
    }
    set_rx_gain_offset(&openair0_cfg[0],0);

    if ((result=skiq_write_rx_gain(skiq->card_list[cardid],
				   skiq_rx_hdl_A1,
				   (uint32_t)openair0_cfg->rx_gain[0]-(int32_t)openair0_cfg[0].rx_gain_offset[0]) < 0)) 
      { 
	printf("SKIQ Error: failed to set Rx gain to %u (offset %d)\n",(uint32_t)openair0_cfg->rx_gain[0],(int32_t)openair0_cfg[0].rx_gain_offset[0]); 
	return(-1); 
      }
    else {
      printf("SKIQ Set rx gain to %u dB (offset %d)\n",(uint32_t)openair0_cfg->rx_gain[0]-(int32_t)openair0_cfg[0].rx_gain_offset[0],(int32_t)openair0_cfg[0].rx_gain_offset[0]); 
    }

    if ((result=skiq_write_tx_attenuation(skiq->card_list[cardid],
					  skiq_rx_hdl_A1,
					  0)))
      { 
	printf("SKIQ Error: failed to set tx attenuation to %u\n",0);
	return(-1); 
      }
    else {
      printf("SKIQ Set tx attenuation %u dB\n",0); 
    }
    
  }
  
  printf("SKIQ: Initializing openair0_device\n");
  device->Mod_id         = num_devices++;
  device->type             = SKIQ_DEV; 
  device->trx_start_func = trx_skiq_start;
  device->trx_end_func   = trx_skiq_end;
  device->trx_read_func  = trx_skiq_read;
  device->trx_write_func = trx_skiq_write;
  device->trx_get_stats_func   = trx_skiq_get_stats;
  device->trx_reset_stats_func = trx_skiq_reset_stats;
  device->trx_stop_func        = trx_skiq_stop;
  device->trx_set_freq_func    = trx_skiq_set_freq;
  device->trx_set_gains_func   = trx_skiq_set_gains;
  device->openair0_cfg = openair0_cfg;
  device->priv = (void *)skiq;

  // initializing tx buffers

  for (int i=0;i<skiq->num_blocks_per_subframe;i++) {
    // allocate buffer for TX packets, 4 = header in 32-bit words, pointer and index at end
      skiq->tx_packet[i] = (tx_packet_t *)malloc(sizeof(tx_packet_t));
  }

  //  memcpy((void*)&device->openair0_cfg,(void*)&openair0_cfg[0],sizeof(openair0_config_t));

  return 0;
}

/*! \brief skiq error report 
 * \param status 
 * \returns 0 on success
 */
int skiq_error(int status) {
  
  //exit(-1);
  return status; // or status error code
}

 

/*@}*/
