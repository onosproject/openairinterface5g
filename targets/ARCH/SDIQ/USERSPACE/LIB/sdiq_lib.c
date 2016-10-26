/*******************************************************************************
    OpenAirInterface 
    Copyright(c) 1999 - 2014 Eurecom

    OpenAirInterface is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.


    OpenAirInterface is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with OpenAirInterface.The full GNU General Public License is 
    included in this distribution in the file called "COPYING". If not, 
    see <http://www.gnu.org/licenses/>.

   Contact Information
   OpenAirInterface Admin: openair_admin@eurecom.fr
   OpenAirInterface Tech : openair_tech@eurecom.fr
   OpenAirInterface Dev  : openair4g-devel@lists.eurecom.fr
  
   Address      : Eurecom, Campus SophiaTech, 450 Route des Chappes, CS 50193 - 06904 Biot Sophia Antipolis cedex, FRANCE

 *******************************************************************************/

/** sdiq_lib.c
 *
 * Author: Raymond Knopp */


#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "math.h"

/** @addtogroup _SIDEKIQ_PHY_RF_INTERFACE_
 * @{
 */


#ifdef __SSE4_1__
#  include <smmintrin.h>
#endif
 
#ifdef __AVX2__
#  include <immintrin.h>
#endif

//! Number of Sidekiq devices 
int num_devices=0;

/*These items configure the underlying asynch stream used by the the sync interface. 
 */

/*! \brief Sidekiq Init function (not used at the moment)
 * \param device RF frontend parameters set by application
 * \returns 0 on success
 */
int trx_sdiq_init(openair0_device *device) {
   return 0;
}

/*! \brief get current timestamp
 *\param device the hardware to use 
 *\param module the bladeRf module
 *\returns timestamp of Sidekiq
 */
 
openair0_timestamp trx_get_timestamp(openair0_device *device, bladerf_module module) {
  int status;
  struct bladerf_metadata meta;
  sdiq_state_t *sdiq = (sdiq_state_t*)device->priv;
  memset(&meta, 0, sizeof(meta));
  
  if ((status=bladerf_get_timestamp(sdiq->dev, module, &meta.timestamp)) != 0) {
    fprintf(stderr,"Failed to get current %s timestamp: %s\n",(module == BLADERF_MODULE_RX ) ? "RX" : "TX", bladerf_strerror(status));
    return -1; 
  } // else {printf("Current RX timestampe  0x%016"PRIx64"\n", meta.timestamp); }

  return meta.timestamp;
}

/*! \brief Start Sidekiq
 * \param device the hardware to use 
 * \returns 0 on success
 */
int trx_sdiq_start(openair0_device *device) {

  return 0;
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
static int trx_sdiq_write(openair0_device *device,openair0_timestamp ptimestamp, void **buff, int nsamps, int cc, int flags) {
  
  int status;
  sdiq_state_t *sdiq = (sdiq_state_t*)device->priv;
  /* SDIQ has only 1 rx/tx chaine : is it correct? */
  int16_t *samples = (int16_t*)buff[0];
  
  //memset(&sdiq->meta_tx, 0, sizeof(sdiq->meta_tx));
  // When  BLADERF_META_FLAG_TX_NOW is used the timestamp is not used, so one can't schedule a tx 
  if (sdiq->meta_tx.flags == 0 ) 
    sdiq->meta_tx.flags = (BLADERF_META_FLAG_TX_BURST_START);// | BLADERF_META_FLAG_TX_BURST_END);// |  BLADERF_META_FLAG_TX_NOW);
  
  
  sdiq->meta_tx.timestamp= (uint64_t) (ptimestamp); 
  status = bladerf_sync_tx(sdiq->dev, samples, (unsigned int) nsamps, &sdiq->meta_tx, 2*sdiq->tx_timeout_ms);
 
  if (sdiq->meta_tx.flags == BLADERF_META_FLAG_TX_BURST_START) 
    sdiq->meta_tx.flags =  BLADERF_META_FLAG_TX_UPDATE_TIMESTAMP;
  

  if (status != 0) {
    //fprintf(stderr,"Failed to TX sample: %s\n", bladerf_strerror(status));
    sdiq->num_tx_errors++;
    sdiq_error(status);
  } else if (sdiq->meta_tx.status & BLADERF_META_STATUS_UNDERRUN){
    /* libbladeRF does not report this status. It is here for future use. */ 
    fprintf(stderr, "TX Underrun detected. %u valid samples were read.\n",  sdiq->meta_tx.actual_count);
    sdiq->num_underflows++;
  } 
  //printf("Provided TX timestampe  %u, meta timestame %u\n", ptimestamp,sdiq->meta_tx.timestamp);
  
  //    printf("tx status %d \n",sdiq->meta_tx.status);
  sdiq->tx_current_ts=sdiq->meta_tx.timestamp;
  sdiq->tx_actual_nsamps+=sdiq->meta_tx.actual_count;
  sdiq->tx_nsamps+=nsamps;
  sdiq->tx_count++;
  

  return(0);
}

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
static int trx_sdiq_read(openair0_device *device, openair0_timestamp *ptimestamp, void **buff, int nsamps, int cc) {

  int status=0;
  sdiq_state_t *sdiq = (sdiq_state_t*)device->priv;
  
  // SDIQ has only one rx/tx chain
  int16_t *samples = (int16_t*)buff[0];
  
  sdiq->meta_rx.flags = BLADERF_META_FLAG_RX_NOW;
  status = bladerf_sync_rx(sdiq->dev, samples, (unsigned int) nsamps, &sdiq->meta_rx, 2*sdiq->rx_timeout_ms);
  
  //  printf("Current RX timestampe  %u, nsamps %u, actual %u, cc %d\n",  sdiq->meta_rx.timestamp, nsamps, sdiq->meta_rx.actual_count, cc);
   
  if (status != 0) {
    fprintf(stderr, "RX failed: %s\n", bladerf_strerror(status)); 
    //    printf("RX failed: %s\n", bladerf_strerror(status)); 
    sdiq->num_rx_errors++;
  } else if ( sdiq->meta_rx.status & BLADERF_META_STATUS_OVERRUN) {
    sdiq->num_overflows++;
    printf("RX overrun (%d) is detected. t=" "%" PRIu64 "Got %u samples. nsymps %d\n", 
	   sdiq->num_overflows,sdiq->meta_rx.timestamp,  sdiq->meta_rx.actual_count, nsamps);
  } 

  //printf("Current RX timestampe  %u\n",  sdiq->meta_rx.timestamp);
  //printf("[SDIQ] (buff %p) ts=0x%"PRIu64" %s\n",samples, sdiq->meta_rx.timestamp,bladerf_strerror(status));
  sdiq->rx_current_ts=sdiq->meta_rx.timestamp;
  sdiq->rx_actual_nsamps+=sdiq->meta_rx.actual_count;
  sdiq->rx_nsamps+=nsamps;
  sdiq->rx_count++;
  
  
  *ptimestamp = sdiq->meta_rx.timestamp;
 
  return sdiq->meta_rx.actual_count;

}

/*! \brief Terminate operation of the Sidekiq transceiver -- free all associated resources 
 * \param device the hardware to use
 */
void trx_sdiq_end(openair0_device *device) {

  int status;
  sdiq_state_t *sdiq = (sdiq_state_t*)device->priv;
  // Disable RX module, shutting down our underlying RX stream
  if ((status=bladerf_enable_module(sdiq->dev, BLADERF_MODULE_RX, false))  != 0) {
    fprintf(stderr, "Failed to disable RX module: %s\n", bladerf_strerror(status));
  }
  if ((status=bladerf_enable_module(sdiq->dev, BLADERF_MODULE_TX, false))  != 0) {
    fprintf(stderr, "Failed to disable TX module: %s\n",  bladerf_strerror(status));
  }
  bladerf_close(sdiq->dev);
}

/*! \brief print the Sidekiq statistics  
* \param device the hardware to use
* \returns  0 on success
*/
int trx_sdiq_get_stats(openair0_device* device) {

  return(0);

}

/*! \brief Reset the Sidekiq statistics  
* \param device the hardware to use
* \returns  0 on success
*/
int trx_sdiq_reset_stats(openair0_device* device) {

  return(0);

}

/*! \brief Stop Sidekiq
 * \param card the hardware to use
 * \returns 0 in success 
 */
int trx_sdiq_stop(int card) {

  return(0);

}

/*! \brief Set frequencies (TX/RX)
 * \param device the hardware to use
 * \param openair0_cfg1 openair0 Config structure (ignored. It is there to comply with RF common API)
 * \param exmimo_dump_config (ignored)
 * \returns 0 in success 
 */
int trx_sdiq_set_freq(openair0_device* device, openair0_config_t *openair0_cfg1,int exmimo_dump_config) {

  int status;
  sdiq_state_t *sdiq = (sdiq_state_t *)device->priv;
  openair0_config_t *openair0_cfg = (openair0_config_t *)device->openair0_cfg;


  if ((status=bladerf_set_frequency(sdiq->dev, BLADERF_MODULE_TX, (unsigned int) openair0_cfg->tx_freq[0])) != 0){
    fprintf(stderr,"Failed to set TX frequency: %s\n",bladerf_strerror(status));
    sdiq_error(status);
  }else 
    printf("[SDIQ] set TX Frequency to %u\n", (unsigned int) openair0_cfg->tx_freq[0]);

  if ((status=bladerf_set_frequency(sdiq->dev, BLADERF_MODULE_RX, (unsigned int) openair0_cfg->rx_freq[0])) != 0){
    fprintf(stderr,"Failed to set RX frequency: %s\n",bladerf_strerror(status));
    sdiq_error(status);
  } else 
    printf("[SDIQ] set RX frequency to %u\n",(unsigned int)openair0_cfg->rx_freq[0]);

  return(0);

}

/*! \brief Set Gains (TX/RX)
 * \param device the hardware to use
 * \param openair0_cfg openair0 Config structure
 * \returns 0 in success 
 */
int trx_sdiq_set_gains(openair0_device* device, openair0_config_t *openair0_cfg) {

  return(0);

}



#define RXDCLENGTH 16384
int16_t cos_fsover8[8]  = {2047,   1447,      0,  -1448,  -2047,  -1448,     0,   1447};
int16_t cos_3fsover8[8] = {2047,  -1448,      0,   1447,  -2047,   1447,     0,  -1448};

/*! \brief calibration table for Sidekiq */
rx_gain_calib_table_t calib_table_sdiq[] = {
  {2300000000.0,53.5},
  {1880000000.0,57.0},
  {816000000.0,73.0},
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
  int status;
  sdiq_state_t *sdiq = (sdiq_state_t*)malloc(sizeof(sdiq_state_t));
  memset(sdiq, 0, sizeof(sdiq_state_t));
  /* device specific */
  //openair0_cfg->txlaunch_wait = 1;//manage when TX processing is triggered
  //openair0_cfg->txlaunch_wait_slotcount = 1; //manage when TX processing is triggered
  openair0_cfg->iq_txshift = 0;// shift
  openair0_cfg->iq_rxrescale = 15;//rescale iqs
  
  // init required params
  switch ((int)openair0_cfg->sample_rate) {
  case 30720000:
    openair0_cfg->samples_per_packet    = 2048;
    openair0_cfg->tx_sample_advance     = 0;
    break;
  case 15360000:
    openair0_cfg->samples_per_packet    = 2048;
    openair0_cfg->tx_sample_advance     = 0;
    break;
  case 7680000:
    openair0_cfg->samples_per_packet    = 1024;
    openair0_cfg->tx_sample_advance     = 0;
    break;
  case 1920000:
    openair0_cfg->samples_per_packet    = 256;
    openair0_cfg->tx_sample_advance     = 50;
    break;
  default:
    printf("Error: unknown sampling rate %f\n",openair0_cfg->sample_rate);
    exit(-1);
    break;
  }
  openair0_cfg->iq_txshift= 0;
  openair0_cfg->iq_rxrescale = 15; /*not sure*/
  openair0_cfg->rx_gain_calib_table = calib_table_sdiq;


  printf("\n[SDIQ] sampling_rate %d, num_buffers %d,  buffer_size %d, num transfer %d, timeout_ms (rx %d, tx %d)\n", 
	 sdiq->sample_rate, sdiq->num_buffers, sdiq->buffer_size,sdiq->num_transfers, sdiq->rx_timeout_ms, sdiq->tx_timeout_ms);

  // open device

  printf("[SDIQ] init dev %p\n", sdiq->dev);

  // RX  
    
  if ((status=sdiq_set_frequency(sdiq->dev, BLADERF_MODULE_RX, (unsigned int) openair0_cfg->rx_freq[0])) != 0){
    fprintf(stderr,"Failed to set RX frequency: %s\n",bladerf_strerror(status));
    sdiq_error(status);
  } else 
    printf("[SDIQ] set RX frequency to %u\n",(unsigned int)openair0_cfg->rx_freq[0]);
  


  unsigned int actual_value=0;
  if ((status=sdiq_set_sample_rate(sdiq->dev, SDIQ_MODULE_RX, (unsigned int) openair0_cfg->sample_rate, &actual_value)) != 0){
    fprintf(stderr,"Failed to set RX sample rate: %s\n", sdiq_strerror(status));
    sdiq_error(status);
  }else  
    printf("[SDIQ] set RX sample rate to %u, %u\n", (unsigned int) openair0_cfg->sample_rate, actual_value);
 

  if ((status=sdiq_set_bandwidth(sdiq->dev, SDIQ_MODULE_RX, (unsigned int) openair0_cfg->rx_bw*2, &actual_value)) != 0){
    fprintf(stderr,"Failed to set RX bandwidth: %s\n", sdiq_strerror(status));
    sdiq_error(status);
  }else 
    printf("[SDIQ] set RX bandwidth to %u, %u\n",(unsigned int)openair0_cfg->rx_bw*2, actual_value);
 
  set_rx_gain_offset(&openair0_cfg[0],0);
  if ((status=sdiq_set_gain(sdiq->dev, SDIQ_MODULE_RX, (int) openair0_cfg->rx_gain[0]-openair0_cfg[0].rx_gain_offset[0])) != 0) {
    fprintf(stderr,"Failed to set RX gain: %s\n",sdiq_strerror(status));
    sdiq_error(status);
  } else 
    printf("[SDIQ] set RX gain to %d (%d)\n",(int)(openair0_cfg->rx_gain[0]-openair0_cfg[0].rx_gain_offset[0]),(int)openair0_cfg[0].rx_gain_offset[0]);

  // TX
  
  if ((status=sdiq_set_frequency(sdiq->dev, SDIQ_MODULE_TX, (unsigned int) openair0_cfg->tx_freq[0])) != 0){
    fprintf(stderr,"Failed to set TX frequency: %s\n",sdiq_strerror(status));
    sdiq_error(status);
  }else 
    printf("[SDIQ] set TX Frequency to %u\n", (unsigned int) openair0_cfg->tx_freq[0]);

  if ((status=sdiq_set_sample_rate(sdiq->dev, SDIQ_MODULE_TX, (unsigned int) openair0_cfg->sample_rate, NULL)) != 0){
    fprintf(stderr,"Failed to set TX sample rate: %s\n", sdiq_strerror(status));
    sdiq_error(status);
  }else 
    printf("[SDIQ] set TX sampling rate to %u \n", (unsigned int) openair0_cfg->sample_rate);

  if ((status=sdiq_set_bandwidth(sdiq->dev, SDIQ_MODULE_TX,(unsigned int)openair0_cfg->tx_bw*2, NULL)) != 0){
    fprintf(stderr, "Failed to set TX bandwidth: %s\n", sdiq_strerror(status));
    sdiq_error(status);
  }else 
    printf("[SDIQ] set TX bandwidth to %u \n", (unsigned int) openair0_cfg->tx_bw*2);

  if ((status=sdiq_set_gain(sdiq->dev, SDIQ_MODULE_TX, (int) openair0_cfg->tx_gain[0])) != 0) {
    fprintf(stderr,"Failed to set TX gain: %s\n",sdiq_strerror(status));
    sdiq_error(status);
  }else 
    printf("[SDIQ] set the TX gain to %d\n", (int)openair0_cfg->tx_gain[0]);
  

 /* Configure the device's TX module for use with the sync interface.
   * SC16 Q11 samples *with* metadata are used. */
  if ((status=sdiq_sync_config(sdiq->dev, SDIQ_MODULE_TX,SDIQ_FORMAT_SC16_Q11_META,sdiq->num_buffers,sdiq->buffer_size,sdiq->num_transfers,sdiq->tx_timeout_ms)) != 0 ) {
    fprintf(stderr,"Failed to configure TX sync interface: %s\n", sdiq_strerror(status));
     sdiq_error(status);
  }else 
    printf("[SDIQ] configured TX  sync interface \n");

/* Configure the device's RX module for use with the sync interface.
   * SC16 Q11 samples *with* metadata are used. */
  if ((status=sdiq_sync_config(sdiq->dev, SDIQ_MODULE_RX, SDIQ_FORMAT_SC16_Q11_META,sdiq->num_buffers,sdiq->buffer_size,sdiq->num_transfers,sdiq->rx_timeout_ms)) != 0 ) {
    fprintf(stderr,"Failed to configure RX sync interface: %s\n", sdiq_strerror(status));
    sdiq_error(status);
  }else 
    printf("[SDIQ] configured Rx sync interface \n");


   /* We must always enable the TX module after calling sdiq_sync_config(), and 
    * before  attempting to TX samples via  sdiq_sync_tx(). */
  if ((status=sdiq_enable_module(sdiq->dev, SDIQ_MODULE_TX, true)) != 0) {
    fprintf(stderr,"Failed to enable TX module: %s\n", sdiq_strerror(status));
    sdiq_error(status);
  } else 
    printf("[SDIQ] TX module enabled \n");
 
 /* We must always enable the RX module after calling sdiq_sync_config(), and 
    * before  attempting to RX samples via  sdiq_sync_rx(). */
  if ((status=sdiq_enable_module(sdiq->dev, SDIQ_MODULE_RX, true)) != 0) {
    fprintf(stderr,"Failed to enable RX module: %s\n", sdiq_strerror(status));
    sdiq_error(status);
  }else 
    printf("[SDIQ] RX module enabled \n");

  // calibrate 
    
 if ((status=sdiq_calibrate_dc(sdiq->dev, SDIQ_MODULE_TX)) != 0) {
    fprintf(stderr,"Failed to calibrate TX DC: %s\n", sdiq_strerror(status));
    sdiq_error(status);
  } else 
    printf("[SDIQ] TX module calibrated DC \n");
 
  if ((status=sdiq_calibrate_dc(sdiq->dev, SDIQ_MODULE_RX)) != 0) {
    fprintf(stderr,"Failed to calibrate RX DC: %s\n", sdiq_strerror(status));
    sdiq_error(status);
  }else 
    printf("[SDIQ] RX module calibrated DC \n");
  
  sdiq_log_set_verbosity(get_sdiq_log_level(openair0_cfg->log_level));
  
  printf("SDIQ: Initializing openair0_device\n");
  device->Mod_id         = num_devices++;
  device->type             = SDIQ_DEV; 
  device->trx_start_func = trx_sdiq_start;
  device->trx_end_func   = trx_sdiq_end;
  device->trx_read_func  = trx_sdiq_read;
  device->trx_write_func = trx_sdiq_write;
  device->trx_get_stats_func   = trx_sdiq_get_stats;
  device->trx_reset_stats_func = trx_sdiq_reset_stats;
  device->trx_stop_func        = trx_sdiq_stop;
  device->trx_set_freq_func    = trx_sdiq_set_freq;
  device->trx_set_gains_func   = trx_sdiq_set_gains;
  device->openair0_cfg = openair0_cfg;
  device->priv = (void *)sdiq;


  //  memcpy((void*)&device->openair0_cfg,(void*)&openair0_cfg[0],sizeof(openair0_config_t));

  return 0;
}

/*! \brief sdiq error report 
 * \param status 
 * \returns 0 on success
 */
int sdiq_error(int status) {
  
  //exit(-1);
  return status; // or status error code
}



/*@}*/
