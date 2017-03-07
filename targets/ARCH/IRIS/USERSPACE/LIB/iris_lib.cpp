
/** iris_lib.cpp
 *
 * \author: Rahman Doost-Mohammady : doost@rice.edu
 */

#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <SoapySDR/Device.hpp>
#include <SoapySDR/Formats.hpp>
#include <SoapySDR/Time.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <complex>
#include <fstream>
#include <cmath>
#include <time.h>
#include <limits>
#include "UTIL/LOG/log_extern.h"
#include "common_lib.h"


/*! \brief Iris Configuration */
typedef struct
{

  // --------------------------------
  // variables for Iris configuration
  // --------------------------------
  //! Iris device pointer
  SoapySDR::Device *iris;

  //create a send streamer and a receive streamer
  //! Iris TX Stream
  SoapySDR::Stream *txStream;
  //! Iris RX Stream
  SoapySDR::Stream *rxStream;

  //! Sampling rate
  double sample_rate;

  //! time offset between transmiter timestamp and receiver timestamp;
  double tdiff;

  //! TX forward samples.
  int tx_forward_nsamps; //166 for 20Mhz


  // --------------------------------
  // Debug and output control
  // --------------------------------
  //! Number of underflows
  int num_underflows;
  //! Number of overflows
  int num_overflows;

  //! Number of sequential errors
  int num_seq_errors;
  //! tx count
  int64_t tx_count;
  //! rx count
  int64_t rx_count;
  //! timestamp of RX packet
  openair0_timestamp rx_timestamp;

} iris_state_t;

/*! \brief Called to start the Iris lime transceiver. Return 0 if OK, < 0 if error
    @param device pointer to the device structure specific to the RF hardware target
*/
static int trx_iris_start(openair0_device *device)
{
	iris_state_t *s = (iris_state_t*)device->priv;
	long long timeNs = s->iris->getHardwareTime("") + 500000;
	int flags = 0;
	flags |= SOAPY_SDR_HAS_TIME;
	int ret = s->iris->activateStream(s->rxStream, flags, timeNs, 0);
	int ret2 = s->iris->activateStream(s->txStream);
	if (ret < 0 | ret2 < 0)
		return - 1;
	return 0;
}
/*! \brief Terminate operation of the Iris lime transceiver -- free all associated resources 
 * \param device the hardware to use
 */
static void trx_iris_end(openair0_device *device)
{
	LOG_I(HW,"Closing Iris device.\n");
	iris_state_t *s = (iris_state_t*)device->priv;
	s->iris->closeStream(s->txStream);
	s->iris->closeStream(s->rxStream);
	SoapySDR::Device::unmake(s->iris);
}

/*! \brief Called to send samples to the Iris RF target
      @param device pointer to the device structure specific to the RF hardware target
      @param timestamp The timestamp at whicch the first sample MUST be sent 
      @param buff Buffer which holds the samples
      @param nsamps number of samples to be sent
      @param antenna_id index of the antenna if the device has multiple anteannas
      @param flags flags must be set to TRUE if timestamp parameter needs to be applied
*/ 
static int trx_iris_write(openair0_device *device, openair0_timestamp timestamp, void **buff, int nsamps, int cc, int flags)
{
	static long long int loop=0;
	static long time_min=0, time_max=0, time_avg=0;
	struct timespec tp_start, tp_end;
	long time_diff;

	int ret=0, ret_i=0;
	int flag = 0;
	iris_state_t *s = (iris_state_t*)device->priv;

	clock_gettime(CLOCK_MONOTONIC_RAW, &tp_start);
	if (flags)
		flag |= SOAPY_SDR_HAS_TIME;

	long long timeNs = SoapySDR::ticksToTimeNs(timestamp, s->sample_rate);
	int samples_sent = 0;
	uint32_t **samps = (uint32_t **)buff;
	while (samples_sent < nsamps)
	{
		ret = s->iris->writeStream(s->txStream, (void **)samps, (size_t)(nsamps - samples_sent), flag, timeNs, 100000);
		if (ret < 0) {
			printf("Unable to write stream!\n");
			break;
		}
		samples_sent += ret;
		samps[0] += ret;
		if (cc > 1)
			samps[1] += ret;
	}

	if (samples_sent != nsamps) {
		printf("[xmit] tx samples %d != %d\n",samples_sent,nsamps);
	}
	/*
	flag = 0;
	size_t channel = 0;
	ret = s->iris->readStreamStatus(s->txStream, channel, flag, timeNs, 0);
	if (ret == SOAPY_SDR_TIME_ERROR)
		printf("[xmit] Time Error in tx stream!\n");
	else if (ret == SOAPY_SDR_UNDERFLOW)
		printf("[xmit] Underflow occured!\n");
	else if (ret == SOAPY_SDR_TIMEOUT)
		printf("[xmit] Timeout occured!\n");
	else if (ret == SOAPY_SDR_STREAM_ERROR)
		printf("[xmit] Stream (tx) error occured!\n");
	*/
	return nsamps;
}

/*! \brief Receive samples from hardware.
 * Read \ref nsamps samples from each channel to buffers. buff[0] is the array for
 * the first channel. *ptimestamp is the time at which the first sample
 * was received.
 * \param device the hardware to use
 * \param[out] ptimestamp the time at which the first sample was received.
 * \param[out] buff An array of pointers to buffers for received samples. The buffers must be large enough to hold the number of samples \ref nsamps.
 * \param nsamps Number of samples. One sample is 2 byte I + 2 byte Q => 4 byte.
 * \param antenna_id Index of antenna for which to receive samples
 * \returns the number of sample read
*/
static int trx_iris_read(openair0_device *device, openair0_timestamp *ptimestamp, void **buff, int nsamps, int cc)
{
	int ret = 0;
	static long long nextTime;
	static bool nextTimeValid = false;
	iris_state_t *s = (iris_state_t*)device->priv;
	bool time_set = false;
	long long timeNs = 0;
	int flags = 0;
	int samples_received = 0;
	uint32_t *samps[2] = {(uint32_t *)buff[0], (uint32_t *)buff[1]}; //cws: it seems another thread can clobber these, so we need to save them locally.
	//printf("Reading %d samples from Iris...\n", nsamps);
	//fflush(stdout);
	while (samples_received < nsamps)
	{
		flags = 0;
		ret = s->iris->readStream(s->rxStream, (void **)samps, (size_t)(nsamps-samples_received), flags, timeNs, 100000);
		if (ret < 0)
		{
			if (ret == SOAPY_SDR_TIME_ERROR)
				printf("[recv] Time Error in tx stream!\n");
			else if (ret == SOAPY_SDR_OVERFLOW | (flags & SOAPY_SDR_END_ABRUPT))
				printf("[recv] Overflow occured!\n");
			else if (ret == SOAPY_SDR_TIMEOUT)
				printf("[recv] Timeout occured!\n");
			else if (ret == SOAPY_SDR_STREAM_ERROR)
				printf("[recv] Stream (tx) error occured!\n");
			else if (ret == SOAPY_SDR_CORRUPTION)
				printf("[recv] Bad packet occured!\n");
			break;
		}

		samples_received += ret;
		samps[0] += ret;
		if (cc > 1)
			samps[1] += ret;

		if (samples_received == ret) // first batch
		{
			if (flags & SOAPY_SDR_HAS_TIME)
			{
				s->rx_timestamp = SoapySDR::timeNsToTicks(timeNs, s->sample_rate);
				*ptimestamp = s->rx_timestamp;
				nextTime = timeNs;
				nextTimeValid = true;
				time_set = true;
				//printf("1) time set %llu \n", *ptimestamp);
			}
		}
	}

	if (samples_received < nsamps)
		printf("[recv] received %d samples out of %d\n",samples_received,nsamps);

	s->rx_count += samples_received;

	if (s->sample_rate != 0 && nextTimeValid)
	{
		if (!time_set)
		{
			s->rx_timestamp = SoapySDR::timeNsToTicks(nextTime, s->sample_rate);
			*ptimestamp = s->rx_timestamp;
			//printf("2) time set %llu, nextTime will be %llu \n", *ptimestamp, nextTime);
		}
		nextTime += SoapySDR::ticksToTimeNs(samples_received, s->sample_rate);
	}

	return samples_received;
}

/*! \brief Get current timestamp of Iris
 * \param device the hardware to use
*/
openair0_timestamp get_iris_time(openair0_device *device)
{
	iris_state_t *s = (iris_state_t*)device->priv;
	return SoapySDR::timeNsToTicks(s->iris->getHardwareTime(""), s->sample_rate);
}

/*! \brief Compares two variables within precision
 * \param a first variable
 * \param b second variable
*/
static bool is_equal(double a, double b)
{
	return std::fabs(a-b) < std::numeric_limits<double>::epsilon();
}

void *set_freq_thread(void *arg) {

    openair0_device *device=(openair0_device *)arg;
    iris_state_t *s = (iris_state_t*)device->priv;
    int i;
    printf("Setting Iris TX Freq %f, RX Freq %f\n",device->openair0_cfg[0].tx_freq[0],device->openair0_cfg[0].rx_freq[0]);
    // add check for the number of channels in the cfg
    for(i=0; i < s->iris->getNumChannels(SOAPY_SDR_RX); i++) {
	    if (i < device->openair0_cfg[0].rx_num_channels)
            s->iris->setFrequency(SOAPY_SDR_RX, i, "RF", device->openair0_cfg[0].rx_freq[i]);
    }
    for(i=0; i < s->iris->getNumChannels(SOAPY_SDR_TX); i++) {
	    if (i < device->openair0_cfg[0].tx_num_channels)
            s->iris->setFrequency(SOAPY_SDR_TX, i, "RF", device->openair0_cfg[0].tx_freq[i]);
    }
}
/*! \brief Set frequencies (TX/RX)
 * \param device the hardware to use
 * \param openair0_cfg RF frontend parameters set by application
 * \param dummy dummy variable not used
 * \returns 0 in success
 */
int trx_iris_set_freq(openair0_device* device, openair0_config_t *openair0_cfg, int dont_block)
{
    iris_state_t *s = (iris_state_t*)device->priv;
    pthread_t f_thread;
    if (dont_block)
        pthread_create(&f_thread, NULL, set_freq_thread, (void*)device);
    else
    {
        int i;
        printf("Setting Iris TX Freq %f, RX Freq %f\n",openair0_cfg[0].tx_freq[0],openair0_cfg[0].rx_freq[0]);
    	for(i=0; i < s->iris->getNumChannels(SOAPY_SDR_RX); i++) {
	    	if (i < openair0_cfg[0].rx_num_channels) {
                s->iris->setFrequency(SOAPY_SDR_RX, i, "RF", openair0_cfg[0].rx_freq[i]);
            }
        }
    	for(i=0; i < s->iris->getNumChannels(SOAPY_SDR_TX); i++) {
	    	if (i < openair0_cfg[0].tx_num_channels) {
                s->iris->setFrequency(SOAPY_SDR_TX, i, "RF", openair0_cfg[0].tx_freq[i]);
            }
        }
	}
    return(0);
}


/*! \brief Set Gains (TX/RX)
 * \param device the hardware to use
 * \param openair0_cfg RF frontend parameters set by application
 * \returns 0 in success
 */
int trx_iris_set_gains(openair0_device* device,
		       openair0_config_t *openair0_cfg) {
	iris_state_t *s = (iris_state_t*)device->priv;
	s->iris->setGain(SOAPY_SDR_RX, 0, openair0_cfg[0].rx_gain[0]);
	s->iris->setGain(SOAPY_SDR_TX, 0, openair0_cfg[0].tx_gain[0]);
	s->iris->setGain(SOAPY_SDR_RX, 1, openair0_cfg[0].rx_gain[1]);
	s->iris->setGain(SOAPY_SDR_TX, 1, openair0_cfg[0].tx_gain[1]);
	return(0);
}

/*! \brief Stop Iris
 * \param card refers to the hardware index to use
 */
int trx_iris_stop(openair0_device* device) {
	iris_state_t *s = (iris_state_t*)device->priv;
	s->iris->deactivateStream(s->txStream);
	s->iris->deactivateStream(s->rxStream);
	return(0);
}

/*! \brief Iris RX calibration table */
rx_gain_calib_table_t calib_table_iris[] = {
  {3500000000.0,0},
  {2660000000.0,12},
  {2300000000.0,0},
  {1880000000.0,0},
  {816000000.0,0},
  {-1,0}};


/*! \brief Set RX gain offset
 * \param openair0_cfg RF frontend parameters set by application
 * \param chain_index RF chain to apply settings to
 * \returns 0 in success
 */
void set_rx_gain_offset(openair0_config_t *openair0_cfg, int chain_index,int bw_gain_adjust) {

  int i=0;
  // loop through calibration table to find best adjustment factor for RX frequency
  double min_diff = 6e9,diff,gain_adj=0.0;
  if (bw_gain_adjust==1) {
    switch ((int)openair0_cfg[0].sample_rate) {
    case 30720000:
      break;
    case 23040000:
      gain_adj=1.25;
      break;
    case 15360000:
      gain_adj=3.0;
      break;
    case 7680000:
      gain_adj=6.0;
      break;
    case 3840000:
      gain_adj=9.0;
      break;
    case 1920000:
      gain_adj=12.0;
      break;
    default:
      printf("unknown sampling rate %d\n",(int)openair0_cfg[0].sample_rate);
      exit(-1);
      break;
    }
  }
  while (openair0_cfg->rx_gain_calib_table[i].freq>0) {
    diff = fabs(openair0_cfg->rx_freq[chain_index] - openair0_cfg->rx_gain_calib_table[i].freq);
    printf("cal %d: freq %f, offset %f, diff %f\n",
	   i,
	   openair0_cfg->rx_gain_calib_table[i].freq,
	   openair0_cfg->rx_gain_calib_table[i].offset,diff);
    if (min_diff > diff) {
      min_diff = diff;
      openair0_cfg->rx_gain_offset[chain_index] = openair0_cfg->rx_gain_calib_table[i].offset+gain_adj;
    }
    i++;
  }

}

/*! \brief print the Iris statistics
* \param device the hardware to use
* \returns  0 on success
*/
int trx_iris_get_stats(openair0_device* device) {

  return(0);

}

/*! \brief Reset the Iris statistics
* \param device the hardware to use
* \returns  0 on success
*/
int trx_iris_reset_stats(openair0_device* device) {

  return(0);

}



extern "C" {
/*! \brief Initialize Openair Iris target. It returns 0 if OK
* \param device the hardware to use
* \param openair0_cfg RF frontend parameters set by application
*/
  int device_init(openair0_device* device, openair0_config_t *openair0_cfg) {

	size_t i;
	int bw_gain_adjust=0;
	openair0_cfg[0].rx_gain_calib_table = calib_table_iris;
	iris_state_t *s = (iris_state_t*)malloc(sizeof(iris_state_t));
	memset(s, 0, sizeof(iris_state_t));

	// Initialize Iris device
	device->openair0_cfg = openair0_cfg;
	char* remote_addr = device->openair0_cfg->remote_addr;
	LOG_I(HW,"Attempting to open Iris device: %s\n", remote_addr);
	std::string args = "driver=remote,serial="+std::string(remote_addr);

	s->iris = SoapySDR::Device::make(args);
	device->type=IRIS_DEV;

	s->iris->setMasterClockRate(8*openair0_cfg[0].sample_rate); // sample*8=clock_rate for Soapy
	printf("tx_sample_advance %d\n", openair0_cfg[0].tx_sample_advance);
	switch ((int)openair0_cfg[0].sample_rate) {
	case 30720000:
		//openair0_cfg[0].samples_per_packet    = 1024;
		//openair0_cfg[0].tx_sample_advance     = 115;
		openair0_cfg[0].tx_bw                 = 30e6;
		openair0_cfg[0].rx_bw                 = 30e6;
		break;
	case 23040000:
		//openair0_cfg[0].samples_per_packet    = 1024;
		//openair0_cfg[0].tx_sample_advance     = 113;
		openair0_cfg[0].tx_bw                 = 30e6;
		openair0_cfg[0].rx_bw                 = 30e6;
		break;
	case 15360000:
		//openair0_cfg[0].samples_per_packet    = 1024;
		//openair0_cfg[0].tx_sample_advance     = 103;
		openair0_cfg[0].tx_bw                 = 30e6;
		openair0_cfg[0].rx_bw                 = 30e6;
		break;
	case 7680000:
		//openair0_cfg[0].samples_per_packet    = 1024;
		//openair0_cfg[0].tx_sample_advance     = 80;
		openair0_cfg[0].tx_bw                 = 30e6;
		openair0_cfg[0].rx_bw                 = 30e6;
		break;
	case 1920000:
		//openair0_cfg[0].samples_per_packet    = 1024;
		//openair0_cfg[0].tx_sample_advance     = 40;
		openair0_cfg[0].tx_bw                 = 30e6;
		openair0_cfg[0].rx_bw                 = 30e6;
		break;
	default:
		printf("Error: unknown sampling rate %f\n",openair0_cfg[0].sample_rate);
		exit(-1);
		break;
	}


	for(i=0; i < s->iris->getNumChannels(SOAPY_SDR_RX); i++) {
		if (i < openair0_cfg[0].rx_num_channels) {
			s->iris->setSampleRate(SOAPY_SDR_RX, i, openair0_cfg[0].sample_rate);
			s->iris->setFrequency(SOAPY_SDR_RX, i, "RF", openair0_cfg[0].rx_freq[i]);
			set_rx_gain_offset(&openair0_cfg[0],i,bw_gain_adjust);

			s->iris->setGain(SOAPY_SDR_RX, i, openair0_cfg[0].rx_gain[i]-openair0_cfg[0].rx_gain_offset[i]);
			if (openair0_cfg[0].duplex_mode == 1) // TDD
				s->iris->setAntenna(SOAPY_SDR_RX, i, (i==0)?"TRXA":"TRXB");			
			s->iris->setDCOffsetMode(SOAPY_SDR_RX, i, true); // move somewhere else
		}
	}
	for(i=0; i < s->iris->getNumChannels(SOAPY_SDR_TX); i++) {
		if (i < openair0_cfg[0].tx_num_channels) {
			s->iris->setSampleRate(SOAPY_SDR_TX, i, openair0_cfg[0].sample_rate);
			s->iris->setFrequency(SOAPY_SDR_TX, i, "RF", openair0_cfg[0].tx_freq[i]);
			s->iris->setGain(SOAPY_SDR_TX, i, openair0_cfg[0].tx_gain[i]);
		}
	}


	// display Iris settings
	std::cout << boost::format("Actual master clock: %fMHz...") % (s->iris->getMasterClockRate()/1e6) << std::endl;

	sleep(1);
	int samples=openair0_cfg[0].sample_rate;
	samples/=24000;

	// create tx & rx streamer
	const SoapySDR::Kwargs &arg = SoapySDR::Kwargs();
	std::vector<size_t> channels={};
	for (i = 0; i<openair0_cfg[0].rx_num_channels; i++)
		if (i < s->iris->getNumChannels(SOAPY_SDR_RX))
			channels.push_back(i);
	s->rxStream = s->iris->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, channels, arg);

	std::vector<size_t> tx_channels={};
	for (i = 0; i<openair0_cfg[0].tx_num_channels; i++)
		if (i < s->iris->getNumChannels(SOAPY_SDR_TX))
			tx_channels.push_back(i);
	s->txStream = s->iris->setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16, tx_channels, arg);

	/* Setting TX/RX BW after streamers are created due to iris calibration issue */
	for(i = 0; i < openair0_cfg[0].tx_num_channels; i++) {
		if (i < s->iris->getNumChannels(SOAPY_SDR_TX) ) {
			s->iris->setBandwidth(SOAPY_SDR_TX, i, openair0_cfg[0].tx_bw);
			printf("Setting tx freq/gain on channel %lu/%lu: BW %f (readback %f)\n",i,s->iris->getNumChannels(SOAPY_SDR_TX),openair0_cfg[0].tx_bw/1e6,s->iris->getBandwidth(SOAPY_SDR_TX, i)/1e6);
		}
	}
	for(i = 0; i < openair0_cfg[0].rx_num_channels; i++) {
		if (i < s->iris->getNumChannels(SOAPY_SDR_RX)) {
			s->iris->setBandwidth(SOAPY_SDR_RX, i, openair0_cfg[0].rx_bw);
			printf("Setting rx freq/gain on channel %lu/%lu : BW %f (readback %f)\n",i,s->iris->getNumChannels(SOAPY_SDR_RX),openair0_cfg[0].rx_bw/1e6,s->iris->getBandwidth(SOAPY_SDR_RX, i)/1e6);
		}
	}

	s->iris->setHardwareTime(0, "");


	for (i = 0; i < openair0_cfg[0].rx_num_channels; i++) {
		if (i < s->iris->getNumChannels(SOAPY_SDR_RX)) {
			printf("RX Channel %lu\n",i);
			std::cout << boost::format("Actual RX sample rate: %fMSps...") % (s->iris->getSampleRate(SOAPY_SDR_RX, i)/1e6) << std::endl;
			std::cout << boost::format("Actual RX frequency: %fGHz...") % (s->iris->getFrequency(SOAPY_SDR_RX, i)/1e9) << std::endl;
			std::cout << boost::format("Actual RX gain: %f...") % (s->iris->getGain(SOAPY_SDR_RX, i)) << std::endl;
			std::cout << boost::format("Actual RX bandwidth: %fM...") % (s->iris->getBandwidth(SOAPY_SDR_RX, i)/1e6) << std::endl;
			std::cout << boost::format("Actual RX antenna: %s...") % (s->iris->getAntenna(SOAPY_SDR_RX, i)) << std::endl;
		}
	}

	for (i=0;i<openair0_cfg[0].tx_num_channels;i++) {
		if (i < s->iris->getNumChannels(SOAPY_SDR_TX)) {
			printf("TX Channel %lu\n",i);
			std::cout << std::endl<<boost::format("Actual TX sample rate: %fMSps...") % (s->iris->getSampleRate(SOAPY_SDR_TX, i)/1e6) << std::endl;
			std::cout << boost::format("Actual TX frequency: %fGHz...") % (s->iris->getFrequency(SOAPY_SDR_TX, i)/1e9) << std::endl;
			std::cout << boost::format("Actual TX gain: %f...") % (s->iris->getGain(SOAPY_SDR_TX, i)) << std::endl;
			std::cout << boost::format("Actual TX bandwidth: %fM...") % (s->iris->getBandwidth(SOAPY_SDR_TX, i)/1e6) << std::endl;
			std::cout << boost::format("Actual TX antenna: %s...") % (s->iris->getAntenna(SOAPY_SDR_TX, i)) << std::endl;
		}
	}

	std::cout << boost::format("Device timestamp: %f...") % (s->iris->getHardwareTime()/1e9) << std::endl;

	device->priv = s;
	device->trx_start_func = trx_iris_start;
	device->trx_write_func = trx_iris_write;
	device->trx_read_func  = trx_iris_read;
	device->trx_get_stats_func = trx_iris_get_stats;
	device->trx_reset_stats_func = trx_iris_reset_stats;
	device->trx_end_func   = trx_iris_end;
	device->trx_stop_func  = trx_iris_stop;
	device->trx_set_freq_func = trx_iris_set_freq;
	device->trx_set_gains_func   = trx_iris_set_gains;
	device->openair0_cfg = openair0_cfg;

	s->sample_rate = openair0_cfg[0].sample_rate;
	// TODO:
	// init tx_forward_nsamps based iris_time_offset ex
	if(is_equal(s->sample_rate, (double)30.72e6))
		s->tx_forward_nsamps  = 176;
	if(is_equal(s->sample_rate, (double)15.36e6))
		s->tx_forward_nsamps = 90;
	if(is_equal(s->sample_rate, (double)7.68e6))
		s->tx_forward_nsamps = 50;

	LOG_I(HW,"Finished initializing Iris device. %d %f \n");
	return 0;
  }
}
/*@}*/
