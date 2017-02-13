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

/** openair0_lib : API to interface with ExpressMIMO-1&2 kernel driver
*
*  Authors: Matthias Ihmig <matthias.ihmig@mytum.de>, 2013
*           Raymond Knopp <raymond.knopp@eurecom.fr>
*
*  Changelog:
*  28.01.2013: Initial version
*/

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "openair0_lib.h"
#include "openair_device.h"
#include "common_lib.h"

#include "syr_pio.h"

#define max(a,b) ((a)>(b) ? (a) : (b))

exmimo_pci_interface_bot_virtual_t openair0_exmimo_pci[MAX_CARDS]; // contains userspace pointers for each card

char *bigshm_top[MAX_CARDS] = INIT_ZEROS;

int openair0_fd;
int openair0_num_antennas[MAX_CARDS];
int openair0_num_detected_cards = 0;

unsigned int PAGE_SHIFT;

static uint32_t                      rf_local[4]		= {8255000,8255000,8255000,8255000}; // UE zepto
//{8254617, 8254617, 8254617, 8254617}; //eNB khalifa
//{8255067,8254810,8257340,8257340}; // eNB PETRONAS

static uint32_t                      rf_vcocal[4]		= {910,910,910,910};
static uint32_t                      rf_vcocal_850[4]	= {2015, 2015, 2015, 2015};
static uint32_t                      rf_rxdc[4]			= {32896,32896,32896,32896};

uint32_t			leon3_spiwrite_intr_cnt				= 0;
uint32_t			leon3_call_exec_cnt					= 0;
uint32_t			leon3_intr_exec_cnt					= 0;

unsigned int log2_int( unsigned int x )
{
	unsigned int ans = 0 ;

	while( x>>=1 ) ans++;

	return ans ;
}

int openair0_open(void)
{
	exmimo_pci_interface_bot_virtual_t exmimo_pci_kvirt[MAX_CARDS];
	void *bigshm_top_kvirtptr[MAX_CARDS];

	int card;
	int ant;

	PAGE_SHIFT = log2_int( sysconf( _SC_PAGESIZE ) );

	if ((openair0_fd = open("/dev/openair0", O_RDWR,0)) <0) {
	return -1;
	}

	ioctl(openair0_fd, openair_GET_NUM_DETECTED_CARDS, &openair0_num_detected_cards);

	if ( openair0_num_detected_cards == 0 ) {
	fprintf(stderr, "No cards detected!\n");
	return -4;
	}

	ioctl(openair0_fd, openair_GET_BIGSHMTOPS_KVIRT, &bigshm_top_kvirtptr[0]);
	ioctl(openair0_fd, openair_GET_PCI_INTERFACE_BOTS_KVIRT, &exmimo_pci_kvirt[0]);

	//printf("bigshm_top_kvirtptr (MAX_CARDS %d): %p  %p  %p  %p\n", MAX_CARDS,bigshm_top_kvirtptr[0], bigshm_top_kvirtptr[1], bigshm_top_kvirtptr[2], bigshm_top_kvirtptr[3]);

	for( card=0; card < openair0_num_detected_cards; card++) {
	bigshm_top[card] = (char *)mmap( NULL,
	BIGSHM_SIZE_PAGES<<PAGE_SHIFT,
	PROT_READ|PROT_WRITE,
	MAP_SHARED, //|MAP_FIXED,//MAP_SHARED,
	openair0_fd,
	( openair_mmap_BIGSHM | openair_mmap_Card(card) )<<PAGE_SHIFT);

	if (bigshm_top[card] == MAP_FAILED) {
	openair0_close();
	return -2;
	}

	// calculate userspace addresses
	#if __x86_64
	openair0_exmimo_pci[card].firmware_block_ptr = (bigshm_top[card] +  (int64_t)exmimo_pci_kvirt[0].firmware_block_ptr - (int64_t)bigshm_top_kvirtptr[0]);
	openair0_exmimo_pci[card].printk_buffer_ptr  = (bigshm_top[card] +  (int64_t)exmimo_pci_kvirt[0].printk_buffer_ptr  - (int64_t)bigshm_top_kvirtptr[0]);
	openair0_exmimo_pci[card].exmimo_config_ptr  = (exmimo_config_t*) (bigshm_top[card] +  (int64_t)exmimo_pci_kvirt[0].exmimo_config_ptr  - (int64_t)bigshm_top_kvirtptr[0]);
	openair0_exmimo_pci[card].exmimo_id_ptr      = (exmimo_id_t*)     (bigshm_top[card] +  (int64_t)exmimo_pci_kvirt[0].exmimo_id_ptr      - (int64_t)bigshm_top_kvirtptr[0]);
	#else
	openair0_exmimo_pci[card].firmware_block_ptr = (bigshm_top[card] +  (int32_t)exmimo_pci_kvirt[0].firmware_block_ptr - (int32_t)bigshm_top_kvirtptr[0]);
	openair0_exmimo_pci[card].printk_buffer_ptr  = (bigshm_top[card] +  (int32_t)exmimo_pci_kvirt[0].printk_buffer_ptr  - (int32_t)bigshm_top_kvirtptr[0]);
	openair0_exmimo_pci[card].exmimo_config_ptr  = (exmimo_config_t*) (bigshm_top[card] +  (int32_t)exmimo_pci_kvirt[0].exmimo_config_ptr  - (int32_t)bigshm_top_kvirtptr[0]);
	openair0_exmimo_pci[card].exmimo_id_ptr      = (exmimo_id_t*)     (bigshm_top[card] +  (int32_t)exmimo_pci_kvirt[0].exmimo_id_ptr      - (int32_t)bigshm_top_kvirtptr[0]);
	#endif

	/*
	printf("openair0_exmimo_pci.firmware_block_ptr (%p) =  bigshm_top(%p) + exmimo_pci_kvirt.firmware_block_ptr(%p) - bigshm_top_kvirtptr(%p)\n",
	openair0_exmimo_pci[card].firmware_block_ptr, bigshm_top, exmimo_pci_kvirt[card].firmware_block_ptr, bigshm_top_kvirtptr[card]);
	printf("card%d, openair0_exmimo_pci.exmimo_id_ptr      (%p) =  bigshm_top(%p) + exmimo_pci_kvirt.exmimo_id_ptr     (%p) - bigshm_top_kvirtptr(%p)\n",
	card, openair0_exmimo_pci[card].exmimo_id_ptr, bigshm_top[card], exmimo_pci_kvirt[card].exmimo_id_ptr, bigshm_top_kvirtptr[card]);
	*/

	/*
	if (openair0_exmimo_pci[card].exmimo_id_ptr->board_swrev != BOARD_SWREV_CNTL2)
	{
	error("Software revision %d and firmware revision %d do not match, Please update either Software or Firmware",BOARD_SWREV_CNTL2,openair0_exmimo_pci[card].exmimo_id_ptr->board_swrev);
	return -5;
	}
	*/

	if ( openair0_exmimo_pci[card].exmimo_id_ptr->board_exmimoversion == 1)
	openair0_num_antennas[card] = 2;

	if ( openair0_exmimo_pci[card].exmimo_id_ptr->board_exmimoversion == 2)
	openair0_num_antennas[card] = 4;

	for (ant=0; ant<openair0_num_antennas[card]; ant++) {
	#if __x86_64__
	openair0_exmimo_pci[card].rxcnt_ptr[ant] = (unsigned int *) (bigshm_top[card] +  (int64_t)exmimo_pci_kvirt[card].rxcnt_ptr[ant] - (int64_t)bigshm_top_kvirtptr[card]);
	openair0_exmimo_pci[card].txcnt_ptr[ant] = (unsigned int *) (bigshm_top[card] +  (int64_t)exmimo_pci_kvirt[card].txcnt_ptr[ant] - (int64_t)bigshm_top_kvirtptr[card]);
	#else
	openair0_exmimo_pci[card].rxcnt_ptr[ant] = (unsigned int *) (bigshm_top[card] +  (int32_t)exmimo_pci_kvirt[card].rxcnt_ptr[ant] - (int32_t)bigshm_top_kvirtptr[card]);
	openair0_exmimo_pci[card].txcnt_ptr[ant] = (unsigned int *) (bigshm_top[card] +  (int32_t)exmimo_pci_kvirt[card].txcnt_ptr[ant] - (int32_t)bigshm_top_kvirtptr[card]);
	#endif
	}

	for (ant=0; ant<openair0_num_antennas[card]; ant++) {
	openair0_exmimo_pci[card].adc_head[ant] = mmap( NULL,
	ADAC_BUFFERSZ_PERCHAN_B,
	PROT_READ|PROT_WRITE,
	MAP_SHARED, //|MAP_FIXED,//MAP_SHARED,
	openair0_fd,
	( openair_mmap_RX(ant) | openair_mmap_Card(card) )<<PAGE_SHIFT );

	openair0_exmimo_pci[card].dac_head[ant] = mmap( NULL,
	ADAC_BUFFERSZ_PERCHAN_B,
	PROT_READ|PROT_WRITE,
	MAP_SHARED, //|MAP_FIXED,//MAP_SHARED,
	openair0_fd,
	( openair_mmap_TX(ant) | openair_mmap_Card(card) )<<PAGE_SHIFT );

	if (openair0_exmimo_pci[card].adc_head[ant] == MAP_FAILED || openair0_exmimo_pci[card].dac_head[ant] == MAP_FAILED) {
	openair0_close();
	return -3;
	}
	}

	//printf("p_exmimo_config = %p, p_exmimo_id = %p\n", openair0_exmimo_pci.exmimo_config_ptr, openair0_exmimo_pci.exmimo_id_ptr);

	printf("card %d: ExpressMIMO %d, HW Rev %d, SW Rev 0x%d, %d antennas\n", card, openair0_exmimo_pci[card].exmimo_id_ptr->board_exmimoversion,
	openair0_exmimo_pci[card].exmimo_id_ptr->board_hwrev, openair0_exmimo_pci[card].exmimo_id_ptr->board_swrev, openair0_num_antennas[card]);

	} // end for(card)

	return 0;
}


int openair0_close(void)
{
	int ant;
	int card;

	close(openair0_fd);

	for (card=0; card<openair0_num_detected_cards; card++) {
	if (bigshm_top[card] != NULL && bigshm_top[card] != MAP_FAILED)
	munmap(bigshm_top[card], BIGSHM_SIZE_PAGES<<PAGE_SHIFT);

	for (ant=0; ant<openair0_num_antennas[card]; ant++) {
	if (openair0_exmimo_pci[card].adc_head[ant] != NULL && openair0_exmimo_pci[card].adc_head[ant] != MAP_FAILED)
	munmap(openair0_exmimo_pci[card].adc_head[ant], ADAC_BUFFERSZ_PERCHAN_B);

	if (openair0_exmimo_pci[card].dac_head[ant] != NULL && openair0_exmimo_pci[card].dac_head[ant] != MAP_FAILED)
	munmap(openair0_exmimo_pci[card].dac_head[ant], ADAC_BUFFERSZ_PERCHAN_B);
	}
	}

	return 0;
}

int openair0_dump_config(int card)
{
	return ioctl(openair0_fd, openair_DUMP_CONFIG, card);
}

int openair0_get_frame(int card)
{
	return ioctl(openair0_fd, openair_GET_FRAME, card);
}

int openair0_start_rt_acquisition(int card)
{
	return ioctl(openair0_fd, openair_START_RT_ACQUISITION, card);
}

int openair0_stop(int card)
{
	return ioctl(openair0_fd, openair_STOP, card);
}

int openair0_stop_without_reset(int card)
{
	return ioctl(openair0_fd, openair_STOP_WITHOUT_RESET, card);
}

#define MY_RF_MODE      (RXEN + TXEN + TXLPFNORM + TXLPFEN + TXLPF25 + RXLPFNORM + RXLPFEN + RXLPF25 + LNA1ON + LNAMax + RFBBNORM + DMAMODE_RX + DMAMODE_TX)
#define RF_MODE_BASE    (LNA1ON + RFBBNORM)

int openair0_dev_init_exmimo(openair0_device *device, openair0_config_t *openair0_cfg)
{
	// Initialize card
	//  exmimo_config_t         *p_exmimo_config;
	exmimo_id_t             *p_exmimo_id;
	int ret;

	ret = openair0_open();


	if ( ret != 0 ) {
	if (ret == -1)
	printf("Error opening /dev/openair0");

	if (ret == -2)
	printf("Error mapping bigshm");

	if (ret == -3)
	printf("Error mapping RX or TX buffer");

	return(ret);
	}

	if (openair0_num_detected_cards>MAX_CARDS) {
	printf ("Detected %d number of cards, but MAX_CARDS=%d\n", openair0_num_detected_cards, MAX_CARDS);
	} else {
	printf ("Detected %d number of cards, %d number of antennas.\n", openair0_num_detected_cards, openair0_num_antennas[0]);
	}

	//  p_exmimo_config = openair0_exmimo_pci[0].exmimo_config_ptr;
	p_exmimo_id     = openair0_exmimo_pci[0].exmimo_id_ptr;

	printf("Card %d: ExpressMIMO %d, HW Rev %d, SW Rev 0x%d\n", 0, p_exmimo_id->board_exmimoversion, p_exmimo_id->board_hwrev, p_exmimo_id->board_swrev);

	// check if the software matches firmware
	if (p_exmimo_id->board_swrev!=BOARD_SWREV_CNTL2) {
	printf("Software revision %d and firmware revision %d do not match. Please update either the firmware or the software!\n",BOARD_SWREV_CNTL2,p_exmimo_id->board_swrev);
	return(-1);
	}

	return(0);
}

int openair0_config(openair0_config_t *openair0_cfg, int UE_flag)
{
	int	ret;
	int	ant;
	int	card;
	int	resampling_factor	= 2;
	int	rx_filter			= RXLPF25;
	int	tx_filter			= TXLPF25;

	exmimo_config_t			*p_exmimo_config;
	exmimo_id_t				*p_exmimo_id;

	if (!openair0_cfg)
	{
		printf("Error, openair0_cfg is null!!\n");
		return(-1);
	}

	for (card = 0; card < openair0_num_detected_cards; card++)
	{
		p_exmimo_config	= openair0_exmimo_pci[card].exmimo_config_ptr;
		p_exmimo_id		= openair0_exmimo_pci[card].exmimo_id_ptr;

		if (p_exmimo_id->board_swrev >= 9)
			p_exmimo_config->framing.eNB_flag	= 0;
		else
			p_exmimo_config->framing.eNB_flag	= !UE_flag;

		p_exmimo_config->framing.tdd_config	= DUPLEXMODE_FDD + TXRXSWITCH_LSB;

		if (openair0_num_detected_cards == 1)
			p_exmimo_config->framing.multicard_syncmode	= SYNCMODE_FREE;
		else if (card == 0)
			p_exmimo_config->framing.multicard_syncmode	= SYNCMODE_MASTER;
		else
			p_exmimo_config->framing.multicard_syncmode	= SYNCMODE_SLAVE;

		if (openair0_cfg[card].sample_rate == 30.72e6)
		{
			resampling_factor	= 0;
			rx_filter			= RXLPF10;
			tx_filter			= TXLPF10;
		} else if (openair0_cfg[card].sample_rate == 15.36e6)
		{
			resampling_factor	= 1;
			rx_filter			= RXLPF5;
			tx_filter			= TXLPF5;
		} else if (openair0_cfg[card].sample_rate == 7.68e6)
		{
			resampling_factor	= 2;
			rx_filter			= RXLPF25;
			tx_filter			= TXLPF25;
		}
		else
		{
			printf("Sampling rate not supported, using default 7.68MHz");
			resampling_factor	= 2;
			rx_filter			= RXLPF25;
			tx_filter			= TXLPF25;
		}

#if (BOARD_SWREV_CNTL2 >= 0x0A)
		for (ant = 0; ant < 4; ant++)
			p_exmimo_config->framing.resampling_factor[ant] = resampling_factor;
#else
		p_exmimo_config->framing.resampling_factor = resampling_factor;
#endif
		for (ant=0; ant<4; ant++)
		{
			if (openair0_cfg[card].rx_freq[ant] || openair0_cfg[card].tx_freq[ant])
			{
				p_exmimo_config->rf.rf_mode[ant]	= RF_MODE_BASE;
				p_exmimo_config->rf.do_autocal[ant]	= 1;//openair0_cfg[card].autocal[ant];
				printf("card %d, antenna %d, autocal %d\n", card, ant, openair0_cfg[card].autocal[ant]);
			}

			if (openair0_cfg[card].tx_freq[ant])
			{
				p_exmimo_config->rf.rf_mode[ant]	+= (TXEN + DMAMODE_TX + TXLPFNORM + TXLPFEN + tx_filter);
				p_exmimo_config->rf.rf_freq_tx[ant]	= (unsigned int)openair0_cfg[card].tx_freq[ant];
				p_exmimo_config->rf.tx_gain[ant][0]	= (unsigned int)openair0_cfg[card].tx_gain[ant];
			}

			if (openair0_cfg[card].rx_freq[ant])
			{
				p_exmimo_config->rf.rf_mode[ant]	+= (RXEN + DMAMODE_RX + RXLPFNORM + RXLPFEN + rx_filter);
				p_exmimo_config->rf.rf_freq_rx[ant]	= (unsigned int)openair0_cfg[card].rx_freq[ant];
				p_exmimo_config->rf.rx_gain[ant][0]	= (unsigned int)openair0_cfg[card].rx_gain[ant];
				printf("openair0 : programming card %d RX antenna %d (freq %u, gain %d)\n",card,ant,p_exmimo_config->rf.rf_freq_rx[ant],p_exmimo_config->rf.rx_gain[ant][0]);

				p_exmimo_config->rf.rf_mode[ant]	= p_exmimo_config->rf.rf_mode[ant] & ~LNAMASK;
				if ( (300000000 <= p_exmimo_config->rf.rf_freq_rx[ant]) && (p_exmimo_config->rf.rf_freq_rx[ant] <= 2200000000) )
				{
					p_exmimo_config->rf.rf_mode[ant]	+= LNA1ON;
					printf("openair0 : programming card%d antenna %d LNA 1\n", card, ant);
				}
				else if ( (1500000000 <= p_exmimo_config->rf.rf_freq_rx[ant]) && (p_exmimo_config->rf.rf_freq_rx[ant] <= 3800000000) )
				{
					p_exmimo_config->rf.rf_mode[ant]	+= LNA2ON;
					printf("openair0 : programming card%d antenna %d LNA 2\n", card, ant);
				}
				else
				{
					p_exmimo_config->rf.rf_mode[ant]	+= LNA1ON;	// keep default value
					printf("openair0 : programming card%d antenna %d LNA UNKNOWN\n", card, ant);
				}

				switch (openair0_cfg[card].rxg_mode[ant])
				{
					default:
					case max_gain:
						p_exmimo_config->rf.rf_mode[ant]	+= LNAMax;
						break;
					case med_gain:
						p_exmimo_config->rf.rf_mode[ant]	+= LNAMed;
						break;
					case byp_gain:
						p_exmimo_config->rf.rf_mode[ant]	+= LNAByp;
						break;
				}
			}
			else
			{
				p_exmimo_config->rf.rf_mode[ant]	= 0;
				p_exmimo_config->rf.do_autocal[ant]	= 0;
			}

			p_exmimo_config->rf.rf_local[ant]	= rf_local[ant];
			p_exmimo_config->rf.rf_rxdc[ant]	= rf_rxdc[ant];

			if (( p_exmimo_config->rf.rf_freq_tx[ant] >= 850000000) && ( p_exmimo_config->rf.rf_freq_tx[ant] <= 865000000))
			{
				p_exmimo_config->rf.rf_vcocal[ant]  = rf_vcocal_850[ant];
				p_exmimo_config->rf.rffe_band_mode[ant] = DD_TDD;
			}
			else if (( p_exmimo_config->rf.rf_freq_tx[ant] >= 1900000000) && ( p_exmimo_config->rf.rf_freq_tx[ant] <= 2000000000))
			{
				p_exmimo_config->rf.rf_vcocal[ant]  = rf_vcocal[ant];
				p_exmimo_config->rf.rffe_band_mode[ant] = B19G_TDD;
			}
			else
			{
				p_exmimo_config->rf.rf_vcocal[ant]  = rf_vcocal[ant];
				p_exmimo_config->rf.rffe_band_mode[ant] = 0;
			}
			printf("openair0 : ioctl p_exmimo_config->rf.rf_mode[ant]=0x%x\n", p_exmimo_config->rf.rf_mode[ant]);
		}

		ret	= ioctl(openair0_fd, openair_DUMP_CONFIG, card);
		if (ret != 0)
			return (-1);
	}
	return(0);
}

int openair0_reconfig(openair0_config_t *openair0_cfg)
{
	int ant, card;

	exmimo_config_t         *p_exmimo_config;
	//  exmimo_id_t             *p_exmimo_id;

	if (!openair0_cfg) {
	printf("Error, openair0_cfg is null!!\n");
	return(-1);
	}

	for (card=0; card<openair0_num_detected_cards; card++) {

	p_exmimo_config = openair0_exmimo_pci[card].exmimo_config_ptr;
	//    p_exmimo_id     = openair0_exmimo_pci[card].exmimo_id_ptr;

	for (ant=0; ant<4; ant++) {
	if (openair0_cfg[card].tx_freq[ant]) {
	p_exmimo_config->rf.rf_freq_tx[ant] = (unsigned int)openair0_cfg[card].tx_freq[ant];
	p_exmimo_config->rf.tx_gain[ant][0] = (unsigned int)openair0_cfg[card].tx_gain[ant];
	//printf("openair0 : programming TX antenna %d (freq %u, gain %d)\n",ant,p_exmimo_config->rf.rf_freq_tx[ant],p_exmimo_config->rf.tx_gain[ant][0]);
	}

	if (openair0_cfg[card].rx_freq[ant]) {
	p_exmimo_config->rf.rf_freq_rx[ant] = (unsigned int)openair0_cfg[card].rx_freq[ant];
	p_exmimo_config->rf.rx_gain[ant][0] = (unsigned int)openair0_cfg[card].rx_gain[ant];
	//printf("openair0 : programming RX antenna %d (freq %u, gain %d)\n",ant,p_exmimo_config->rf.rf_freq_rx[ant],p_exmimo_config->rf.rx_gain[ant][0]);

	switch (openair0_cfg[card].rxg_mode[ant]) {
	default:
	case max_gain:
	p_exmimo_config->rf.rf_mode[ant] = (p_exmimo_config->rf.rf_mode[ant]&(~LNAGAINMASK))|LNAMax;
	break;

	case med_gain:
	p_exmimo_config->rf.rf_mode[ant] = (p_exmimo_config->rf.rf_mode[ant]&(~LNAGAINMASK))|LNAMed;
	break;

	case byp_gain:
	p_exmimo_config->rf.rf_mode[ant] = (p_exmimo_config->rf.rf_mode[ant]&(~LNAGAINMASK))|LNAByp;
	break;
	}
	}
	}
	}

	return(0);
}

int openair0_set_frequencies(openair0_device* device, openair0_config_t *openair0_cfg,int exmimo_dump_config)
{
	if (exmimo_dump_config > 0) {
	// do a full configuration
	openair0_config(openair0_cfg,0);
	}
	else {  // just change the frequencies in pci descriptor
	openair0_reconfig(openair0_cfg);
	}
	return(0);
}

int openair0_set_gains(openair0_device* device, openair0_config_t *openair0_cfg){
	return(0);
}

unsigned int *openair0_daq_cnt(void)
{
	return((unsigned int *)openair0_exmimo_pci[0].rxcnt_ptr[0]);
}



/* SYRTEM functions */
int syrtem_get_ctrl0(argument_t *arg)
{
	return ioctl(openair0_fd, syrtem_GET_CTRL0, arg);
}

int syrtem_get_ctrl1(argument_t *arg)
{
	return ioctl(openair0_fd, syrtem_GET_CTRL1, arg);
}

int syrtem_get_ctrl2(argument_t *arg)
{
	return ioctl(openair0_fd, syrtem_GET_CTRL2, arg);
}

int syrtem_set_ctrl0(argument_t *arg)
{
	return ioctl(openair0_fd, syrtem_SET_CTRL0, arg);
}

int syrtem_set_ctrl1(argument_t *arg)
{
	return ioctl(openair0_fd, syrtem_SET_CTRL1, arg);
}

int syrtem_set_ctrl2(argument_t *arg)
{
	return ioctl(openair0_fd, syrtem_SET_CTRL2, arg);
}

int syrtem_get_counters(void)
{
	return ioctl(openair0_fd, syrtem_GET_COUNTERS, NULL);
}

static unsigned char	tid		= 0x00;

void *syrtem_spi_initctxt(void)
{
	ioctl_arg_spi_t		*spictxt	= (ioctl_arg_spi_t *)malloc(sizeof(ioctl_arg_spi_t));
	return (void *)spictxt;
}

void syrtem_spi_finictxt(void *spictxt)
{
	free(spictxt);
	return;
}

syr_pio_cmdretval_e syrtem_spi_write(void *spictxt, unsigned char reg, unsigned char data)
{
	syr_pio_cmdretval_e	retval	= SYR_PIO_CMDSEND;
	int					ioctl_retval	= 0;
//	printf("syrtem_spi_write called REG(0x%02X) DATA(0x%02X)\n", reg, data);
	if (spictxt)
	{
		if (tid >= 0x0F)
			tid	= 0x00;
		tid++;
		((ioctl_arg_spi_t *)spictxt)->card_id			= 0;
		((ioctl_arg_spi_t *)spictxt)->spi_rqrs			= SYR_PIO_SPI_REQUEST;
		((ioctl_arg_spi_t *)spictxt)->spi_rw			= SYR_PIO_SPI_WRITE;
		((ioctl_arg_spi_t *)spictxt)->spi_tid			= tid;
		((ioctl_arg_spi_t *)spictxt)->spi_reg			= reg;
		((ioctl_arg_spi_t *)spictxt)->spi_data			= data;
		((ioctl_arg_spi_t *)spictxt)->spi_pio_result	= 0;
		((ioctl_arg_spi_t *)spictxt)->drv_comp			= IOCTL_ARG_UNKNOWN;
		ioctl_retval	= ioctl(openair0_fd, syrtem_SPI_WRITE, spictxt);
		if (ioctl_retval != 0)
		{
			printf("   ioctl_retval is not NULL (%d)\n", ioctl_retval);
			retval	= SYR_PIO_CMDERRIOCTL;
		}
		else
		{
			if (((ioctl_arg_spi_t *)spictxt)->drv_comp == IOCTL_ARG_SUCCESS)
			{
				retval	= SYR_PIO_CMDSEND;
			}
			else	// still IOCTL_ARG_UNKNOWN or fill IOCTL_ARG_FAILED by Driver
			{
				printf("   driver completion is not IOCTL_ARG_SUCCESS (%d)\n", ((ioctl_arg_spi_t *)spictxt)->drv_comp);
				retval	= SYR_PIO_CMDERRDRV;
			}
		}
	}
	else
	{
		retval	= SYR_PIO_CMDERRAPP;
	}
	return retval;
}

syr_pio_cmdretval_e syrtem_spi_read(void *spictxt, unsigned char reg)
{
	syr_pio_cmdretval_e	retval			= SYR_PIO_CMDSEND;
	int					ioctl_retval	= 0;
//	printf("syrtem_spi_read called REG(0x%02X)\n", reg);
	if (spictxt)
	{
//		printf("   spictxt is not NULL (0x%08X)\n", spictxt);
		if (tid >= 0x0F)
			tid	= 0x00;
		tid++;
		((ioctl_arg_spi_t *)spictxt)->card_id			= 0;
		((ioctl_arg_spi_t *)spictxt)->spi_rqrs			= SYR_PIO_SPI_REQUEST;
		((ioctl_arg_spi_t *)spictxt)->spi_rw			= SYR_PIO_SPI_READ;
		((ioctl_arg_spi_t *)spictxt)->spi_tid			= tid;
		((ioctl_arg_spi_t *)spictxt)->spi_reg			= reg;
		((ioctl_arg_spi_t *)spictxt)->spi_data			= 0xFF;
		((ioctl_arg_spi_t *)spictxt)->spi_pio_result	= 0;
		((ioctl_arg_spi_t *)spictxt)->drv_comp			= IOCTL_ARG_UNKNOWN;
		ioctl_retval	= ioctl(openair0_fd, syrtem_SPI_READ, spictxt);
		if (ioctl_retval != 0)
		{
			printf("   ioctl_retval is not NULL (%d)\n", ioctl_retval);
			retval	= SYR_PIO_CMDERRIOCTL;
		}
		else
		{
//			printf("   ioctl(openair0_fd, syrtem_SPI_READ, spictxt) DONE\n");
			if (((ioctl_arg_spi_t *)spictxt)->drv_comp == IOCTL_ARG_SUCCESS)
			{
//				printf("   driver completion is IOCTL_ARG_SUCCESS\n");
				retval	= SYR_PIO_CMDSEND;
			}
			else	// still IOCTL_ARG_UNKNOWN or fill IOCTL_ARG_FAILED by Driver
			{
				printf("   driver completion is not IOCTL_ARG_SUCCESS (%d)\n", ((ioctl_arg_spi_t *)spictxt)->drv_comp);
				retval	= SYR_PIO_CMDERRDRV;
			}
		}
	}
	else
	{
		printf("   spictxt is NULL\n");
		retval	= SYR_PIO_CMDERRAPP;
	}
	return retval;
}

syr_pio_cmdretval_e syrtem_spi_chkstatus(void)
{
	syr_pio_cmdretval_e	retval		= SYR_PIO_CMDIDLE;
	argument_t 			arg;

	arg.card_id	= 0;
	arg.value	= 0;
//	printf("syrtem_spi_chkstatus : ioctl(openair0_fd, syrtem_GET_CTRL1, &arg)\n");
	if (ioctl(openair0_fd, syrtem_GET_CTRL1, &arg))
	{
		retval	= SYR_PIO_CMDERRDRV;
	}
	else
	{
//		printf("   syrtem_spi_chkstatus VALUE(0x%08X)\r\n", arg.value);
		if ((((arg.value & PIO_COMMAND_MASK) >> PIO_COMMAND_SHIFT) & PIO_COMMAND_SIZE) == PIO_CMD_SYR_SPI_FLAG)
		{
			if ((((arg.value & PIO_SPI_DREADY_MASK) >> PIO_SPI_DREADY_SHIFT) & PIO_SPI_DREADY_SIZE) == SYR_PIO_SPI_DATAREADY)	// DATA READY is SET
			{
				if ((((arg.value & PIO_SPI_REQRESP_MASK) >> PIO_SPI_REQRESP_SHIFT) & PIO_SPI_REQRESP_SIZE) == SYR_PIO_SPI_REQUEST)
				{
					retval	= SYR_PIO_CMDSEND;
				}
				else	// (arg.value & PIO_SPI_REQRESP_MASK) == SYR_PIO_SPI_RESPONSE
				{
					retval	= SYR_PIO_CMDRECV;
				}
			}
			else	// (arg.value & PIO_SPI_DREADY_MASK) == SYR_PIO_SPI_IDLE : DATA READY is CLEARED
			{
				retval	= SYR_PIO_CMDIDLE;
//				leon3_spiwrite_intr_cnt	= arg.value & 0xFFFF;
			}
		}
	}
	return retval;
}

syr_pio_cmdretval_e syrtem_spi_getdata(void *spictxt, unsigned char *data)
{
	pio_spi_t			spictxt_recv; /*	= (pio_spi_t *)malloc(sizeof(pio_spi_t));*/
	syr_pio_cmdretval_e	retval			= SYR_PIO_CMDIDLE;
	argument_t 			arg;

	(*data)	= 0x00;
	arg.card_id	= 0;
	arg.value	= 0;
	if (ioctl(openair0_fd, syrtem_GET_CTRL1, &arg))
	{
		retval	= SYR_PIO_CMDERRDRV;
	}
	else
	{
//		printf("   syrtem_spi_getdata VALUE(0x%08X)\r\n", arg.value);
		if ((((arg.value & PIO_COMMAND_MASK) >> PIO_COMMAND_SHIFT) & PIO_COMMAND_SIZE) == PIO_CMD_SYR_SPI_FLAG)
		{
//			printf("   (arg.value & PIO_COMMAND_MASK) == PIO_CMD_SYR_SPI_FLAG\r\n");
			if ((((arg.value & PIO_SPI_DREADY_MASK) >> PIO_SPI_DREADY_SHIFT) & PIO_SPI_DREADY_SIZE) == SYR_PIO_SPI_DATAREADY)	// DATA READY is SET
			{
//				printf("   (arg.value & PIO_SPI_DREADY_MASK) == SYR_PIO_SPI_DATAREADY\r\n");
				if ((((arg.value & PIO_SPI_REQRESP_MASK) >> PIO_SPI_REQRESP_SHIFT) & PIO_SPI_REQRESP_SIZE) == SYR_PIO_SPI_REQUEST)	// REQUEST
				{
					printf("   (arg.value & PIO_SPI_REQRESP_MASK) == SYR_PIO_SPI_REQUEST\r\n");
					retval	= SYR_PIO_CMDSEND;
				}
				else	// (arg.value & PIO_SPI_REQRESP_MASK) == SYR_PIO_SPI_RESPONSE
				{
//					printf("   (arg.value & PIO_SPI_REQRESP_MASK) == SYR_PIO_SPI_RESPONSE\r\n");
					spictxt_recv.spi_dready		= ((arg.value & PIO_SPI_DREADY_MASK)	>> PIO_SPI_DREADY_SHIFT)	& PIO_SPI_DREADY_SIZE;
					spictxt_recv.spi_rqrs		= ((arg.value & PIO_SPI_REQRESP_MASK)	>> PIO_SPI_REQRESP_SHIFT)	& PIO_SPI_REQRESP_SIZE;
					spictxt_recv.spi_rw			= ((arg.value & PIO_SPI_RW_MASK)		>> PIO_SPI_RW_SHIFT)		& PIO_SPI_RW_SIZE;
					spictxt_recv.spi_tid		= ((arg.value & PIO_SPI_TID_MASK)		>> PIO_SPI_TID_SHIFT)		& PIO_SPI_TID_SIZE;
					spictxt_recv.spi_reg		= ((arg.value & PIO_SPI_REG_MASK)		>> PIO_SPI_REG_SHIFT)		& PIO_SPI_REG_SIZE;
					spictxt_recv.spi_data		= ((arg.value & PIO_SPI_DATA_MASK)		>> PIO_SPI_DATA_SHIFT)		& PIO_SPI_DATA_SIZE;
					if (	   ((((ioctl_arg_spi_t *)spictxt)->spi_rw)  == spictxt_recv.spi_rw)
							&& ((((ioctl_arg_spi_t *)spictxt)->spi_tid) == spictxt_recv.spi_tid) )
					{
						(*data)	= spictxt_recv.spi_data;
						retval	= SYR_PIO_CMDRECV;
//						printf("   retval	= SYR_PIO_CMDRECV\r\n");
					}
					else
					{
						retval	= SYR_PIO_CMDNOTFORTID;
						printf("   retval	= SYR_PIO_CMDNOTFORTID\r\n");
					}
				}
			}
//			else	// (arg.value & PIO_SPI_DREADY_MASK) == SYR_PIO_SPI_IDLE : DATA READY is CLEARED
//			{
//				retval	= SYR_PIO_CMDIDLE;
//			}
		}
	}
	return retval;
}

syr_pio_cmdretval_e syrtem_cfg_write(void *cfgctxt, unsigned char reg, unsigned char data)
{
	syr_pio_cmdretval_e	retval	= SYR_PIO_CMDSEND;
	int					ioctl_retval	= 0;
//	printf("syrtem_spi_write called REG(0x%02X) DATA(0x%02X)\n", reg, data);
	if (cfgctxt)
	{
		if (tid >= 0x0F)
			tid	= 0x00;
		tid++;
		((ioctl_arg_cfg_t *)cfgctxt)->card_id			= 0;
		((ioctl_arg_cfg_t *)cfgctxt)->cfg_rqrs			= SYR_PIO_CFG_REQUEST;
		((ioctl_arg_cfg_t *)cfgctxt)->cfg_rw			= SYR_PIO_CFG_WRITE;
		((ioctl_arg_cfg_t *)cfgctxt)->cfg_tid			= tid;
		((ioctl_arg_cfg_t *)cfgctxt)->cfg_reg			= reg;
		((ioctl_arg_cfg_t *)cfgctxt)->cfg_data			= data;
		((ioctl_arg_cfg_t *)cfgctxt)->cfg_pio_result	= 0;
		((ioctl_arg_cfg_t *)cfgctxt)->drv_comp			= IOCTL_ARG_UNKNOWN;
		ioctl_retval	= ioctl(openair0_fd, syrtem_CFG_WRITE, cfgctxt);
		if (ioctl_retval != 0)
		{
			printf("   ioctl_retval is not NULL (%d)\n", ioctl_retval);
			retval	= SYR_PIO_CMDERRIOCTL;
		}
		else
		{
			if (((ioctl_arg_cfg_t *)cfgctxt)->drv_comp == IOCTL_ARG_SUCCESS)
			{
				retval	= SYR_PIO_CMDSEND;
			}
			else	// still IOCTL_ARG_UNKNOWN or fill IOCTL_ARG_FAILED by Driver
			{
				printf("   driver completion is not IOCTL_ARG_SUCCESS (%d)\n", ((ioctl_arg_cfg_t *)cfgctxt)->drv_comp);
				retval	= SYR_PIO_CMDERRDRV;
			}
		}
	}
	else
	{
		retval	= SYR_PIO_CMDERRAPP;
	}
	return retval;
}

syr_pio_cmdretval_e syrtem_cfg_chkstatus(void)
{
	syr_pio_cmdretval_e	retval		= SYR_PIO_CMDIDLE;
	argument_t 			arg;

	arg.card_id	= 0;
	arg.value	= 0;
//	printf("syrtem_spi_chkstatus : ioctl(openair0_fd, syrtem_GET_CTRL1, &arg)\n");
	if (ioctl(openair0_fd, syrtem_GET_CTRL1, &arg))
	{
		retval	= SYR_PIO_CMDERRDRV;
	}
	else
	{
//		printf("   syrtem_spi_chkstatus VALUE(0x%08X)\r\n", arg.value);
		if ((((arg.value & PIO_COMMAND_MASK) >> PIO_COMMAND_SHIFT) & PIO_COMMAND_SIZE) == PIO_CMD_SYR_CFG_FLAG)
		{
			if ((((arg.value & PIO_CFG_DREADY_MASK) >> PIO_CFG_DREADY_SHIFT) & PIO_CFG_DREADY_SIZE) == SYR_PIO_CFG_DATAREADY)	// DATA READY is SET
			{
				if ((((arg.value & PIO_CFG_REQRESP_MASK) >> PIO_CFG_REQRESP_SHIFT) & PIO_CFG_REQRESP_SIZE) == SYR_PIO_CFG_REQUEST)
				{
					retval	= SYR_PIO_CMDSEND;
				}
				else	// (arg.value & PIO_CFG_REQRESP_MASK) == SYR_PIO_CFG_RESPONSE
				{
					retval	= SYR_PIO_CMDRECV;
				}
			}
			else	// (arg.value & PIO_CFG_DREADY_MASK) == SYR_PIO_CFG_IDLE : DATA READY is CLEARED
			{
				retval	= SYR_PIO_CMDIDLE;
//				leon3_spiwrite_intr_cnt	= arg.value & 0xFFFF;
			}
		}
	}
	return retval;
}

void *syrtem_int_initctxt(void)
{
	ioctl_arg_int_t		*intctxt	= (ioctl_arg_int_t *)malloc(sizeof(ioctl_arg_int_t));
	return (void *)intctxt;
}

void syrtem_int_finictxt(void *intctxt)
{
	free(intctxt);
	return;
}

syr_pio_cmdretval_e syrtem_int_exec(void *intctxt)
{
	syr_pio_cmdretval_e	retval			= SYR_PIO_CMDSEND;
	int					ioctl_retval	= 0;

	if (intctxt)
	{
		if (tid >= 0x0F)
			tid	= 0x00;
		tid++;
		((ioctl_arg_int_t *)intctxt)->card_id			= 0;
		((ioctl_arg_int_t *)intctxt)->int_rqrs			= SYR_PIO_INT_REQUEST;
		((ioctl_arg_int_t *)intctxt)->int_rw			= SYR_PIO_INT_EXEC;
		((ioctl_arg_int_t *)intctxt)->int_tid			= tid;
		((ioctl_arg_int_t *)intctxt)->int_cnt			= 0;
		((ioctl_arg_int_t *)intctxt)->int_pio_result	= 0;
		((ioctl_arg_int_t *)intctxt)->drv_comp			= IOCTL_ARG_UNKNOWN;
		ioctl_retval	= ioctl(openair0_fd, syrtem_INT_EXEC, intctxt);
		if (ioctl_retval != 0)
		{
//			printf("   ioctl_retval is not NULL (%d)\n", ioctl_retval);
			retval	= SYR_PIO_CMDERRIOCTL;
		}
		else
		{
			if (((ioctl_arg_int_t *)intctxt)->drv_comp == IOCTL_ARG_SUCCESS)
			{
				retval	= SYR_PIO_CMDSEND;
			}
			else	// still IOCTL_ARG_UNKNOWN or fill IOCTL_ARG_FAILED by Driver
			{
//				printf("   driver completion is not IOCTL_ARG_SUCCESS (%d)\n", ((ioctl_arg_int_t *)intctxt)->drv_comp);
				retval	= SYR_PIO_CMDERRDRV;
			}
		}
	}
	else
	{
		retval	= SYR_PIO_CMDERRAPP;
	}
	return retval;
}

syr_pio_cmdretval_e syrtem_int_chkstatus(void)
{
	syr_pio_cmdretval_e	retval		= SYR_PIO_CMDIDLE;
	argument_t 			arg;

	if (ioctl(openair0_fd, syrtem_GET_CTRL1, &arg))
	{
		retval	= SYR_PIO_CMDERRDRV;
	}
	else
	{
		if ((((arg.value & PIO_COMMAND_MASK) >> PIO_COMMAND_SHIFT) & PIO_COMMAND_SIZE) == PIO_CMD_SYR_INT_FLAG)
		{
			if ((((arg.value & PIO_INT_DREADY_MASK) >> PIO_INT_DREADY_SHIFT) & PIO_INT_DREADY_SIZE) == SYR_PIO_INT_DATAREADY)	// DATA READY is SET
			{
				if ((((arg.value & PIO_INT_REQRESP_MASK) >> PIO_INT_REQRESP_SHIFT) & PIO_INT_REQRESP_SIZE) == SYR_PIO_INT_REQUEST)
				{
					retval	= SYR_PIO_CMDSEND;
				}
				else	// (arg.value & PIO_INT_REQRESP_MASK) == SYR_PIO_INT_RESPONSE
				{
					retval	= SYR_PIO_CMDRECV;
					leon3_intr_exec_cnt	= arg.value & 0xFFFF;
				}
			}
			else	// (arg.value & PIO_INT_DREADY_MASK) == SYR_PIO_INT_IDLE : DATA READY is CLEARED
			{
				retval	= SYR_PIO_CMDIDLE;
				leon3_call_exec_cnt	= arg.value & 0xFFFF;
			}
		}
	}
	return retval;
}
