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

#include <stdio.h>
#include <unistd.h>
#include "openair0_lib.h"

#include "syr_pio.h"
#include "lime_spi_cmd.h"
#include "lime_reg_cmd.h"
#include "lime_cfg_cmd.h"
#include "lime_cal_cmd.h"

#define ENABLE_CALIBRATION					1

#define ENABLE_SOFT_RESET					1
#define ENABLE_INIT_DEVICE					1
#define ENABLE_BLADERF_DC_CAL_LPF_TUNING	1
#define ENABLE_CAL_TX_LPF					1
#define ENABLE_BLADERF_DC_CAL_RX_LPF		1
#define ENABLE_BLADERF_DC_CAL_RXVGA2		1
#define ENABLE_BLADERF_MODULE_RX			1
#define ENABLE_BLADERF_MODULE_TX			0	// Not working
#define ENABLE_BLADERF_MODULE_TX_MANUAL		1

//extern uint32_t	leon3_spiwrite_intr_cnt;
//extern uint32_t	leon3_call_exec_cnt;
//extern uint32_t	leon3_intr_exec_cnt;

void				*cmdcontext_spi		= NULL;

void main(void)
{
	int				i				= 0;
	int				retval			= 0;
	unsigned char	reg0x00_val		= 0;
	unsigned char	reg0x01_val		= 0;
	unsigned char	reg0x03_val		= 0;
	unsigned char	reg0x04_val		= 0;
	unsigned char	reg0x05_val		= 0;
	unsigned char	reg0x06_val		= 0;
	unsigned char	reg0x07_val		= 0;
	unsigned char	reg0x09_val		= 0;
	unsigned char	reg0x10_val		= 0;
	unsigned char	reg0x11_val		= 0;
	unsigned char	reg0x12_val		= 0;
	unsigned char	reg0x13_val		= 0;
	unsigned char	reg0x14_val		= 0;
	unsigned char	reg0x15_val		= 0;
	unsigned char	reg0x30_val		= 0;
	unsigned char	reg0x31_val		= 0;
	unsigned char	reg0x32_val		= 0;
	unsigned char	reg0x33_val		= 0;
	unsigned char	reg0x34_val		= 0;
	unsigned char	reg0x35_val		= 0;
	unsigned char	reg0x36_val		= 0;
	unsigned char	reg0x41_val		= 0;
	unsigned char	reg0x42_val		= 0;
	unsigned char	reg0x43_val		= 0;
	unsigned char	reg0x44_val		= 0;
	unsigned char	reg0x45_val		= 0;
	unsigned char	reg0x4B_val		= 0;
	unsigned char	reg0x50_val		= 0;
	unsigned char	reg0x52_val		= 0;
	unsigned char	reg0x53_val		= 0;
	unsigned char	reg0x55_val		= 0;
	unsigned char	reg0x56_val		= 0;
	unsigned char	reg0x5A_val		= 0;
	unsigned char	reg0x60_val		= 0;
	unsigned char	reg0x62_val		= 0;
	unsigned char	reg0x63_val		= 0;
	unsigned char	clk_en			= 0;
	unsigned char	clk_en5			= 0;
	unsigned char	dccal			= 0;
	unsigned char	rccal_lpf		= 0;
	unsigned char	c				= 0;

	int								status	= 0;
	struct dc_calibration_params	p;

	system("/bin/stty raw");

	retval	= openair0_open();
	if (retval != 0)
	{
		if (retval == (-1))
			printf("Error opening /dev/openair0\r\n");
		if (retval == (-2))
			printf("Error mapping bigshm\r\n");
		if (retval == (-3))
			printf("Error mapping RX or TX buffer\r\n");
		system("/bin/stty cooked");
		return;
	}
	printf("\rOpenair0 opened\r\n");

	cmdcontext_spi	=	syrtem_spi_initctxt();

	lime_spi_blk_write(cmdcontext_spi,	0x09, 0x45);	// EURECOM value 0xC5, DEFAULT value 0x40
//	lime_spi_blk_write(cmdcontext_spi,	0x09, 0xC5);	// EURECOM value 0xC5, DEFAULT value 0x40
	lime_spi_blk_write(cmdcontext_spi,	0x5A, 0x30);	// EURECOM value 0x30, DEFAULT value 0x20
	lime_spi_blk_write(cmdcontext_spi,	0x75, 0xE0);	// EURECOM value 0xE0 (LNA2), DEFAULT value 0xD0 (LNA1)

#if ENABLE_CALIBRATION
#if ENABLE_SOFT_RESET
	lms_soft_reset(cmdcontext_spi);
	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);
#endif

#if ENABLE_INIT_DEVICE
	printf("\r************************************************************************\r\n");
	printf("\r* init_device(cmdcontext_spi);                                         *\r\n");
	init_device(cmdcontext_spi);
	printf("\r************************************************************************\r\n");
	printf("\r* lms_set_frequency(cmdcontext_spi, BLADERF_MODULE_RX, 2685000000);    *\r\n");
	lms_set_frequency(cmdcontext_spi, BLADERF_MODULE_RX, 2685000000);
	printf("\r************************************************************************\r\n");
	printf("\r* lms_set_frequency(cmdcontext_spi, BLADERF_MODULE_TX, 2565000000);    *\r\n");
	lms_set_frequency(cmdcontext_spi, BLADERF_MODULE_TX, 2565000000);
	printf("\r************************************************************************\r\n");
	printf("\r* lms_rxvga2_set_gain(cmdcontext_spi, 30);                             *\r\n");
	lms_rxvga2_set_gain(cmdcontext_spi, 30);
	printf("\r************************************************************************\r\n");
	printf("\r* lms_txvga2_set_gain(cmdcontext_spi, 0);                              *\r\n");
	lms_txvga2_set_gain(cmdcontext_spi, 0);
#endif

#if ENABLE_BLADERF_DC_CAL_LPF_TUNING
	printf("\r************************************************************************\r\n");
	printf("\r* bladerf_calibrate_dc(cmdcontext_spi, BLADERF_DC_CAL_LPF_TUNING);     *\r\n");
	bladerf_calibrate_dc(cmdcontext_spi, BLADERF_DC_CAL_LPF_TUNING);
	// save in Leon3 Storage Calib
	lime_spi_blk_read(cmdcontext_spi,	0x35, &reg0x35_val);
	lime_spi_blk_read(cmdcontext_spi,	0x55, &reg0x55_val);
	printf("   - DCCAL : reg0x35_val=0x%.2x: \r\n", reg0x35_val&0x3F);
	printf("   - DCCAL : reg0x55_val=0x%.2x: \r\n", reg0x55_val&0x3F);
	if ( (reg0x35_val & 0x3F) == (reg0x55_val & 0x3F) )
		lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_LPF_TUNING_MOD, reg0x35_val&0x3F);
#endif

#if ENABLE_CAL_TX_LPF
	printf("\r************************************************************************\r\n");
	printf("\r* cal_tx_lpf(cmdcontext_spi);                                          *\r\n");
	cal_tx_lpf_init(cmdcontext_spi);
	cal_tx_lpf(cmdcontext_spi);
	cal_tx_lpf_deinit(cmdcontext_spi);

	lime_spi_blk_write(cmdcontext_spi,	0x33, 0x08);
	lime_spi_blk_read(cmdcontext_spi,	0x30, &reg0x30_val);
	printf("   - I : reg0x30_val=0x%.2x: \r\n", reg0x30_val&0x3F);
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_TXLPF_I, reg0x30_val&0x3F);

	lime_spi_blk_write(cmdcontext_spi,	0x33, 0x09);		/* DC_ADDR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x30, &reg0x30_val);
	printf("   - Q : reg0x30_val=0x%.2x: \r\n", reg0x30_val&0x3F);
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_TXLPF_Q, reg0x30_val&0x3F);
#endif

#if ENABLE_BLADERF_DC_CAL_RX_LPF
	printf("\r************************************************************************\r\n");
	printf("\r* bladerf_calibrate_dc(cmdcontext_spi, BLADERF_DC_CAL_RX_LPF);         *\r\n");
	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);

	bladerf_calibrate_dc(cmdcontext_spi, BLADERF_DC_CAL_RX_LPF);

	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x08);
	lime_spi_blk_read(cmdcontext_spi,	0x50, &reg0x50_val);
	printf("   - I : reg0x50_val=0x%.2x: \r\n", reg0x50_val&0x3F);
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXLPF_I, reg0x50_val&0x3F);

	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x09);		/* DC_ADDR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x50, &reg0x50_val);
	printf("   - Q : reg0x50_val=0x%.2x: \r\n", reg0x50_val&0x3F);
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXLPF_Q, reg0x50_val&0x3F);
#endif

#if ENABLE_BLADERF_DC_CAL_RXVGA2
	printf("\r************************************************************************\r\n");
	printf("\r* bladerf_calibrate_dc(cmdcontext_spi, BLADERF_DC_CAL_RXVGA2);         *\r\n");
	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);

	bladerf_calibrate_dc(cmdcontext_spi, BLADERF_DC_CAL_RXVGA2);

	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x08);
	lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);
	printf("   - I : reg0x60_val=0x%.2x: \r\n", reg0x60_val&0x3F);
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXVGA2_DC, reg0x60_val&0x3F);

	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x09);		/* DC_ADDR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);
	printf("   - Q : reg0x60_val=0x%.2x: \r\n", reg0x60_val&0x3F);
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXVGA2_Ia, reg0x60_val&0x3F);

	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0A);
	lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);
	printf("   - I : reg0x60_val=0x%.2x: \r\n", reg0x60_val&0x3F);
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXVGA2_Qa, reg0x60_val&0x3F);

	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0B);		/* DC_ADDR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);
	printf("   - Q : reg0x60_val=0x%.2x: \r\n", reg0x60_val&0x3F);
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXVGA2_Ib, reg0x60_val&0x3F);
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0C);
	lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);
	printf("   - I : reg0x60_val=0x%.2x: \r\n", reg0x60_val&0x3F);
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXVGA2_Qb, reg0x60_val&0x3F);
#endif

#if 0	// TEST DUMP CONFIG TO LEON3
	exmimo_config_t	*p_exmimo_config	= NULL;
	exmimo_id_t		*p_exmimo_id		= NULL;
	int				card				= 0;
	int				ant					= 0;
	int				rffe_band_int;

	p_exmimo_config = openair0_exmimo_pci[0].exmimo_config_ptr;
	p_exmimo_id     = openair0_exmimo_pci[0].exmimo_id_ptr;
	p_exmimo_config->framing.eNB_flag   = 0;
	p_exmimo_config->framing.tdd_config = DUPLEXMODE_FDD + TXRXSWITCH_LSB;;
	p_exmimo_config->framing.multicard_syncmode = SYNCMODE_FREE;
	for (ant = 0; ant < 4; ant++)
	{
		p_exmimo_config->rf.do_autocal[ant]	= 1;
		if (ant)
		{
			p_exmimo_config->rf.rf_freq_rx[ant]				= 0;
			p_exmimo_config->rf.rf_freq_tx[ant]				= 0;	
			p_exmimo_config->rf.rx_gain[ant][0]				= (uint32_t) 30;
			p_exmimo_config->rf.tx_gain[ant][0]				= (uint32_t) 25;
			p_exmimo_config->rf.rf_mode[ant]				= 0;
			p_exmimo_config->rf.rf_local[ant]				= 1156692;
			p_exmimo_config->rf.rf_rxdc[ant]				= 32896;
			p_exmimo_config->rf.rf_vcocal[ant]				= ((0xE)*(2^6)) + (0xE);
			p_exmimo_config->rf.rffe_gain_txlow[ant]		= 31;
			p_exmimo_config->rf.rffe_gain_txhigh[ant]		= 31;
			p_exmimo_config->rf.rffe_gain_rxfinal[ant]		= 31;
			p_exmimo_config->rf.rffe_gain_rxlow[ant]		= 63;
			p_exmimo_config->framing.resampling_factor[ant] = 2; 
		}
		else
		{
			p_exmimo_config->rf.rf_freq_rx[ant]				= 2662000000;
			p_exmimo_config->rf.rf_freq_tx[ant]				= 2542000000;
			p_exmimo_config->rf.rx_gain[ant][0]				= (uint32_t) 30;
			p_exmimo_config->rf.tx_gain[ant][0]				= (uint32_t) 25;
			p_exmimo_config->rf.rf_mode[ant]				= (RXEN + TXEN + TXLPFNORM + TXLPFEN + TXLPF25 + RXLPFNORM + RXLPFEN + RXLPF25 + LNA2ON + LNAMax + RFBBNORM) + (DMAMODE_RX + DMAMODE_TX);
			p_exmimo_config->rf.rf_local[ant]				= 1156692;
			p_exmimo_config->rf.rf_rxdc[ant]				= 32896;
			p_exmimo_config->rf.rf_vcocal[ant]				= ((0xE)*(2^6)) + (0xE);
			p_exmimo_config->rf.rffe_gain_txlow[ant]		= 31;
			p_exmimo_config->rf.rffe_gain_txhigh[ant]		= 31;
			p_exmimo_config->rf.rffe_gain_rxfinal[ant]		= 31;
			p_exmimo_config->rf.rffe_gain_rxlow[ant]		= 63;
			p_exmimo_config->framing.resampling_factor[ant] = 2; 
		}
		p_exmimo_config->rf.rffe_band_mode[ant]				= TVWS_TDD;
	}
	openair0_dump_config(/*card=0*/0);
        
 	printf("Card %d: ExpressMIMO %d, HW Rev. 0x%d, SW Rev 0x%d, SVN Rev %d, Builddate %d,  %d antennas\n", card, p_exmimo_id->board_exmimoversion,
	       p_exmimo_id->board_hwrev, p_exmimo_id->board_swrev,
	       p_exmimo_id->system_id.bitstream_id, p_exmimo_id->system_id.bitstream_build_date,
	       openair0_num_antennas[card]);

	short	*dump_sig	= NULL;
	int		j			= 0;
	dump_sig = (short*) openair0_exmimo_pci[0].adc_head[0];
	printf("dump_sig = 0x%08lx\r\n", (long unsigned int)dump_sig);

	for( j=0; j<32;j++ )	
	{
		printf("i=%d rx_sig[I]=0x%hx, rx_sig[Q]=0x%hx, Ox%lx\r\n", j, dump_sig[j*2], dump_sig[j*2+1], (unsigned long)dump_sig[j*2]);
	}
#endif

#if ENABLE_BLADERF_MODULE_RX
	printf("\r************************************************************************\r\n");
	printf("\r* dc_calibration(cmdcontext_spi, BLADERF_MODULE_RX, &p, 1, false);     *\r\n");
	p.frequency	= 2662000000;
	p.corr_i	= 0;
	p.corr_q	= 0;
	p.error_i	= 0;
	p.error_q	= 0;

	status = dc_calibration(cmdcontext_spi, BLADERF_MODULE_RX, &p, 1, false);
	if (status == 0)
	{
		printf("F=%10u, Corr_I=%4d, Corr_Q=%4d, Error_I=%4.2f, Error_Q=%4.2f\r\n",
				p.frequency, p.corr_i, p.corr_q, p.error_i, p.error_q);
	}
	else
	{
		bladerf_strerror(status);
	}
#endif

#if ENABLE_BLADERF_MODULE_TX	// Not working
	printf("\r************************************************************************\r\n");
	printf("\r* dc_calibration(cmdcontext_spi, BLADERF_MODULE_TX, &p, 1, false);     *\r\n");
	p.frequency	= 2542000000;
	p.corr_i	= 0;
	p.corr_q	= 0;
	p.error_i	= 0;
	p.error_q	= 0;

	status = dc_calibration(cmdcontext_spi, BLADERF_MODULE_TX, &p, 1, false);
	if (status == 0)
	{
		printf("F=%10u, Corr_I=%4d, Corr_Q=%4d, Error_I=%4.2f, Error_Q=%4.2f\r\n",
				p.frequency, p.corr_i, p.corr_q, p.error_i, p.error_q);
	}
	else
	{
		bladerf_strerror(status);
	}
#endif

#if 0	// TEST
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXLPF_I, 0x15);	// 
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXLPF_Q, 0x19);	// 
	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);

	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x08);
	lime_spi_blk_read(cmdcontext_spi,	0x50, &reg0x50_val);
	printf("   - RXLPF I : reg0x50_val=0x%.2x: \r\n", reg0x50_val);

	lime_spi_blk_read(cmdcontext_spi,	0x09, &reg0x09_val);
	lime_spi_blk_write(cmdcontext_spi,	0x09, reg0x09_val|(1<<4));

	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x08);
	lime_spi_blk_write(cmdcontext_spi,	0x52, 0x15);
	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x18);
	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x08);

	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x08);
	lime_spi_blk_read(cmdcontext_spi,	0x50, &reg0x50_val);
	printf("   - RXLPF I : reg0x50_val=0x%.2x: \r\n", reg0x50_val);
#endif

#if ENABLE_BLADERF_MODULE_TX_MANUAL
	/* Tx LO Leakage Calibration prerequisites */
	printf("*** LO leakage DAC Out registers  prerequisites ***\r\n");
	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);
	lime_spi_blk_read(cmdcontext_spi,	0x05, &reg0x05_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x05, reg0x05_val|(1<<3));
	lime_spi_blk_read(cmdcontext_spi,	0x09, &reg0x09_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x09, reg0x09_val|(1<<0));
	lime_spi_blk_read(cmdcontext_spi,	0x34, &reg0x34_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x34, reg0x34_val|(0x26<<0));
	lime_spi_blk_read(cmdcontext_spi,	0x44, &reg0x44_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x44, reg0x44_val|(2<<2));
	lime_spi_blk_read(cmdcontext_spi,	0x41, &reg0x41_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x41, reg0x41_val|(0x19<<0));
	lime_spi_blk_read(cmdcontext_spi,	0x45, &reg0x45_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x45, reg0x45_val|((15<<3)&0x78));

//	lms_read_registers(cmdcontext_spi);
	c	= 0x00;
//	lime_spi_blk_write(cmdcontext_spi,	0x42, 0x7D);	// force I : 0x7D
//	lime_spi_blk_write(cmdcontext_spi,	0x43, 0x85);	// force Q : 0x85

	while ( (c != 'q') && (c != 'n') && (c != 'N') )
	{
		printf("*** LO leakage DAC Out registers ***\r\n");
		lime_spi_blk_read(cmdcontext_spi,	0x42, &reg0x42_val);		/* Save Register */
		lime_spi_blk_read(cmdcontext_spi,	0x43, &reg0x43_val);		/* Save Register */
		printf("    - LO leakage DAC I Out = 0x%.2x\r\n", reg0x42_val);
		printf("    - LO leakage DAC Q Out = 0x%.2x\r\n", reg0x43_val);
		printf("Change LO leakage DAC I [y/N] : ");
		c	= getchar();
		printf("\r\n");
		while ( (c != 'N') && (c != 'n') && (c != 'q') )
		{
			printf("Change LO leakage DAC I [+/-]:");
			c	= getchar();
			printf("\r\n");
			if (c == '+')
			{
				lime_spi_blk_write(cmdcontext_spi,	0x42, (reg0x42_val+1));
			}
			else if (c == '-')
			{
				lime_spi_blk_write(cmdcontext_spi,	0x42, (reg0x42_val-1));
			}
			lime_spi_blk_read(cmdcontext_spi,	0x42, &reg0x42_val);		/* Save Register */
			lime_spi_blk_read(cmdcontext_spi,	0x43, &reg0x43_val);		/* Save Register */
			printf("    - LO leakage DAC I Out = 0x%.2x\r\n", reg0x42_val);
			printf("    - LO leakage DAC Q Out = 0x%.2x\r\n", reg0x43_val);
			printf("Change LO leakage DAC I [y/N] : ");
			c	= getchar();
			printf("\r\n");
		}
		printf("Change LO leakage DAC Q [y/N] : ");
		c	= getchar();
		printf("\r\n");
		while ( (c != 'N') && (c != 'n') && (c != 'q') )
		{
			printf("Change LO leakage DAC Q [+/-]:");
			c	= getchar();
			printf("\r\n");
			if (c == '+')
			{
				lime_spi_blk_write(cmdcontext_spi,	0x43, (reg0x43_val+1));
			}
			else if (c == '-')
			{
				lime_spi_blk_write(cmdcontext_spi,	0x43, (reg0x43_val-1));
			}
			lime_spi_blk_read(cmdcontext_spi,	0x42, &reg0x42_val);		/* Save Register */
			lime_spi_blk_read(cmdcontext_spi,	0x43, &reg0x43_val);		/* Save Register */
			printf("    - LO leakage DAC I Out = 0x%.2x\r\n", reg0x42_val);
			printf("    - LO leakage DAC Q Out = 0x%.2x\r\n", reg0x43_val);
			printf("Change LO leakage DAC Q [y/N]\r\n");
			c	= getchar();
		}
	}
#endif

	printf("\r* End of Calibration                                                   *\r\n");
	printf("\r************************************************************************\r\n");
#endif

#if 0	// TEST

	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_LPF_TUNING_MOD,	0x17);	// LPF Core		REG 0x35 0x55
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_LPF_BANDWIDTH_TUNING,			0x20);	// BW Tuning	REG 0x36 0x56
	lime_spi_blk_write(cmdcontext_spi,	0x36, 0x30);	// TX_RF_2	REG 0x42
	lime_spi_blk_write(cmdcontext_spi,	0x56, 0x30);	// TX_RF_3	REG 0x43

	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_TXLPF_I,			0x19);	// TX_LPF_0 I	REG 0x30 0x31 0x32 0x33
	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_TXLPF_Q,			0x29);	// TX_LPF_0 Q	REG 0x30 0x31 0x32 0x33

//	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXLPF_I,			0x15);	// TX_LPF_0 I	REG 0x50 0x51 0x52 0x53
//	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXLPF_Q,			0x01);	// TX_LPF_0 Q	REG 0x50 0x51 0x52 0x53

//	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXVGA2_DC,		0x19);	// RX_VGA2_0 DC	REG 0x60 0x61 0x62 0x63
//	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXVGA2_Ia,		0x15);	// RX_VGA2_0 Ia	REG 0x60 0x61 0x62 0x63
//	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXVGA2_Qa,		0x21);	// RX_VGA2_0 Qa	REG 0x60 0x61 0x62 0x63
//	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXVGA2_Ib,		0x21);	// RX_VGA2_0 Ib	REG 0x60 0x61 0x62 0x63
//	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_RXVGA2_Qb,		0x25);	// RX_VGA2_0 Qb	REG 0x60 0x61 0x62 0x63

//	lime_spi_blk_write(cmdcontext_spi,	0x71, 0x80);	// RX_FE_1	REG 0x71
//	lime_spi_blk_write(cmdcontext_spi,	0x72, 0x83);	// RX_FE_2	REG 0x72

	lime_spi_blk_write(cmdcontext_spi,	0x42, 0x77);	// TX_RF_2	REG 0x42
	lime_spi_blk_write(cmdcontext_spi,	0x43, 0x79);	// TX_RF_3	REG 0x43
#endif

//	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_TXLPF_I,			0x19);	// TX_LPF_0 I	REG 0x30 0x31 0x32 0x33
//	lime_cfg_blk_write(cmdcontext_spi, PIO_CFG_CALIB_DC_OFF_CANCEL_TXLPF_Q,			0x29);	// TX_LPF_0 Q	REG 0x30 0x31 0x32 0x33

	lime_spi_blk_write(cmdcontext_spi,	0x09, 0x45);	// EURECOM value 0xC5, DEFAULT value 0x40
//	lime_spi_blk_write(cmdcontext_spi,	0x09, 0xC5);	// EURECOM value 0xC5, DEFAULT value 0x40
	lime_spi_blk_write(cmdcontext_spi,	0x5A, 0x30);	// EURECOM value 0x30, DEFAULT value 0x20
	lime_spi_blk_write(cmdcontext_spi,	0x75, 0xE0);	// EURECOM value 0xE0 (LNA2), DEFAULT value 0xD0 (LNA1)

	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);

	usleep(1000);
	syrtem_spi_finictxt(cmdcontext_spi);
	openair0_close();
	system("/bin/stty cooked");
	return;
}
