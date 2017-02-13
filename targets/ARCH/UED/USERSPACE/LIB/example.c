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

#define MAX_LOOP	10
#define TIMEOUT		100

extern uint32_t	leon3_spiwrite_intr_cnt;
extern uint32_t	leon3_call_exec_cnt;
extern uint32_t	leon3_intr_exec_cnt;



void lime_spi_blk_write(void *spictxt, unsigned char reg, unsigned char data)
{
	int					counter	= 0;
	syr_pio_cmdretval_e	status	= SYR_PIO_CMDIDLE;

//	printf("lime_spi_blk_write(reg=0x%.2x, data=0x%.2x)\r\n", reg, data);
	if (syrtem_spi_write(spictxt, reg, data) == SYR_PIO_CMDSEND)
	{
		usleep(1000);
		counter	= 0;
		status	= syrtem_spi_chkstatus();
		while ( (status != SYR_PIO_CMDIDLE) && (status != SYR_PIO_CMDRECV) && (counter < TIMEOUT) )
		{
			counter++;
			usleep(1000);
			status	= syrtem_spi_chkstatus();
		}
		if (counter >= TIMEOUT)
		{
			printf("lime_spi_blk_write SYR_PIO_CMDIDLE counter >= TIMEOUT\r\n");
		}
		else
		{
#if 0
//			printf("status=%d\r\n", status);
			counter		= 0;
			(*pdata)	= 0xFF;
			getdata		= syrtem_spi_getdata(spictxt, pdata);
			while ( (getdata != SYR_PIO_CMDRECV) && (counter < TIMEOUT) )
			{
				counter++;
				usleep(1000);
				getdata		= syrtem_spi_getdata(spictxt, pdata);
			}
			if (counter >= TIMEOUT)
			{
				printf("lime_spi_blk_write SYR_PIO_CMDRECV counter >= TIMEOUT\r\n");
			}
			else
			{
			}
#endif
		}
	}
	else
	{
		printf("syrtem_spi_write 1 REG() FAILED\r\n");
	}
	usleep(1000);
	return;
}

void lime_spi_blk_read(void *spictxt, unsigned char reg, unsigned char *pdata)
{
	int					i;
	int					counter	= 0;
	syr_pio_cmdretval_e	status	= SYR_PIO_CMDIDLE;
	syr_pio_cmdretval_e	getdata	= SYR_PIO_CMDIDLE;

//	printf("lime_spi_blk_read(reg=0x%.2x)\r\n", reg);
	if (syrtem_spi_read(spictxt, reg) == SYR_PIO_CMDSEND)
	{
		usleep(1000);
		counter	= 0;
		status	= syrtem_spi_chkstatus();
		while ( (status != SYR_PIO_CMDIDLE) && (status != SYR_PIO_CMDRECV) && (counter < TIMEOUT) )
		{
			counter++;
			usleep(1000);
			status	= syrtem_spi_chkstatus();
		}
		if (counter >= TIMEOUT)
		{
			printf("lime_spi_blk_read SYR_PIO_CMDIDLE counter >= TIMEOUT\r\n");
		}
		else
		{
//			printf("status=%d\r\n", status);
			counter		= 0;
			(*pdata)	= 0xFF;
			getdata		= syrtem_spi_getdata(spictxt, pdata);
			while ( (getdata != SYR_PIO_CMDRECV) && (counter < TIMEOUT) )
			{
				counter++;
				usleep(1000);
				getdata		= syrtem_spi_getdata(spictxt, pdata);
			}
			if (counter >= TIMEOUT)
			{
				printf("lime_spi_blk_read SYR_PIO_CMDRECV counter >= TIMEOUT\r\n");
				printf("getdata=%d, pdata=0x%.4x\r\n", getdata, (*pdata));
				printf("spictxt->spi_tid=%d\r\n", ((ioctl_arg_spi_t *)spictxt)->spi_tid);
			}
			else
			{
			}
		}
	}
	else
	{
		printf("lime_spi_blk_read(cmdcontext, reg) FAILED\r\n");
	}
	usleep(1000);
	return;
}

void lms_read_registers(void *cmdcontext_spi)
{
	int					index			= 0;
	unsigned int		device			= 0;
	unsigned char		data			= 0x00;

	printf("lms_read_registers(CardId:%d);\r\n", device);
	printf("Top level Configuration :\r\n");
	for (index = 0; index < 16; index++)
	{
		lime_spi_blk_read(cmdcontext_spi, index, &data);
		printf("   Reg[0x%.2x]=0x%.2x\r\n", index, data);
	}
	printf("TX PLL Configuration :\r\n");
	for (index = 16; index < 32; index++)
	{
		lime_spi_blk_read(cmdcontext_spi, index, &data);
		printf("   Reg[0x%.2x]=0x%.2x\r\n", index, data);
	}
	printf("RX PLL Configuration :\r\n");
	for (index = 32; index < 48; index++)
	{
		lime_spi_blk_read(cmdcontext_spi, index, &data);
		printf("   Reg[0x%.2x]=0x%.2x\r\n", index, data);
	}
	printf("TX LPF Modules Configuration :\r\n");
	for (index = 48; index < 64; index++)
	{
		lime_spi_blk_read(cmdcontext_spi, index, &data);
		printf("   Reg[0x%.2x]=0x%.2x\r\n", index, data);
	}
	printf("TX RF Modules Configuration :\r\n");
	for (index = 64; index < 80; index++)
	{
		lime_spi_blk_read(cmdcontext_spi, index, &data);
		printf("   Reg[0x%.2x]=0x%.2x\r\n", index, data);
	}
	printf("RX LPF, ADC and DAC Modules Configuration :\r\n");
	for (index = 80; index < 96; index++)
	{
		lime_spi_blk_read(cmdcontext_spi, index, &data);
		printf("   Reg[0x%.2x]=0x%.2x\r\n", index, data);
	}
	printf("RX VGA2 Configuration :\r\n");
	for (index = 96; index < 112; index++)
	{
		lime_spi_blk_read(cmdcontext_spi, index, &data);
		printf("   Reg[0x%.2x]=0x%.2x\r\n", index, data);
	}
	printf("RX FE Modules Configuration :\r\n");
	for (index = 112; index < 128; index++)
	{
		lime_spi_blk_read(cmdcontext_spi, index, &data);
		printf("   Reg[0x%.2x]=0x%.2x\r\n", index, data);
	}
	printf("\r\n");
	return;
}

void lms_read_cal_registers(void *cmdcontext_spi)
{
	int					index			= 0;
	unsigned int		device			= 0;
	unsigned char		data			= 0x00;

	printf("lms_read_cal_registers(CardId:%d);\r\n", device);
	printf("DC Offset Calibration of LPF Tuning Module :\r\n");
	lime_spi_blk_read(cmdcontext_spi, 0x55, &data);
	printf("RX_LPF_AD_DA_5   = 0x%.2x\r\n", data);
	lime_spi_blk_read(cmdcontext_spi, 0x35, &data);
	printf("TX_LPF_5         = 0x%.2x\r\n", data);

	printf("LPF Bandwidth Tuning :\r\n");
	lime_spi_blk_read(cmdcontext_spi, 0x56, &data);
	printf("RX_LPF_AD_DA_6   = 0x%.2x\r\n", data);
	lime_spi_blk_read(cmdcontext_spi, 0x36, &data);
	printf("TX_LPF_6         = 0x%.2x\r\n", data);

	printf("TX LPF DC Offset Calibration :\r\n");
	lime_spi_blk_write(cmdcontext_spi,	0x33, 0x08 );
	lime_spi_blk_read(cmdcontext_spi, 0x30, &data);
	printf("TX_LPF_0 I       = 0x%.2x\r\n", data);
	lime_spi_blk_write(cmdcontext_spi,	0x33, 0x09 );
	lime_spi_blk_read(cmdcontext_spi, 0x30, &data);
	printf("TX_LPF_0 Q       = 0x%.2x\r\n", data);

	lime_spi_blk_read(cmdcontext_spi, 0x19, &data);
	printf("TX VCOCAP        = 0x%.2x\r\n", data);
	lime_spi_blk_read(cmdcontext_spi, 0x29, &data);
	printf("RX VCOCAP        = 0x%.2x\r\n", data);

	printf("RX LPF ADC and DAC registers configuration :\r\n");
	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x08 );
	lime_spi_blk_read(cmdcontext_spi, 0x50, &data);
	printf("RX_LPF_AD_DA_0 I = 0x%.2x\r\n", data);
	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x09 );
	lime_spi_blk_read(cmdcontext_spi, 0x50, &data);
	printf("RX_LPF_AD_DA_0 Q = 0x%.2x\r\n", data);

	printf("RXVGA2 DC Offset Calibration :\r\n");
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x08 );
	lime_spi_blk_read(cmdcontext_spi, 0x60, &data);
	printf("RX_VGA2_0 DC     = 0x%.2x\r\n", data);
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x09 );
	lime_spi_blk_read(cmdcontext_spi, 0x60, &data);
	printf("RX_VGA2_0 Ia     = 0x%.2x\r\n", data);
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0A );
	lime_spi_blk_read(cmdcontext_spi, 0x60, &data);
	printf("RX_VGA2_0 Qa     = 0x%.2x\r\n", data);
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0B );
	lime_spi_blk_read(cmdcontext_spi, 0x60, &data);
	printf("RX_VGA2_0 Ib     = 0x%.2x\r\n", data);
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0C );
	lime_spi_blk_read(cmdcontext_spi, 0x60, &data);
	printf("RX_VGA2_0 Qb     = 0x%.2x\r\n", data);

	printf("TX LO leakage cancellation :\r\n");
	lime_spi_blk_read(cmdcontext_spi, 0x42, &data);
	printf("TX_RF_2          = 0x%.2x\r\n", data);
	lime_spi_blk_read(cmdcontext_spi, 0x43, &data);
	printf("TX_RF_3          = 0x%.2x\r\n", data);

	printf("\r\n");
	return;
}

void lime_cfg_blk_write(void *spictxt, unsigned char pid, unsigned char pval)
{
	int					counter	= 0;
	syr_pio_cmdretval_e	status	= SYR_PIO_CMDIDLE;

//	printf("lime_cfg_blk_write(reg=0x%.2x, data=0x%.2x)\r\n", reg, data);
	if (syrtem_cfg_write(spictxt, pid, pval) == SYR_PIO_CMDSEND)
	{
		usleep(1000);
		counter	= 0;
		status	= syrtem_cfg_chkstatus();
		while ( (status != SYR_PIO_CMDIDLE) && (status != SYR_PIO_CMDRECV) && (counter < TIMEOUT) )
		{
			counter++;
			usleep(1000);
			status	= syrtem_cfg_chkstatus();
		}
		if (counter >= TIMEOUT)
		{
			printf("lime_cfg_blk_write SYR_PIO_CMDIDLE counter >= TIMEOUT\r\n");
		}
		else
		{
#if 0
//			printf("status=%d\r\n", status);
			counter		= 0;
			(*pdata)	= 0xFF;
			getdata		= syrtem_spi_getdata(spictxt, pdata);
			while ( (getdata != SYR_PIO_CMDRECV) && (counter < TIMEOUT) )
			{
				counter++;
				usleep(1000);
				getdata		= syrtem_spi_getdata(spictxt, pdata);
			}
			if (counter >= TIMEOUT)
			{
				printf("lime_spi_blk_write SYR_PIO_CMDRECV counter >= TIMEOUT\r\n");
			}
			else
			{
			}
#endif
		}
	}
	else
	{
		printf("syrtem_cfg_write 1 REG() FAILED\r\n");
	}
	usleep(1000);
	return;
}

unsigned char	tunevcocap_tx(void *cmdcontext_spi)
{
	unsigned char	retval			= 0xFF;
	unsigned char	reg0x19_val		= 0;
	unsigned char	reg0x1A_val		= 0;
	int				i				= 0;
	unsigned char	vtune			= 0xFF;
	unsigned char	tunevcocap_min	= 0xFF;
	unsigned char	tunevcocap_max	= 0x3F;
	unsigned char	tunevcocap_res	= 0xFF;
	unsigned char	min_is_done		= 0;
	lime_spi_blk_read(cmdcontext_spi,	0x19, &reg0x19_val);
	for (i = 0; i < 64; i++)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x19, (reg0x19_val&0xC0)|i );
		lime_spi_blk_read(cmdcontext_spi,	0x1A, &reg0x1A_val);
		vtune	= (reg0x1A_val>>6)&0x03;
//		printf("i=%d vtune=0x%x reg0x1A_val=0x%.2x\r\n", i, vtune, reg0x1A_val);
		if ( (vtune == 0) && (!min_is_done) )	// pass 0b10 to 0b00
		{
			tunevcocap_min	= i;
			min_is_done		= 1;
		}
		else if (vtune == 1)	// pass 0b00 to 0b01
		{
			tunevcocap_max	= i;
			break;
		}
	}
	tunevcocap_res	= (tunevcocap_max + tunevcocap_min) >> 1;	// (max+min) / 2
	printf("tunevcocap_tx_min=%d\r\n", tunevcocap_min);
	printf("tunevcocap_tx_max=%d\r\n", tunevcocap_max);
	printf("tunevcocap_tx_res=%d\r\n", tunevcocap_res);
	lime_spi_blk_write(cmdcontext_spi,	0x19, (reg0x19_val&0xC0)|tunevcocap_res );
	return tunevcocap_res;
}

unsigned char	tunevcocap_rx(void *cmdcontext_spi)
{
	unsigned char	retval			= 0xFF;
	unsigned char	reg0x29_val		= 0;
	unsigned char	reg0x2A_val		= 0;
	int				i				= 0;
	unsigned char	vtune			= 0xFF;
	unsigned char	tunevcocap_min	= 0xFF;
	unsigned char	tunevcocap_max	= 0x3F;
	unsigned char	tunevcocap_res	= 0xFF;
	unsigned char	min_is_done		= 0;
	lime_spi_blk_read(cmdcontext_spi,	0x29, &reg0x29_val);
	for (i = 0; i < 64; i++)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x29, (reg0x29_val&0xC0)|i );
		lime_spi_blk_read(cmdcontext_spi,	0x2A, &reg0x2A_val);
		vtune	= (reg0x2A_val>>6)&0x03;
//		printf("i=%d vtune=0x%x reg0x2A_val=0x%.2x\r\n", i, vtune, reg0x2A_val);
		if ( (vtune == 0) && (!min_is_done) )	// pass 0b10 to 0b00
		{
			tunevcocap_min	= i;
			min_is_done		= 1;
		}
		else if (vtune == 1)	// pass 0b00 to 0b01
		{
			tunevcocap_max	= i;
			break;
		}
	}
	tunevcocap_res	= (tunevcocap_max + tunevcocap_min) >> 1;	// (max+min) / 2
	printf("tunevcocap_rx_min=%d\r\n", tunevcocap_min);
	printf("tunevcocap_rx_max=%d\r\n", tunevcocap_max);
	printf("tunevcocap_rx_res=%d\r\n", tunevcocap_res);
	lime_spi_blk_write(cmdcontext_spi,	0x29, (reg0x29_val&0xC0)|tunevcocap_res );
	return tunevcocap_res;
}

void main(void)
{
	int				i				= 0;
	int				retval			= 0;
	void			*cmdcontext_spi	= NULL;
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
	printf("Openair0 opened\r\n");

	cmdcontext_spi	=	syrtem_spi_initctxt();

#if 1
#if 1	/* DC Offset Calibration of LPF Tuning Module (LPF Core) */
	printf("*** DC Offset Calibration of LPF Tuning Module **********\r\n");
	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);
	lime_spi_blk_read(cmdcontext_spi,	0x09, &reg0x09_val);			/* Save TopSPI::CLK_EN[5] Register */
	clk_en5	= reg0x09_val&0x20;
	lime_spi_blk_write(cmdcontext_spi,	0x09, reg0x09_val|(1<<5));		/* TopSPI::CLK_EN[5] := 1 */
	/*    - Perform DC Calibration Procedure in TopSPI with ADDR := 0 and get Result */
	lime_spi_blk_read(cmdcontext_spi,	0x03, &reg0x03_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x03, reg0x03_val&0xF8);		/* DC_ADDR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x03, &reg0x03_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x03, reg0x03_val|(1<<5));		/* DC_START_CLBR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x03, &reg0x03_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x03, reg0x03_val&0xDF);		/* DC_START_CLBR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x00, &reg0x00_val);			/* Read DC_REG_VAL */
	if ( (reg0x00_val&0x3F) == 31)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x00, reg0x00_val&0xC0);	/* Set DC_REG_VAL := 0 */
		lime_spi_blk_read(cmdcontext_spi,	0x03, &reg0x03_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x03, reg0x03_val|(1<<5));	/* DC_START_CLBR := 1 */
		lime_spi_blk_read(cmdcontext_spi,	0x03, &reg0x03_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x03, reg0x03_val&0xDF);	/* DC_START_CLBR := 0 */
		lime_spi_blk_read(cmdcontext_spi,	0x00, &reg0x00_val);		/* Read DC_REG_VAL */
		if ( (reg0x00_val&0x3F) == 0)
		{
			printf("PANIC: Algorithm does not converge!\r\n");
			syrtem_spi_finictxt(cmdcontext_spi);
			openair0_close();
			system("/bin/stty cooked");
			return;
		}
	}
	dccal	= reg0x00_val&0x3F;												/* DCCAL := TopSPI::DC_REGVAL */
	printf("dccal     = 0x%.2x\r\n", dccal);
	lime_spi_blk_read(cmdcontext_spi,	0x55, &reg0x55_val);				/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x55, (reg0x55_val&0xC0)|dccal);	/* RxLPFSPI::DCO_DACCAL := DCCAL */
	lime_spi_blk_read(cmdcontext_spi,	0x35, &reg0x55_val);				/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x35, (reg0x55_val&0xC0)|dccal);	/* TxLPFSPI::DCO_DACCAL := DCCAL */
	lime_spi_blk_read(cmdcontext_spi,	0x09, &reg0x09_val);				/* Restore TopSPI::CLK_EN[5] Register */
	lime_spi_blk_write(cmdcontext_spi,	0x09, (reg0x09_val&0xDF)|clk_en5);	/* TopSPI::CLK_EN[5] := 1 */
#endif

#if 0	/* LPF Bandwidth Tuning */
	printf("LPF Bandwidth Tuning\r\n");
	/* PLL Reference Clock Frequency == 40MHz? */
	lime_spi_blk_read(cmdcontext_spi,	0x05, &reg0x05_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x05, reg0x05_val|(1<<1));	/* SPI Port Mode 4 wires */
	lime_spi_blk_read(cmdcontext_spi,	0x05, &reg0x05_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x05, reg0x05_val|(1<<3));	/* Soft Tx enabled */
	/* PLL Reference Clock Frequency == 40MHz? */
//	lime_spi_blk_read(cmdcontext_spi,	0x44, &reg0x44_val);		/* Save Register */
//	lime_spi_blk_write(cmdcontext_spi,	0x44, reg0x44_val|(1<<2));	/* Power Down TxVGA2 (Optional) */
	lime_spi_blk_read(cmdcontext_spi,	0x14, &reg0x14_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x14, reg0x14_val|(1<<3));	/* Enable TxPLL and set to Produce 320MHz */
//	printf("LPF Bandwidth Tuning step 1\r\n");
	usleep(1000);
	
	/* 320 MHz */
	lime_spi_blk_read(cmdcontext_spi,	0x15, &reg0x15_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x15, (reg0x15_val&0xE3)|(7<<2));	/* Enable TxPLL and set to Produce 320MHz */
	lime_spi_blk_read(cmdcontext_spi,	0x10, &reg0x10_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x10, 0x53);
	lime_spi_blk_read(cmdcontext_spi,	0x11, &reg0x11_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x11, 0x55);
	lime_spi_blk_read(cmdcontext_spi,	0x12, &reg0x12_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x12, 0x55);
	lime_spi_blk_read(cmdcontext_spi,	0x13, &reg0x13_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x13, 0x55);
	lime_spi_blk_read(cmdcontext_spi,	0x15, &reg0x15_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x15, (reg0x15_val&0x1F)|(5<<5));

//	lms_read_registers(cmdcontext_spi);

	tunevcocap_tx(cmdcontext_spi);
//	printf("LPF Bandwidth Tuning step 2\r\n");
	usleep(1000);

	lime_spi_blk_read(cmdcontext_spi,	0x06, &reg0x06_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x06, reg0x06_val&~(1<<3));	/* Use 40MHz generated From TxPLL: TopSPI::CLKSEL_LPFCAL := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x06, &reg0x06_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x06, reg0x06_val&~(1<<2));	/* Power Up LPF tuning clock generation block: TopSPI::PD_CLKLPFCAL := 0 */
//	printf("LPF Bandwidth Tuning step 3\r\n");
	usleep(1000);

	lime_spi_blk_read(cmdcontext_spi,	0x07, &reg0x07_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x07, (reg0x07_val&0xF0)|0x09);	/* TopSPI::BWC_LPFCAL := 0x7 (2.5MHz, as example) */
	lime_spi_blk_read(cmdcontext_spi,	0x07, &reg0x07_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x07, reg0x07_val|(1<<7));		/* TopSPI::EN_CAL_LPFCAL := 1 (Enable) */
//	printf("LPF Bandwidth Tuning step 4\r\n");
	usleep(1000);

	lime_spi_blk_read(cmdcontext_spi,	0x06, &reg0x06_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x06, reg0x06_val|0x01);	/* TopSPI::RST_CAL_LPFCAL := 1 (Rst Active) */
	lime_spi_blk_read(cmdcontext_spi,	0x06, &reg0x06_val);		/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x06, reg0x06_val&~(1<<0));	/* TopSPI::RST_CAL_LPFCAL := 0 (Rst Inactive) */
//	printf("LPF Bandwidth Tuning step 5\r\n");
	usleep(1000);

	lime_spi_blk_read(cmdcontext_spi,	0x01, &reg0x01_val);		/* Save Register */
	rccal_lpf	= ((reg0x01_val&0xE0)>>5)&0x07;						/* RCCAL := TopSPI::RCCAL_LPFCAL */
	printf("rccal_lpf = 0x%.2x, reg0x01_val=0x%.2x\r\n", rccal_lpf, reg0x01_val);
	lime_spi_blk_read(cmdcontext_spi,	0x56, &reg0x56_val);				/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x56, (reg0x56_val&0x8F)|((rccal_lpf<<4)&0x70));	/* RxLPFSPI::RCCAL_LPF := RCCAL */
	lime_spi_blk_read(cmdcontext_spi,	0x36, &reg0x36_val);				/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x36, (reg0x36_val&0x8F)|((rccal_lpf<<4)&0x70));	/* TxLPFSPI::RCCAL_LPF := RCCAL */
//	printf("LPF Bandwidth Tuning step 6\r\n");
	usleep(1000);

	lime_spi_blk_write(cmdcontext_spi,	0x10, reg0x10_val);
	lime_spi_blk_write(cmdcontext_spi,	0x11, reg0x11_val);
	lime_spi_blk_write(cmdcontext_spi,	0x12, reg0x12_val);
	lime_spi_blk_write(cmdcontext_spi,	0x13, reg0x13_val);
	tunevcocap_tx(cmdcontext_spi);

//	lms_read_registers(cmdcontext_spi);
#endif

//	lime_spi_blk_read(cmdcontext_spi,	0x56, &reg0x56_val);				/* Save Register */
//	lime_spi_blk_write(cmdcontext_spi,	0x56, (reg0x56_val&0x8F)|((2<<4)&0x70));	/* RxLPFSPI::RCCAL_LPF := RCCAL */
//	lime_spi_blk_read(cmdcontext_spi,	0x36, &reg0x36_val);				/* Save Register */
//	lime_spi_blk_write(cmdcontext_spi,	0x36, (reg0x36_val&0x8F)|((2<<4)&0x70));	/* TxLPFSPI::RCCAL_LPF := RCCAL */

//	tunevcocap_tx(cmdcontext_spi);
//	tunevcocap_rx(cmdcontext_spi);

#if 1	/* TX LPF DC Offset Calibration */
	printf("*** TX LPF DC Offset Calibration **********\r\n");
//	lime_spi_blk_write(cmdcontext_spi,	0x06, 0x0C);		/* TopSPI::CLK_EN[1] := 1 Tx */
//	lime_spi_blk_write(cmdcontext_spi,	0x09, 0xC5);		/* TopSPI::CLK_EN[1] := 1 Tx */
//	lime_spi_blk_write(cmdcontext_spi,	0x0B, 0x09);		/* TopSPI::CLK_EN[1] := 1 Tx */

//	lime_spi_blk_write(cmdcontext_spi,	0x36, 0x20);		/* TopSPI::CLK_EN[1] := 1 Tx */
//	lime_spi_blk_write(cmdcontext_spi,	0x56, 0x20);		/* TopSPI::CLK_EN[1] := 1 Tx */

//	lime_spi_blk_write(cmdcontext_spi,	0x3F, 0x00);		/* TopSPI::CLK_EN[1] := 1 Tx */

	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);

	lime_spi_blk_read(cmdcontext_spi,	0x09, &reg0x09_val);			/* Save TopSPI::CLK_EN Register */
	clk_en	= reg0x09_val;
	lime_spi_blk_write(cmdcontext_spi,	0x09, reg0x09_val|(1<<1));		/* TopSPI::CLK_EN[1] := 1 Tx */

	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);

	/*    - Perform DC Calibration Procedure in LPFSPI with ADDR := 0 (For channel I) and get Result */
	lime_spi_blk_write(cmdcontext_spi,	0x33, 0x08);		/* RESET REGISTER */
	lime_spi_blk_write(cmdcontext_spi,	0x32, 0x1F);		/* DC_CNT_VAL = 0x1F */
	lime_spi_blk_write(cmdcontext_spi,	0x33, 0x18);		/* START LOAD into DC_REG_VAL 0x30 */
	lime_spi_blk_write(cmdcontext_spi,	0x33, 0x08);		/* STOP  LOAD into DC_REG_VAL 0x30 */

	lime_spi_blk_read(cmdcontext_spi,	0x33, &reg0x33_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x33, reg0x33_val&0xF8);		/* DC_ADDR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x33, &reg0x33_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x33, reg0x33_val|(1<<5));		/* DC_START_CLBR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x33, &reg0x33_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x33, reg0x33_val&0xDF);		/* DC_START_CLBR := 0 */

	lime_spi_blk_read(cmdcontext_spi,	0x30, &reg0x30_val);			/* Read DC_REG_VAL */
	if ( (reg0x30_val&0x3F) == 31)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x33, 0x08);		/* RESET REGISTER */
		lime_spi_blk_write(cmdcontext_spi,	0x32, 0x00);		/* DC_CNT_VAL = 0x00 */
		lime_spi_blk_write(cmdcontext_spi,	0x33, 0x18);		/* START LOAD into DC_REG_VAL 0x30 */
		lime_spi_blk_write(cmdcontext_spi,	0x33, 0x08);		/* STOP  LOAD into DC_REG_VAL 0x30 */

		lime_spi_blk_read(cmdcontext_spi,	0x33, &reg0x33_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x33, reg0x33_val|(1<<5));	/* DC_START_CLBR := 1 */
		lime_spi_blk_read(cmdcontext_spi,	0x33, &reg0x33_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x33, reg0x33_val&0xDF);	/* DC_START_CLBR := 0 */
		lime_spi_blk_read(cmdcontext_spi,	0x30, &reg0x30_val);		/* Read DC_REG_VAL */
		if ( (reg0x30_val&0x3F) == 0)
		{
			printf("PANIC: Algorithm does not converge!\r\n");
			lime_spi_blk_write(cmdcontext_spi,	0x09, clk_en);	/* TopSPI::CLK_EN[5] := 1 */
			syrtem_spi_finictxt(cmdcontext_spi);
			openair0_close();
			system("/bin/stty cooked");
			return;
		}
	}
	printf("   - I : reg0x30_val=0x%.2x: \r\n", reg0x30_val&0x3F);
	/*    - Perform DC Calibration Procedure in LPFSPI with ADDR := 1 (For channel Q) and get Result */
	lime_spi_blk_write(cmdcontext_spi,	0x33, 0x09);		/* RESET REGISTER */
	lime_spi_blk_write(cmdcontext_spi,	0x32, 0x1F);		/* DC_CNT_VAL = 0x1F */
	lime_spi_blk_write(cmdcontext_spi,	0x33, 0x19);		/* START LOAD into DC_REG_VAL 0x30 */
	lime_spi_blk_write(cmdcontext_spi,	0x33, 0x09);		/* STOP  LOAD into DC_REG_VAL 0x30 */

	lime_spi_blk_read(cmdcontext_spi,	0x33, &reg0x33_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x33, (reg0x33_val&0xF8)|(1<<0));	/* DC_ADDR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x33, &reg0x33_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x33, reg0x33_val|(1<<5));		/* DC_START_CLBR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x33, &reg0x33_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x33, reg0x33_val&0xDF);		/* DC_START_CLBR := 0 */

	lime_spi_blk_read(cmdcontext_spi,	0x30, &reg0x30_val);			/* Read DC_REG_VAL */
	if ( (reg0x30_val&0x3F) == 31)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x33, 0x09);		/* RESET REGISTER */
		lime_spi_blk_write(cmdcontext_spi,	0x32, 0x00);		/* DC_CNT_VAL = 0x00 */
		lime_spi_blk_write(cmdcontext_spi,	0x33, 0x19);		/* START LOAD into DC_REG_VAL 0x30 */
		lime_spi_blk_write(cmdcontext_spi,	0x33, 0x09);		/* STOP  LOAD into DC_REG_VAL 0x30 */

		lime_spi_blk_read(cmdcontext_spi,	0x33, &reg0x33_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x33, reg0x33_val|(1<<5));	/* DC_START_CLBR := 1 */
		lime_spi_blk_read(cmdcontext_spi,	0x33, &reg0x33_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x33, reg0x33_val&0xDF);	/* DC_START_CLBR := 0 */
		lime_spi_blk_read(cmdcontext_spi,	0x30, &reg0x30_val);		/* Read DC_REG_VAL */
		if ( (reg0x30_val&0x3F) == 0)
		{
			printf("PANIC: Algorithm does not converge!\r\n");
			lime_spi_blk_write(cmdcontext_spi,	0x09, clk_en);	/* TopSPI::CLK_EN[5] := 1 */
			syrtem_spi_finictxt(cmdcontext_spi);
			openair0_close();
			system("/bin/stty cooked");
			return;
		}
	}
	printf("   - Q : reg0x30_val=0x%.2x: \r\n", reg0x30_val&0x3F);
	lime_spi_blk_write(cmdcontext_spi,	0x09, clk_en);	/* TopSPI::CLK_EN[5] := 1 */
#endif

#if 1
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
	lime_spi_blk_write(cmdcontext_spi,	0x42, 0x7D);	// force I : 0x7D
	lime_spi_blk_write(cmdcontext_spi,	0x43, 0x85);	// force Q : 0x85

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

#if 1
	/* RX LPF DC Offset Calibration */
	printf("*** RX LPF DC Offset Calibration **********\r\n");
//	lime_spi_blk_write(cmdcontext_spi,	0x05, 0x3E);
//	lime_spi_blk_write(cmdcontext_spi,	0x09, 0xCD);
//	lime_spi_blk_write(cmdcontext_spi,	0x19, 0xA9);
//	lime_spi_blk_write(cmdcontext_spi,	0x29, 0xB3);
//	lime_spi_blk_write(cmdcontext_spi,	0x45, 0xF8);
//	lms_read_registers(cmdcontext_spi);
//	lms_read_cal_registers(cmdcontext_spi);
	lime_spi_blk_read(cmdcontext_spi,	0x09, &reg0x09_val);			/* Save TopSPI::CLK_EN Register */
	clk_en	= reg0x09_val;
	lime_spi_blk_write(cmdcontext_spi,	0x09, reg0x09_val|0x08);		/* TopSPI::CLK_EN[3] := 1 Rx */
	lime_spi_blk_write(cmdcontext_spi,	0x76, 0x78);					/* RFB_TIA_RXFE[6:0]: Feedback resistor control of the TIA (RXVGA1) to set the mixer gain. */
																		/* If =120 --> mixer gain = 30dB (default) */
	lime_spi_blk_write(cmdcontext_spi,	0x65, 0x0A);					/* VGA2GAIN[4:0]: RXVGA2 gain control = 30dB */

	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);

	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x08);		/* RESET REGISTER */
	lime_spi_blk_write(cmdcontext_spi,	0x52, 0x1F);		/* DC_CNT_VAL = 0x1F */
	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x18);		/* START LOAD into DC_REG_VAL 0x50 */
	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x08);		/* STOP  LOAD into DC_REG_VAL 0x50 */

	lime_spi_blk_read(cmdcontext_spi,	0x53, &reg0x53_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x53, reg0x53_val&0xF8);		/* DC_ADDR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x53, &reg0x53_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x53, reg0x53_val|(1<<5));		/* DC_START_CLBR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x53, &reg0x53_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x53, reg0x53_val&0xDF);		/* DC_START_CLBR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x50, &reg0x50_val);			/* Read DC_REG_VAL */
	if ( (reg0x50_val&0x3F) == 31)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x53, 0x08);		/* RESET REGISTER */
		lime_spi_blk_write(cmdcontext_spi,	0x52, 0x00);		/* DC_CNT_VAL = 0x00 */
		lime_spi_blk_write(cmdcontext_spi,	0x53, 0x18);		/* START LOAD into DC_REG_VAL 0x50 */
		lime_spi_blk_write(cmdcontext_spi,	0x53, 0x08);		/* STOP  LOAD into DC_REG_VAL 0x50 */

		lime_spi_blk_read(cmdcontext_spi,	0x53, &reg0x53_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x53, reg0x53_val|(1<<5));	/* DC_START_CLBR := 1 */
		lime_spi_blk_read(cmdcontext_spi,	0x53, &reg0x53_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x53, reg0x53_val&0xDF);	/* DC_START_CLBR := 0 */
		lime_spi_blk_read(cmdcontext_spi,	0x50, &reg0x50_val);		/* Read DC_REG_VAL */
		if ( (reg0x50_val&0x3F) == 0)
		{
			printf("PANIC: Algorithm does not converge!\r\n");
			syrtem_spi_finictxt(cmdcontext_spi);
			openair0_close();
			system("/bin/stty cooked");
			return;
		}
	}
	printf("   - I : reg0x50_val=0x%.2x: \r\n", reg0x50_val&0x3F);
	/*    - Perform DC Calibration Procedure in LPFSPI with ADDR := 1 (For channel Q) and get Result */
	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x09);		/* RESET REGISTER */
	lime_spi_blk_write(cmdcontext_spi,	0x52, 0x1F);		/* DC_CNT_VAL = 0x1F */
	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x19);		/* START LOAD into DC_REG_VAL 0x50 */
	lime_spi_blk_write(cmdcontext_spi,	0x53, 0x09);		/* STOP  LOAD into DC_REG_VAL 0x50 */

	lime_spi_blk_read(cmdcontext_spi,	0x53, &reg0x53_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x53, (reg0x53_val&0xF8)|(1<<0));	/* DC_ADDR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x53, &reg0x53_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x53, reg0x53_val|(1<<5));		/* DC_START_CLBR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x53, &reg0x53_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x53, reg0x53_val&0xDF);		/* DC_START_CLBR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x50, &reg0x50_val);			/* Read DC_REG_VAL */
	if ( (reg0x50_val&0x3F) == 31)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x53, 0x09);		/* RESET REGISTER */
		lime_spi_blk_write(cmdcontext_spi,	0x52, 0x00);		/* DC_CNT_VAL = 0x00 */
		lime_spi_blk_write(cmdcontext_spi,	0x53, 0x19);		/* START LOAD into DC_REG_VAL 0x50 */
		lime_spi_blk_write(cmdcontext_spi,	0x53, 0x09);		/* STOP  LOAD into DC_REG_VAL 0x50 */

		lime_spi_blk_read(cmdcontext_spi,	0x53, &reg0x53_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x53, reg0x53_val|(1<<5));	/* DC_START_CLBR := 1 */
		lime_spi_blk_read(cmdcontext_spi,	0x53, &reg0x53_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x53, reg0x53_val&0xDF);	/* DC_START_CLBR := 0 */
		lime_spi_blk_read(cmdcontext_spi,	0x50, &reg0x50_val);		/* Read DC_REG_VAL */
		if ( (reg0x50_val&0x3F) == 0)
		{
			printf("PANIC: Algorithm does not converge!\r\n");
			syrtem_spi_finictxt(cmdcontext_spi);
			openair0_close();
			system("/bin/stty cooked");
			return;
		}
	}
	printf("   - Q : reg0x50_val=0x%.2x: \r\n", reg0x50_val&0x3F);
	lime_spi_blk_write(cmdcontext_spi,	0x09, clk_en);	/* TopSPI::CLK_EN[5] := 1 */
#endif

#if 1
	/* RX VGA2 DC Offset Calibration */
	printf("*** RX VGA2 DC Offset Calibration **********\r\n");
	lime_spi_blk_read(cmdcontext_spi,	0x09, &reg0x09_val);			/* Save TopSPI::CLK_EN Register */
	clk_en	= reg0x09_val;
	lime_spi_blk_write(cmdcontext_spi,	0x09, reg0x09_val|0x10);		/* TopSPI::CLK_EN[4] := 1 Rx VGA2 DC cal */

	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x08);		/* RESET REGISTER */
	lime_spi_blk_write(cmdcontext_spi,	0x62, 0x1F);		/* DC_CNT_VAL = 0x1F */
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x18);		/* START LOAD into DC_REG_VAL 0x60 */
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x08);		/* STOP  LOAD into DC_REG_VAL 0x60 */

	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xF8);		/* DC_ADDR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val|(1<<5));		/* DC_START_CLBR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xDF);		/* DC_START_CLBR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);			/* Read DC_REG_VAL */
	if ( (reg0x60_val&0x3F) == 31)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x08);		/* RESET REGISTER */
		lime_spi_blk_write(cmdcontext_spi,	0x62, 0x00);		/* DC_CNT_VAL = 0x00 */
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x18);		/* START LOAD into DC_REG_VAL 0x60 */
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x08);		/* STOP  LOAD into DC_REG_VAL 0x60 */

		lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val|(1<<5));	/* DC_START_CLBR := 1 */
		lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xDF);	/* DC_START_CLBR := 0 */
		lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);		/* Read DC_REG_VAL */
		if ( (reg0x60_val&0x3F) == 0)
		{
			printf("PANIC: Algorithm does not converge!\r\n");
			syrtem_spi_finictxt(cmdcontext_spi);
			openair0_close();
			system("/bin/stty cooked");
			return;
		}
	}
	printf("   - DC_ADDR=0 : reg0x60_val=0x%.2x: \r\n", reg0x60_val&0x3F);

	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x09);		/* RESET REGISTER */
	lime_spi_blk_write(cmdcontext_spi,	0x62, 0x1F);		/* DC_CNT_VAL = 0x1F */
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x19);		/* START LOAD into DC_REG_VAL 0x60 */
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x09);		/* STOP  LOAD into DC_REG_VAL 0x60 */

	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xF9);		/* DC_ADDR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val|(1<<5));		/* DC_START_CLBR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xDF);		/* DC_START_CLBR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);			/* Read DC_REG_VAL */
	if ( (reg0x60_val&0x3F) == 31)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x09);		/* RESET REGISTER */
		lime_spi_blk_write(cmdcontext_spi,	0x62, 0x00);		/* DC_CNT_VAL = 0x00 */
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x19);		/* START LOAD into DC_REG_VAL 0x60 */
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x09);		/* STOP  LOAD into DC_REG_VAL 0x60 */

		lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val|(1<<5));	/* DC_START_CLBR := 1 */
		lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xDF);	/* DC_START_CLBR := 0 */
		lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);		/* Read DC_REG_VAL */
		if ( (reg0x60_val&0x3F) == 0)
		{
			printf("PANIC: Algorithm does not converge!\r\n");
			syrtem_spi_finictxt(cmdcontext_spi);
			openair0_close();
			system("/bin/stty cooked");
			return;
		}
	}
	printf("   - DC_ADDR=1 : reg0x60_val=0x%.2x: \r\n", reg0x60_val&0x3F);

	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0A);		/* RESET REGISTER */
	lime_spi_blk_write(cmdcontext_spi,	0x62, 0x1F);		/* DC_CNT_VAL = 0x1F */
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x1A);		/* START LOAD into DC_REG_VAL 0x60 */
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0A);		/* STOP  LOAD into DC_REG_VAL 0x60 */

	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xFA);		/* DC_ADDR := 2 */
	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val|(1<<5));		/* DC_START_CLBR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xDF);		/* DC_START_CLBR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);			/* Read DC_REG_VAL */
	if ( (reg0x60_val&0x3F) == 31)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0A);		/* RESET REGISTER */
		lime_spi_blk_write(cmdcontext_spi,	0x62, 0x00);		/* DC_CNT_VAL = 0x00 */
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x1A);		/* START LOAD into DC_REG_VAL 0x60 */
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0A);		/* STOP  LOAD into DC_REG_VAL 0x60 */

		lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val|(1<<5));	/* DC_START_CLBR := 1 */
		lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xDF);	/* DC_START_CLBR := 0 */
		lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);		/* Read DC_REG_VAL */
		if ( (reg0x60_val&0x3F) == 0)
		{
			printf("PANIC: Algorithm does not converge!\r\n");
			syrtem_spi_finictxt(cmdcontext_spi);
			openair0_close();
			system("/bin/stty cooked");
			return;
		}
	}
	printf("   - DC_ADDR=2 : reg0x60_val=0x%.2x: \r\n", reg0x60_val&0x3F);

	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0B);		/* RESET REGISTER */
	lime_spi_blk_write(cmdcontext_spi,	0x62, 0x1F);		/* DC_CNT_VAL = 0x1F */
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x1B);		/* START LOAD into DC_REG_VAL 0x60 */
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0B);		/* STOP  LOAD into DC_REG_VAL 0x60 */

	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xFB);		/* DC_ADDR := 3 */
	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val|(1<<5));		/* DC_START_CLBR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xDF);		/* DC_START_CLBR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);			/* Read DC_REG_VAL */
	if ( (reg0x60_val&0x3F) == 31)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0B);		/* RESET REGISTER */
		lime_spi_blk_write(cmdcontext_spi,	0x62, 0x00);		/* DC_CNT_VAL = 0x00 */
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x1B);		/* START LOAD into DC_REG_VAL 0x60 */
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0B);		/* STOP  LOAD into DC_REG_VAL 0x60 */

		lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val|(1<<5));	/* DC_START_CLBR := 1 */
		lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xDF);	/* DC_START_CLBR := 0 */
		lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);		/* Read DC_REG_VAL */
		if ( (reg0x60_val&0x3F) == 0)
		{
			printf("PANIC: Algorithm does not converge!\r\n");
			syrtem_spi_finictxt(cmdcontext_spi);
			openair0_close();
			system("/bin/stty cooked");
			return;
		}
	}
	printf("   - DC_ADDR=3 : reg0x60_val=0x%.2x: \r\n", reg0x60_val&0x3F);

	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0C);		/* RESET REGISTER */
	lime_spi_blk_write(cmdcontext_spi,	0x62, 0x1F);		/* DC_CNT_VAL = 0x1F */
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x1C);		/* START LOAD into DC_REG_VAL 0x60 */
	lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0C);		/* STOP  LOAD into DC_REG_VAL 0x60 */

	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xFC);		/* DC_ADDR := 4 */
	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val|(1<<5));		/* DC_START_CLBR := 1 */
	lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);			/* Save Register */
	lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xDF);		/* DC_START_CLBR := 0 */
	lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);			/* Read DC_REG_VAL */
	if ( (reg0x60_val&0x3F) == 31)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0C);		/* RESET REGISTER */
		lime_spi_blk_write(cmdcontext_spi,	0x62, 0x00);		/* DC_CNT_VAL = 0x00 */
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x1C);		/* START LOAD into DC_REG_VAL 0x60 */
		lime_spi_blk_write(cmdcontext_spi,	0x63, 0x0C);		/* STOP  LOAD into DC_REG_VAL 0x60 */

		lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val|(1<<5));	/* DC_START_CLBR := 1 */
		lime_spi_blk_read(cmdcontext_spi,	0x63, &reg0x63_val);		/* Save Register */
		lime_spi_blk_write(cmdcontext_spi,	0x63, reg0x63_val&0xDF);	/* DC_START_CLBR := 0 */
		lime_spi_blk_read(cmdcontext_spi,	0x60, &reg0x60_val);		/* Read DC_REG_VAL */
		if ( (reg0x60_val&0x3F) == 0)
		{
			printf("PANIC: Algorithm does not converge!\r\n");
			syrtem_spi_finictxt(cmdcontext_spi);
			openair0_close();
			system("/bin/stty cooked");
			return;
		}
	}
	printf("   - DC_ADDR=4 : reg0x60_val=0x%.2x: \r\n", reg0x60_val&0x3F);

	lime_spi_blk_write(cmdcontext_spi,	0x09, clk_en);	/* TopSPI::CLK_EN[5] := 1 */

	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);
#endif
#endif

#if 0
	printf("VGA1GAIN log-linear vs raw access test\r\n");
	printf(" Change log-linear\r\n");
	lime_spi_blk_write(cmdcontext_spi,	0x41, 0x00);
	lime_spi_blk_read(cmdcontext_spi,	0x41, &reg0x41_val);
	lime_spi_blk_read(cmdcontext_spi,	0x4B, &reg0x4B_val);
	printf("VGA1GAIN reg0x41_val=0x%.2x, reg0x4B_val=0x%.2x\r\n", reg0x41_val, reg0x4B_val);
	lime_spi_blk_write(cmdcontext_spi,	0x41, 0x15);
	lime_spi_blk_read(cmdcontext_spi,	0x41, &reg0x41_val);
	lime_spi_blk_read(cmdcontext_spi,	0x4B, &reg0x4B_val);
	printf("VGA1GAIN reg0x41_val=0x%.2x, reg0x4B_val=0x%.2x\r\n", reg0x41_val, reg0x4B_val);
	lime_spi_blk_write(cmdcontext_spi,	0x41, 0x1E);
	lime_spi_blk_read(cmdcontext_spi,	0x41, &reg0x41_val);
	lime_spi_blk_read(cmdcontext_spi,	0x4B, &reg0x4B_val);
	printf("VGA1GAIN reg0x41_val=0x%.2x, reg0x4B_val=0x%.2x\r\n", reg0x41_val, reg0x4B_val);
	lime_spi_blk_write(cmdcontext_spi,	0x41, 0x1F);
	lime_spi_blk_read(cmdcontext_spi,	0x41, &reg0x41_val);
	lime_spi_blk_read(cmdcontext_spi,	0x4B, &reg0x4B_val);
	printf("VGA1GAIN reg0x41_val=0x%.2x, reg0x4B_val=0x%.2x\r\n", reg0x41_val, reg0x4B_val);

	printf(" Change raw access\r\n");
	lime_spi_blk_write(cmdcontext_spi,	0x4B, 0x06);
	lime_spi_blk_read(cmdcontext_spi,	0x41, &reg0x41_val);
	lime_spi_blk_read(cmdcontext_spi,	0x4B, &reg0x4B_val);
	printf("VGA1GAIN reg0x41_val=0x%.2x, reg0x4B_val=0x%.2x\r\n", reg0x41_val, reg0x4B_val);
	lime_spi_blk_write(cmdcontext_spi,	0x4B, 0x50);
	lime_spi_blk_read(cmdcontext_spi,	0x41, &reg0x41_val);
	lime_spi_blk_read(cmdcontext_spi,	0x4B, &reg0x4B_val);
	printf("VGA1GAIN reg0x41_val=0x%.2x, reg0x4B_val=0x%.2x\r\n", reg0x41_val, reg0x4B_val);
	lime_spi_blk_write(cmdcontext_spi,	0x4B, 0xE3);
	lime_spi_blk_read(cmdcontext_spi,	0x41, &reg0x41_val);
	lime_spi_blk_read(cmdcontext_spi,	0x4B, &reg0x4B_val);
	printf("VGA1GAIN reg0x41_val=0x%.2x, reg0x4B_val=0x%.2x\r\n", reg0x41_val, reg0x4B_val);
	lime_spi_blk_write(cmdcontext_spi,	0x4B, 0xFF);
	lime_spi_blk_read(cmdcontext_spi,	0x41, &reg0x41_val);
	lime_spi_blk_read(cmdcontext_spi,	0x4B, &reg0x4B_val);
	printf("VGA1GAIN reg0x41_val=0x%.2x, reg0x4B_val=0x%.2x\r\n", reg0x41_val, reg0x4B_val);

#endif

//	lime_cfg_blk_write(cmdcontext_spi,	PIO_CFG_PARAMID_DCO_DACCAL, 0x19);
//	lime_cfg_blk_write(cmdcontext_spi,	PIO_CFG_PARAMID_RCCAL_LPF,  0x20);

//	lime_cfg_blk_write(cmdcontext_spi,	PIO_CFG_PARAMID_TX_LPF_DC_CNTVAL_I, 0x18);
//	lime_cfg_blk_write(cmdcontext_spi,	PIO_CFG_PARAMID_TX_LPF_DC_CNTVAL_Q, 0x28);

//	lime_cfg_blk_write(cmdcontext_spi,	PIO_CFG_PARAMID_RX_LPF_DC_CNTVAL_I, 0x11);
//	lime_cfg_blk_write(cmdcontext_spi,	PIO_CFG_PARAMID_RX_LPF_DC_CNTVAL_Q, 0x2B);

//	lime_cfg_blk_write(cmdcontext_spi,	PIO_CFG_PARAMID_RX_VGA2_DC_CNTVAL_DC, 0x1B);
//	lime_cfg_blk_write(cmdcontext_spi,	PIO_CFG_PARAMID_RX_VGA2_DC_CNTVAL_Ia, 0x21);
//	lime_cfg_blk_write(cmdcontext_spi,	PIO_CFG_PARAMID_RX_VGA2_DC_CNTVAL_Qa, 0x25);
//	lime_cfg_blk_write(cmdcontext_spi,	PIO_CFG_PARAMID_RX_VGA2_DC_CNTVAL_Ib, 0x15);
//	lime_cfg_blk_write(cmdcontext_spi,	PIO_CFG_PARAMID_RX_VGA2_DC_CNTVAL_Qb, 0x20);

//	lms_read_registers(cmdcontext_spi);
//	lms_read_cal_registers(cmdcontext_spi);

//	tunevcocap_tx(cmdcontext_spi);
//	tunevcocap_rx(cmdcontext_spi);

//	lime_spi_blk_write(cmdcontext_spi,	0x5A, 0x30);
//	printf("cmdcontext_spi->spi_tid=%d\r\n", ((ioctl_arg_spi_t *)cmdcontext_spi)->spi_tid);
//	usleep(1000);
//	lime_spi_blk_read(cmdcontext_spi,	0x04, &reg0x04_val);
//	printf("cmdcontext_spi->spi_tid=%d\r\n", ((ioctl_arg_spi_t *)cmdcontext_spi)->spi_tid);
//	printf("reg0x04_val = 0x%.2x\r\n", reg0x04_val);

	lms_read_registers(cmdcontext_spi);
	lms_read_cal_registers(cmdcontext_spi);

	usleep(1000);

	syrtem_spi_finictxt(cmdcontext_spi);

	openair0_close();

	system("/bin/stty cooked");
	return;
}
