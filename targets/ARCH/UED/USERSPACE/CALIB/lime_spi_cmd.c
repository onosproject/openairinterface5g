
#include <stdio.h>
#include <unistd.h>
#include "openair0_lib.h"

#include "syr_pio.h"

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
