#include <stdio.h>
#include <unistd.h>
#include "openair0_lib.h"

#include "syr_pio.h"
#include "lime_spi_cmd.h"

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
