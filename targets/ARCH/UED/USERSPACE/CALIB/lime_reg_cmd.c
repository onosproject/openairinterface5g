#include <stdio.h>
#include <unistd.h>
#include "openair0_lib.h"

#include "syr_pio.h"
#include "lime_spi_cmd.h"

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

	printf("RX LO leakage cancellation :\r\n");
	lime_spi_blk_read(cmdcontext_spi, 0x71, &data);
	printf("RX_FE_1          = 0x%.2x\r\n", data);
	lime_spi_blk_read(cmdcontext_spi, 0x72, &data);
	printf("RX_FE_2          = 0x%.2x\r\n", data);

	printf("TX LO leakage cancellation :\r\n");
	lime_spi_blk_read(cmdcontext_spi, 0x42, &data);
	printf("TX_RF_2          = 0x%.2x\r\n", data);
	lime_spi_blk_read(cmdcontext_spi, 0x43, &data);
	printf("TX_RF_3          = 0x%.2x\r\n", data);

	printf("\r\n");
	return;
}
