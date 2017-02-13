/*
 * lime_spi_cmd.h
 *
 *  Created on: 5 févr. 2016
 *      Author: root
 */

#ifndef SYRPCIEAPP_ARCH_EXMIMO_USERSPACE_CALIB_LIME_SPI_CMD_H_
#define SYRPCIEAPP_ARCH_EXMIMO_USERSPACE_CALIB_LIME_SPI_CMD_H_

#define	LMS_WRITE(dev, addr, data) ({			\
		lime_spi_blk_write(dev, addr, data);	\
		0; /* "Return" 0 */						\
	})

#define	LMS_READ(dev, addr, data_ptr) ({			\
		lime_spi_blk_read(dev, addr, data_ptr);	\
		0; /* "Return" 0 */						\
	})

/**
 * Module selection for those which have both RX and TX constituents
 */
void lime_spi_blk_write(void *spictxt, unsigned char reg, unsigned char data);
void lime_spi_blk_read(void *spictxt, unsigned char reg, unsigned char *pdata);

#endif /* SYRPCIEAPP_ARCH_EXMIMO_USERSPACE_CALIB_LIME_SPI_CMD_H_ */
