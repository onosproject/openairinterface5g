/*
 * syr_pio.h
 *
 *  Created on: 10 déc. 2015
 *      Author: root
 */

#ifndef EXPRESS_MIMO_SOFTWARE_SDR_EXMIMO2_SYR_PIO_H_
#define EXPRESS_MIMO_SOFTWARE_SDR_EXMIMO2_SYR_PIO_H_

#define TIMEOUT		100

#define	PIO_COMMAND_MASK			0xFF000000	//	0b11111111 00000000 00000000 00000000
#define	PIO_COMMAND_SIZE			0xFF
#define	PIO_COMMAND_SHIFT			24
#define	PIO_CMD_SYR_EURECOM_FLAG	0x00
#define	PIO_CMD_SYR_SPI_FLAG		0x01
#define	PIO_CMD_SYR_INT_FLAG		0x02
#define	PIO_CMD_SYR_CFG_FLAG		0x03
#define	PIO_CMD_SYR_TEST_FLAG		0x0F

typedef enum
{
	SYR_PIO_CMD_EURECOM = 0x00,	// see syr_pio_eurecom_t
	SYR_PIO_CMD_SPI		= 0x01, // see syr_pio_spi_t
	SYR_PIO_CMD_INT		= 0x02, // see syr_pio_int_t
	SYR_PIO_CMD_CFG		= 0x03, // see syr_pio_cfg_t
	SYR_PIO_CMD_TEST	= 0x0F
} syr_pio_command_e;

typedef enum
{
	SYR_PIO_CMDSEND 		= 0,	// Sending  : command post in register and not yet read by leon3
	SYR_PIO_CMDIDLE			= 1,	// Idle     : last command has been acked either by Leon3 or GPP
	SYR_PIO_CMDRECV			= 2,	// Received : response from Leon3 to GPP has been post in register and is waiting for reading by GPP
	SYR_PIO_CMDNOTFORTID	= 3,	// Command received but not match with the expected context
	SYR_PIO_CMDERRAPP		= 4,	// Application Error : GPP side
	SYR_PIO_CMDERRIOCTL		= 5,	// IOCTL Error : error during ioctl call
	SYR_PIO_CMDERRDRV		= 6		// Driver Error : error detected in Driver
} syr_pio_cmdretval_e;

// command flag = EURECOM     ********** ********** ********** ********** ********** ********** **********
#define	PIO_EUR_RSVD_MASK			0x00FF0000	//	0b00000000 11111111 00000000 00000000
#define	PIO_EUR_RSVD_SIZE			0xFF
#define	PIO_EUR_RSVD_SHIFT			16

#define	PIO_EUR_CMD_MASK			0x0000FFFF	//	0b00000000 10000000 00000000 00000000
#define	PIO_EUR_CMD_SIZE			0xFFFF
#define	PIO_EUR_CMD_SHIFT			0

// command flag = SPI         ********** ********** ********** ********** ********** ********** **********
#define	PIO_SPI_DREADY_MASK			0x00800000	//	0b00000000 10000000 00000000 00000000
#define	PIO_SPI_DREADY_SIZE			0x1
#define	PIO_SPI_DREADY_SHIFT		23

typedef enum
{
	SYR_PIO_SPI_IDLE	 	= 0,
	SYR_PIO_SPI_DATAREADY	= 1
} syr_pio_spi_dataready_e;

#define	PIO_SPI_REQRESP_MASK		0x00400000	//	0b00000000 01000000 00000000 00000000
#define	PIO_SPI_REQRESP_SIZE		0x1
#define	PIO_SPI_REQRESP_SHIFT		22
typedef enum
{
	SYR_PIO_SPI_REQUEST 	= 0,
	SYR_PIO_SPI_RESPONSE	= 1
} syr_pio_spi_reqresp_e;

#define	PIO_SPI_RW_MASK				0x00300000	//	0b00000000 00110000 00000000 00000000
#define	PIO_SPI_RW_SIZE				0x3
#define	PIO_SPI_RW_SHIFT			20
typedef enum
{
	SYR_PIO_SPI_WRITE 	= 0,
	SYR_PIO_SPI_READ	= 1,
	SYR_PIO_SPI_ERROR	= 3
} syr_pio_spi_rw_e;

#define	PIO_SPI_TID_MASK			0x000F0000	//	0b00000000 00001111 00000000 00000000
#define	PIO_SPI_TID_SIZE			0xF
#define	PIO_SPI_TID_SHIFT			16

#define	PIO_SPI_REG_MASK			0x0000FF00	//	0b00000000 00000000 11111111 00000000
#define	PIO_SPI_REG_SIZE			0xFF
#define	PIO_SPI_REG_SHIFT			8

#define	PIO_SPI_DATA_MASK			0x000000FF	//	0b00000000 00000000 00000000 11111111
#define	PIO_SPI_DATA_SIZE			0xFF
#define	PIO_SPI_DATA_SHIFT			0

// command flag = INT         ********** ********** ********** ********** ********** ********** **********
#define	PIO_INT_DREADY_MASK			0x00800000	//	0b00000000 10000000 00000000 00000000
#define	PIO_INT_DREADY_SIZE			0x1
#define	PIO_INT_DREADY_SHIFT		23
typedef enum
{
	SYR_PIO_INT_IDLE	 	= 0,
	SYR_PIO_INT_DATAREADY	= 1
} syr_pio_int_dataready_e;

#define	PIO_INT_REQRESP_MASK		0x00400000	//	0b00000000 01000000 00000000 00000000
#define	PIO_INT_REQRESP_SIZE		0x1
#define	PIO_INT_REQRESP_SHIFT		22
typedef enum
{
	SYR_PIO_INT_REQUEST 	= 0,
	SYR_PIO_INT_RESPONSE	= 1
} syr_pio_int_reqresp_e;

#define	PIO_INT_RW_MASK				0x00300000	//	0b00000000 00110000 00000000 00000000
#define	PIO_INT_RW_SIZE				0x3
#define	PIO_INT_RW_SHIFT			20
typedef enum
{
	SYR_PIO_INT_EXEC 	= 0,
	SYR_PIO_INT_ERROR	= 3
} syr_pio_int_rw_e;

#define	PIO_INT_TID_MASK			0x000F0000	//	0b00000000 00001111 00000000 00000000
#define	PIO_INT_TID_SIZE			0xF
#define	PIO_INT_TID_SHIFT			16

#define	PIO_INT_CNT_MASK			0x0000FFFF	//	0b00000000 00000000 11111111 00000000
#define	PIO_INT_CNT_SIZE			0xFFFF
#define	PIO_INT_CNT_SHIFT			0

// command flag = CFG         ********** ********** ********** ********** ********** ********** **********
#define	PIO_CFG_DREADY_MASK			0x00800000	//	0b00000000 10000000 00000000 00000000
#define	PIO_CFG_DREADY_SIZE			0x1
#define	PIO_CFG_DREADY_SHIFT		23

typedef enum
{
	SYR_PIO_CFG_IDLE	 	= 0,
	SYR_PIO_CFG_DATAREADY	= 1
} syr_pio_cfg_dataready_e;

#define	PIO_CFG_REQRESP_MASK		0x00400000	//	0b00000000 01000000 00000000 00000000
#define	PIO_CFG_REQRESP_SIZE		0x1
#define	PIO_CFG_REQRESP_SHIFT		22
typedef enum
{
	SYR_PIO_CFG_REQUEST 	= 0,
	SYR_PIO_CFG_RESPONSE	= 1
} syr_pio_cfg_reqresp_e;

#define	PIO_CFG_RW_MASK				0x00300000	//	0b00000000 00110000 00000000 00000000
#define	PIO_CFG_RW_SIZE				0x3
#define	PIO_CFG_RW_SHIFT			20
typedef enum
{
	SYR_PIO_CFG_WRITE 	= 0,
	SYR_PIO_CFG_READ	= 1,
	SYR_PIO_CFG_ERROR	= 3
} syr_pio_cfg_rw_e;

#define	PIO_CFG_TID_MASK			0x000F0000	//	0b00000000 00001111 00000000 00000000
#define	PIO_CFG_TID_SIZE			0xF
#define	PIO_CFG_TID_SHIFT			16

#define	PIO_CFG_PID_MASK			0x0000FF00	//	0b00000000 00000000 11111111 00000000
#define	PIO_CFG_PID_SIZE			0xFF
#define	PIO_CFG_PID_SHIFT			8

#define	PIO_CFG_PVAL_MASK			0x000000FF	//	0b00000000 00000000 00000000 11111111
#define	PIO_CFG_PVAL_SIZE			0xFF
#define	PIO_CFG_PVAL_SHIFT			0

#define	PIO_CFG_PARAMID_BEGIN						0xE0
#define	PIO_CFG_PARAMID_DCO_DACCAL					0xE1
#define	PIO_CFG_PARAMID_RCCAL_LPF					0xE2
#define	PIO_CFG_PARAMID_TX_LPF_DC_CNTVAL_I			0xE3
#define	PIO_CFG_PARAMID_TX_LPF_DC_CNTVAL_Q			0xE4
#define	PIO_CFG_PARAMID_RX_LPF_DC_CNTVAL_I			0xE5
#define	PIO_CFG_PARAMID_RX_LPF_DC_CNTVAL_Q			0xE6

#define	PIO_CFG_PARAMID_RX_VGA2_DC_CNTVAL_DC		0xE7
#define	PIO_CFG_PARAMID_RX_VGA2_DC_CNTVAL_Ia		0xE8
#define	PIO_CFG_PARAMID_RX_VGA2_DC_CNTVAL_Qa		0xE9
#define	PIO_CFG_PARAMID_RX_VGA2_DC_CNTVAL_Ib		0xEA
#define	PIO_CFG_PARAMID_RX_VGA2_DC_CNTVAL_Qb		0xEB

#define	PIO_CFG_PARAMID_VGA1DC_I					0xF1
#define	PIO_CFG_PARAMID_VGA1DC_Q					0xF2
#define	PIO_CFG_PARAMID_TX_DC_OFFSET_Q				0xF3

#define	PIO_CFG_PARAMID_END							0xFF
#define	PIO_CFG_MAXPARAM_NUM						0xFF

extern unsigned char exmimo_cfg_params[PIO_CFG_MAXPARAM_NUM][MAX_ANTENNAS];

// command flag = TEST        ********** ********** ********** ********** ********** ********** **********
// COMMAND in EURECOM SPACE			0x0F00yyyy
//#define SYR_HEARTBEAT				0x0F004000
#define SYR_TEST					0x0F004000
#define SYR_TEST_NOK				0x0F007001
#define SYR_TEST_OK					0x0F007002

// COMMON DESCRIPTION         ********** ********** ********** ********** ********** ********** **********
typedef	struct pio_eurecom_s
{
	unsigned char	command_flag;		// PIO command flag 0x00 : EURECOM, 0X01 : SPI
	unsigned char	eurecom_rsvd;		// must be 0x00
	unsigned short	eurecom_cmd;
}	pio_eurecom_t;

typedef	struct pio_spi_s
{
	unsigned char		command_flag;		// PIO command flag 0x00 : EURECOM, 0X01 : SPI
	unsigned char		spi_dready	: 1;	// 0b0 NO DATA,		0b1 DATA READY
	unsigned char		spi_rqrs	: 1;	// 0b0 REQUEST,		0b1 RESPONSE
	unsigned char		spi_rw		: 2;	// 0b00 WRITE,		0b01 READ, 		0b11 ERROR
	unsigned char		spi_tid		: 4;	// Transaction Id
	unsigned char		spi_reg;
	unsigned char		spi_data;
}	pio_spi_t;

typedef	struct pio_int_s
{
	unsigned char		command_flag;		// PIO command flag 0x00 : EURECOM, 0X01 : SPI
	unsigned char		int_dready	: 1;	// 0b0 NO DATA,		0b1 DATA READY
	unsigned char		int_rqrs	: 1;	// 0b0 REQUEST,		0b1 RESPONSE
	unsigned char		int_rw		: 2;	// 0b00 WRITE,		0b01 READ, 		0b11 ERROR
	unsigned char		int_tid		: 4;	// Transaction Id
	unsigned short		int_cnt;
}	pio_int_t;

typedef	struct pio_cfg_s
{
	unsigned char		command_flag;		// PIO command flag 0x00 : EURECOM, 0X01 : SPI
	unsigned char		cfg_dready	: 1;	// 0b0 NO DATA,		0b1 DATA READY
	unsigned char		cfg_rqrs	: 1;	// 0b0 REQUEST,		0b1 RESPONSE
	unsigned char		cfg_rw		: 2;	// 0b00 WRITE,		0b01 READ, 		0b11 ERROR
	unsigned char		cfg_tid		: 4;	// Transaction Id
	unsigned char		cfg_pid;			// parameter Identifier
	unsigned char		cfg_pval;			// parameter Value
}	pio_cfg_t;

typedef union
{
	unsigned char		command_flag;		// PIO command flag 0x00 : EURECOM, 0X01 : SPI
	pio_eurecom_t		pio_eurecom;
	pio_spi_t			pio_spi;
	pio_int_t			pio_int;
	pio_cfg_t			pio_cfg;
} syr_pio_prim_t;

typedef	struct argument_s
{
	int	card_id;
	int	value;
}	argument_t;

typedef enum
{
	IOCTL_ARG_UNKNOWN 	= 0,
	IOCTL_ARG_SUCCESS 	= 1,
	IOCTL_ARG_FAILED	= 2
} ioctl_arg_drv_completion_e;

typedef	struct ioctl_arg_spi_s
{
	int							card_id;
	syr_pio_spi_reqresp_e		spi_rqrs;	// 0b0 REQUEST,		0b1 RESPONSE
	syr_pio_spi_rw_e			spi_rw;		// 0b0 WRITE,		0b1 READ
	unsigned char				spi_tid;	// Transaction Id
	unsigned char				spi_reg;
	unsigned char				spi_data;
	uint32_t					spi_pio_result;
	ioctl_arg_drv_completion_e	drv_comp;
}	ioctl_arg_spi_t;

typedef	struct ioctl_arg_int_s
{
	int							card_id;
	syr_pio_int_reqresp_e		int_rqrs;	// 0b0 REQUEST,		0b1 RESPONSE
	syr_pio_int_rw_e			int_rw;		// 0b0 WRITE,		0b1 READ
	unsigned char				int_tid;	// Transaction Id
	unsigned short				int_cnt;
	uint32_t					int_pio_result;
	ioctl_arg_drv_completion_e	drv_comp;
}	ioctl_arg_int_t;

typedef	struct ioctl_arg_cfg_s
{
	int							card_id;
	syr_pio_cfg_reqresp_e		cfg_rqrs;	// 0b0 REQUEST,		0b1 RESPONSE
	syr_pio_cfg_rw_e			cfg_rw;		// 0b0 WRITE,		0b1 READ
	unsigned char				cfg_tid;	// Transaction Id
	unsigned char				cfg_reg;
	unsigned char				cfg_data;
	uint32_t					cfg_pio_result;
	ioctl_arg_drv_completion_e	drv_comp;
}	ioctl_arg_cfg_t;

#endif /* EXPRESS_MIMO_SOFTWARE_SDR_EXMIMO2_SYR_PIO_H_ */
