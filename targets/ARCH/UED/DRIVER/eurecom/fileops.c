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

/** fileops.c
*
*  Device IOCTL File Operations on character device /dev/openair0
*
*  Authors: Matthias Ihmig <matthias.ihmig@mytum.de>, 2012, 2013
*           Riadh Ghaddab <riadh.ghaddab@eurecom.fr>
*           Raymond Knopp <raymond.knopp@eurecom.fr>
*
*  Changelog:
*  14.01.2013: removed remaining of BIGPHYS stuff and replaced with pci_alloc_consistent
*/
#include <linux/delay.h>
#include <asm/msr.h>

#include "openair_device.h"
#include "defs.h"
#include "extern.h"

#include "pcie_interface.h"

#include "syr_pio.h"

/* TODO : change 1 */
#define invert4(x)			\
{							\
	unsigned int	ltmp;	\
    ltmp	= x;			\
	x	= ((ltmp & 0xff) << 24) | ((ltmp & 0xff00) << 8) | ((ltmp & 0xff0000) >> 8) | ((ltmp & 0xff000000) >> 24);	\
}

#define MAX_IOCTL_ACK_CNT	500

extern int			get_frame_done;
extern spinlock_t	mLock;

int is_card_num_invalid(int card)
{
	if ( (card < 0) || (card >= number_of_cards) )
	{
		printk("[openair][IOCTL]: ERROR: received invalid card number (%d)!\n", card);
		return (-1);
	}
	else
	{
		return 0;
	}
}

#define CRC32_BE			1
#define CRC32_BE_POLYNOMIAL	0x04C11DB7
#define CRC32_LE			0
#define CRC32_LE_POLYNOMIAL	0xEDB88320

uint32_t crc32_tab[256];

void gen_crc32_tab(int endianness)
{
	uint32_t	rem	= 0;
	int			i	= 0;
	int			j	= 0;

	for (i = 0; i < 256; i++)
	{
		if (endianness)	// BE
			rem	= (i << 24);
		else
			rem	= i;

		for (j = 8; j; j--)
		{
			if (endianness)	// BE
			{
				if (rem & 0x80000000)
					rem = ((rem << 1) ^ CRC32_BE_POLYNOMIAL);
				else
					rem = (rem << 1);
			}
			else			// LE
			{
				if (rem & 1)
					rem = ((rem >> 1) ^ CRC32_LE_POLYNOMIAL);
				else
					rem = (rem >> 1);
			}
		}
		crc32_tab[i] = rem;
//		printk("crc32_tab[%d]=0x%08X\n", i, rem);
	}
	return;
}

uint32_t update_crc32_be(uint32_t crc32, const void *data, size_t size)
{
	const void *s		= (uint8_t *)data;
	const void *p		= (uint8_t *)data + size;
	uint8_t		octet;

	for (s = data; s < p; s++)
	{
		octet	= *((uint8_t *)s);
		crc32	= (crc32 << 8) ^ crc32_tab[(crc32 >> 24) ^ octet];
	}
	return crc32;
}

uint32_t update_crc32_le(uint32_t crc32, const void *data, size_t size)
{
	const void *s		= (uint8_t *)data;
	const void *p		= (uint8_t *)data + size;
	uint8_t		octet;

	for (s = data; s < p; s++)
	{
		octet	= *((uint8_t *)s);
		crc32	= crc32_tab[(uint8_t)crc32 ^ octet] ^ (crc32 >> 8);
	}
	return crc32;
}

//-----------------------------------------------------------------------------
int openair_device_open (struct inode *inode, struct file *filp)
{
	//printk("[openair][MODULE]  openair_open()\n");
	return 0;
}

//-----------------------------------------------------------------------------
int openair_device_release (struct inode *inode, struct file *filp)
{
	//  printk("[openair][MODULE]  openair_release(), MODE = %d\n",openair_daq_vars.mode);
	return 0;
}

//-----------------------------------------------------------------------------
int openair_device_mmap(struct file *filp, struct vm_area_struct *vma)
{
	unsigned long	phys;
	unsigned long	start	= (unsigned long) vma->vm_start;
	unsigned long	size	= (unsigned long)(vma->vm_end - vma->vm_start);
	unsigned long	maxsize;
	unsigned int	memblock_ind;
	unsigned int	card;

	memblock_ind	= openair_mmap_getMemBlock(vma->vm_pgoff);
	card			= openair_mmap_getCard(vma->vm_pgoff);

	vma->vm_pgoff	= 0;

	// not supported by 64 bit kernels
	//vma->vm_flags |= VM_RESERVED;
	vma->vm_flags	|= VM_IO;

	if ( is_card_num_invalid(card) )
		return (-EINVAL);

	if (memblock_ind == openair_mmap_BIGSHM)
	{
		// map a buffer from bigshm
		maxsize	= BIGSHM_SIZE_PAGES<<PAGE_SHIFT;

		if (size > maxsize)
		{
			printk("[openair][MMAP][ERROR] Trying to map more than %d bytes (req size=%d)\n",
					(unsigned int)(BIGSHM_SIZE_PAGES<<PAGE_SHIFT),
					(unsigned int)size);
			return (-EINVAL);
		}

		phys	= bigshm_head_phys[card];
	}
	else if ( (memblock_ind & 1) == 1 )
	{
		// mmap a RX buffer
		maxsize	= ADAC_BUFFERSZ_PERCHAN_B;

		if (size > maxsize)
		{
			printk("[openair][MMAP][ERROR] Trying to map more than %d bytes (req size=%d)\n",
					(unsigned int)(ADAC_BUFFERSZ_PERCHAN_B),
					(unsigned int)size);
			return (-EINVAL);
		}

		phys	= p_exmimo_pci_phys[card]->adc_head[ openair_mmap_getAntRX(memblock_ind) ];
	}
	else
	{
		// mmap a TX buffer
		maxsize	= ADAC_BUFFERSZ_PERCHAN_B;

		if (size > maxsize)
		{
			printk("[openair][MMAP][ERROR] Trying to map more than %d bytes (%d)\n",
					(unsigned int)(ADAC_BUFFERSZ_PERCHAN_B),
					(unsigned int)size);
			return (-EINVAL);
		}

		phys	= p_exmimo_pci_phys[card]->dac_head[ openair_mmap_getAntTX(memblock_ind) ];
	}

//	if (0)
//		printk("[openair][MMAP] card%d: map phys (%08lx) at start %lx, end %lx, pg_off %lx, size %lx\n",
//			card,
//			phys,
//			vma->vm_start,
//			vma->vm_end,
//			vma->vm_pgoff,
//			size);

	/* loop through all the physical pages in the buffer */
	/* Remember this won't work for vmalloc()d memory ! */
	if ( remap_pfn_range(	vma,
							start,
							phys>>PAGE_SHIFT,
							(vma->vm_end - vma->vm_start),
							vma->vm_page_prot) )
	{
		printk("[openair][MMAP] ERROR EAGAIN\n");
		return (-EAGAIN);
	}

	return 0;
}

volatile unsigned long long diff_iowrite32_start_stop_ctrl0		= 0;
volatile unsigned long long diff_iowrite32_copy_from_user_ctrl0	= 0;
volatile unsigned long long diff_iowrite32_ctrl0				= 0;
volatile unsigned long long diff_iowrite32_printk_ctrl0			= 0;

volatile unsigned long long diff_ioread32_start_stop_ctrl0		= 0;
volatile unsigned long long diff_ioread32_copy_from_user_ctrl0	= 0;
volatile unsigned long long diff_ioread32_ctrl0					= 0;
volatile unsigned long long diff_ioread32_copy_to_user_ctrl0	= 0;
volatile unsigned long long diff_ioread32_printk_ctrl0			= 0;

volatile unsigned long long diff_iowrite32_start_stop_ctrl1		= 0;
volatile unsigned long long diff_iowrite32_copy_from_user_ctrl1	= 0;
volatile unsigned long long diff_iowrite32_ctrl1				= 0;
volatile unsigned long long diff_iowrite32_printk_ctrl1			= 0;

volatile unsigned long long diff_ioread32_start_stop_ctrl1		= 0;
volatile unsigned long long diff_ioread32_copy_from_user_ctrl1	= 0;
volatile unsigned long long diff_ioread32_ctrl1					= 0;
volatile unsigned long long diff_ioread32_copy_to_user_ctrl1	= 0;
volatile unsigned long long diff_ioread32_printk_ctrl1			= 0;

volatile unsigned long long diff_iowrite32_start_stop_ctrl2		= 0;
volatile unsigned long long diff_iowrite32_copy_from_user_ctrl2	= 0;
volatile unsigned long long diff_iowrite32_ctrl2				= 0;
volatile unsigned long long diff_iowrite32_printk_ctrl2			= 0;

volatile unsigned long long diff_ioread32_start_stop_ctrl2		= 0;
volatile unsigned long long diff_ioread32_copy_from_user_ctrl2	= 0;
volatile unsigned long long diff_ioread32_ctrl2					= 0;
volatile unsigned long long diff_ioread32_copy_to_user_ctrl2	= 0;
volatile unsigned long long diff_ioread32_printk_ctrl2			= 0;

volatile unsigned char		flag_is_ioctl_already_in_used		= 0;
volatile unsigned long long	flag_is_ioctl_already_in_used_cnt	= 0;

volatile unsigned long long	spi_write_cnt						= 0;
volatile unsigned long long	int_exec_cnt						= 0;

extern volatile unsigned long long	openair_irq;
extern volatile unsigned long long	openair_irq_not_match;
extern volatile unsigned long long	ioread32_cnt;
extern volatile unsigned long long	ioread32_cnt_loop;
extern volatile unsigned long long	irqval_asserted_cnt;
extern volatile unsigned long long	irqval_not_asserted_cnt;
extern volatile unsigned long long	tasklet_argument_done;
extern volatile unsigned long long	tasklet_argument_missed;

//-----------------------------------------------------------------------------
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,35)
long openair_device_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
#else
int openair_device_ioctl(struct inode *inode, struct file *filp, unsigned int cmd, unsigned long arg)
#endif
{
	/* arg is not meaningful if no arg is passed in user space */
	//-----------------------------------------------------------------------------
	int							i;
//	int							j;
	int							c;
	int							tmp;

	static unsigned int			update_firmware_command;
	static unsigned int			update_firmware_address;
	static unsigned int			update_firmware_length;
	static unsigned int			*update_firmware_kbuffer;
	static unsigned int __user	*update_firmware_ubuffer;
	static unsigned int			update_firmware_start_address;
	static unsigned int			update_firmware_stack_pointer;
	static unsigned int			update_firmware_bss_address;
	static unsigned int			update_firmware_bss_size;
	unsigned int				*fw_block;
	unsigned int				sparc_tmp_0;
	unsigned int				sparc_tmp_1;
	static unsigned int			lendian_length;
//	uint32_t					crc32init		= 0;

	unsigned int				get_frame_cnt	= 0;

	argument_t					argp;
	ioctl_arg_spi_t				argspi;
	uint32_t					spicmd;	/* PIO exchanged with Leon3 */
	ioctl_arg_int_t				argint;
	uint32_t					intcmd;	/* PIO exchanged with Leon3 */

	unsigned long				state_flags;

	unsigned long long			ini		= 0;
	unsigned long long			end		= 0;
	unsigned long long			start	= 0;
	unsigned long long			stop	= 0;

	unsigned long long			ini1_ctrl0		= 0;
	unsigned long long			end1_ctrl0		= 0;
	unsigned long long			ini2_ctrl0		= 0;
	unsigned long long			end2_ctrl0		= 0;
	unsigned long long			ini3_ctrl0		= 0;
	unsigned long long			end3_ctrl0		= 0;
	unsigned long long			start1_ctrl0	= 0;
	unsigned long long			stop1_ctrl0		= 0;

	unsigned long long			ini1_ctrl1		= 0;
	unsigned long long			end1_ctrl1		= 0;
	unsigned long long			ini2_ctrl1		= 0;
	unsigned long long			end2_ctrl1		= 0;
	unsigned long long			ini3_ctrl1		= 0;
	unsigned long long			end3_ctrl1		= 0;
	unsigned long long			start1_ctrl1	= 0;
	unsigned long long			stop1_ctrl1		= 0;

	unsigned long long			ini1_ctrl2		= 0;
	unsigned long long			end1_ctrl2		= 0;
	unsigned long long			ini2_ctrl2		= 0;
	unsigned long long			end2_ctrl2		= 0;
	unsigned long long			ini3_ctrl2		= 0;
	unsigned long long			end3_ctrl2		= 0;
	unsigned long long			start1_ctrl2	= 0;
	unsigned long long			stop1_ctrl2		= 0;

	unsigned long				retval_user		= 0;

	short						sh_cnt			= 0;

	uint32_t					hi				= 0;
	uint32_t					lo				= 0;

	if (!flag_is_ioctl_already_in_used)
	{
		flag_is_ioctl_already_in_used	= 1;
	}
	else
	{
		printk("[openair][IOCTL]     flag_is_ioctl_already_in_used ALREADY SET\n");
		flag_is_ioctl_already_in_used_cnt++;
		flag_is_ioctl_already_in_used	= 0;
		return (-EINVAL);
	}

	switch(cmd)
	{
		case openair_STOP:
			printk("[openair][IOCTL]     openair_STOP(card%d)\n", (int)arg);
			if ( is_card_num_invalid((int)arg) )
			{
				flag_is_ioctl_already_in_used	= 0;
				return (-EINVAL);
			}
			exmimo_send_pccmd((int)arg, EXMIMO_STOP);
			break;

		case openair_STOP_WITHOUT_RESET:
			printk("[openair][IOCTL]     openair_STOP_WITHOUT_RESET(card%d)\n", (int)arg);
			if ( is_card_num_invalid((int)arg) )
			{
				flag_is_ioctl_already_in_used	= 0;
				return (-EINVAL);
			}
			exmimo_send_pccmd((int)arg, EXMIMO_STOP_WITHOUT_RESET);
			break;

		case openair_GET_FRAME:
			get_frame_cnt	= 0;
			get_frame_done	= 0;
			printk("[openair][IOCTL] : openair_GET_FRAME: calling exmimo_send_pccmd(%d, EXMIMO_GET_FRAME)\n", (int)arg);
			if ( is_card_num_invalid((int)arg) )
			{
				flag_is_ioctl_already_in_used	= 0;
				return (-EINVAL);
			}
			exmimo_send_pccmd((int)arg, EXMIMO_GET_FRAME);
			while ( (get_frame_cnt < 10) && (!get_frame_done) )
			{
				msleep(10);
				get_frame_cnt++;
			}
			if (get_frame_cnt == 200)
				printk("[openair][IOCTL] : Get frame error: no IRQ received within 100ms.\n");
			get_frame_done	= 0;
			break;

		case openair_GET_BIGSHMTOPS_KVIRT:
//			printk("[openair][IOCTL] : openair_GET_BIGSHMTOPS_KVIRT  (0x%p)[0] = %p[0] (bigshm_head) for 0..3 (sizeof %d) \n", (void *)arg, bigshm_head[0], sizeof(bigshm_head));
			copy_to_user((void *)arg, bigshm_head, sizeof(bigshm_head));
			break;

		case openair_GET_PCI_INTERFACE_BOTS_KVIRT:
//			printk("[openair][IOCTL] : openair_GET_PCI_INTERFACE_BOTS_KVIRT: copying exmimo_pci_kvirt(@%8p) to %lx (sizeof %d)\n", &exmimo_pci_kvirt[0], arg, sizeof(exmimo_pci_kvirt));
			copy_to_user((void *)arg, exmimo_pci_kvirt, sizeof(exmimo_pci_kvirt));
			break;

		case openair_GET_NUM_DETECTED_CARDS:
//			printk("[openair][IOCTL] : openair_GET_NUM_DETECTED_CARDS: *(0x%p) = %d\n", (void *)arg, number_of_cards);
			copy_to_user((void *)arg, &number_of_cards, sizeof(number_of_cards));
			break;

		case openair_DUMP_CONFIG:
//			printk("[openair][IOCTL]     openair_DUMP_CONFIG(%d)\n", (int)arg);
			if ( is_card_num_invalid((int)arg) )
			{
				printk("[openair][IOCTL]     openair_DUMP_CONFIG: Invalid card number %d.\n", (int)arg);
				flag_is_ioctl_already_in_used	= 0;
				return (-EINVAL);
			}
			printk("[openair][IOCTL] : openair_DUMP_CONFIG(%d):  exmimo_pci_kvirt[%d].exmimo_config_ptr = %p (phys %08x)\n",
					(int)arg,
					(int)arg,
					exmimo_pci_kvirt[(int)arg].exmimo_config_ptr,
					p_exmimo_pci_phys[(int)arg]->exmimo_config_ptr);
//			printk("EXMIMO_CONFIG: freq0 %d Hz, freq1 %d Hz, freqtx0 %d Hz, freqtx1 %d Hz, \nRX gain0 %d dB, RX Gain1 %d dB\n",
//					exmimo_pci_kvirt[(int)arg].exmimo_config_ptr->rf.rf_freq_rx[0],
//					exmimo_pci_kvirt[(int)arg].exmimo_config_ptr->rf.rf_freq_rx[1],
//					exmimo_pci_kvirt[(int)arg].exmimo_config_ptr->rf.rf_freq_tx[0],
//					exmimo_pci_kvirt[(int)arg].exmimo_config_ptr->rf.rf_freq_tx[1],
//					exmimo_pci_kvirt[(int)arg].exmimo_config_ptr->rf.rx_gain[0][0],
//					exmimo_pci_kvirt[(int)arg].exmimo_config_ptr->rf.rx_gain[1][0]);

			exmimo_send_pccmd((int)arg, EXMIMO_CONFIG);
			break;

		case openair_START_RT_ACQUISITION:
			printk("[openair][IOCTL]     openair_START_TX_SIG(%d): send_pccmd(EXMIMO_START_RT_ACQUISITION).\n", (int) arg);
			if ( is_card_num_invalid((int)arg) )
			{
				flag_is_ioctl_already_in_used	= 0;
				return (-EINVAL);
			}
			exmimo_send_pccmd((int)arg, EXMIMO_START_RT_ACQUISITION);
			break;

		case openair_UPDATE_FIRMWARE:
			printk("[openair][IOCTL]     openair_UPDATE_FIRMWARE\n");
			/*****************************************************************
			*   Updating the firmware of Cardbus-MIMO-1 or ExpressMIMO SoC   *
			******************************************************************/
			/* 1st argument of this ioctl indicates the action to perform among these:
			- Transfer a block of data at a specified address (given as the 2nd argument)
			and for a specified length (given as the 3rd argument, in number of 32-bit words).
			The USER-SPACE address where to find the block of data is given as the 4th
			argument.
			- Ask the Leon processor to clear the .bss section. In this case, the base
			address of section .bss is given as the 2nd argument, and its size is
			given as the 3rd one.
			- Ask the Leon processor to jump at a specified address (given as the 2nd
			argument, most oftenly expected to be the top address of Ins, Scratch Pad
			Ram), after having set the stack pointer (given as the 3rd argument).
			For the openair_UPDATE_FIRMWARE ioctl, we perform a partial infinite loop
			while acknowledging the PCI irq from Leon software: the max number of loop
			is yielded by preprocessor constant MAX_IOCTL_ACK_CNT. This avoids handing
			the kernel with an infinite polling loop. An exception is the case of clearing
			the bss: it takes time to Leon3 to perform this operation, so we poll te
			acknowledge with no limit */
			
			gen_crc32_tab(CRC32_BE);
#if 0
	uint32_t		buf[10];
	uint32_t		crc32init	= 0;
	buf[0]	= 0x40000000;	// Head .text
	buf[1]	= 0x00006F28;
	buf[2]	= 0x4001BCA0;	// Head .data
	buf[3]	= 0x00000484;
	buf[4]	= 0x4001D000;	// Head clear
	buf[5]	= 0x00002408;
	buf[6]	= 0x40000000;	// Head start exec
	buf[7]	= 0x43FFFFF0;
	buf[8]	= 0x00000001;	// Head start exec
	buf[9]	= 0x00000002;

	for (i = 0; i < 10; i++)
	{
		invert4(buf[i]);
	}
	crc32init	= update_crc32_be(0, &(buf[0]), 2*sizeof(uint32_t));
	printk("CRC32 GPP Head1 = 0x%08X\n", crc32init);
	crc32init	= update_crc32_be(0, &(buf[2]), 2*sizeof(uint32_t));
	printk("CRC32 GPP Head2 = 0x%08X\n", crc32init);
	crc32init	= update_crc32_be(0, &(buf[4]), 2*sizeof(uint32_t));
	printk("CRC32 GPP Head3 = 0x%08X\n", crc32init);
	crc32init	= update_crc32_be(0, &(buf[6]), 2*sizeof(uint32_t));
	printk("CRC32 GPP Head4 = 0x%08X\n", crc32init);
	crc32init	= update_crc32_be(0, &(buf[8]), 2*sizeof(uint32_t));
	printk("CRC32 GPP Head5 = 0x%08X\n", crc32init);
#endif
			update_firmware_command = *((unsigned int*)arg);
			switch (update_firmware_command)
			{
				case UPDATE_FIRMWARE_TRANSFER_BLOCK:
					update_firmware_address	= ((unsigned int *)arg)[1];
					update_firmware_length	= ((unsigned int *)arg)[2];
					update_firmware_ubuffer	= (unsigned int *)((unsigned int *)arg)[3];
//					update_firmware_ubuffer	= (unsigned int __user *)((unsigned int *)arg)[3];
					update_firmware_kbuffer	= (unsigned int *)kmalloc(update_firmware_length * 4 /* 4 because kmalloc expects bytes */, GFP_KERNEL);

#if 0	/* TEST Firmware at 0 */
					if (update_firmware_length >= 20480)
					{
						kfree(update_firmware_kbuffer);
						update_firmware_length	= 20480;
						update_firmware_kbuffer	= (unsigned int *)kmalloc(update_firmware_length * 4 /* 4 because kmalloc expects bytes */, GFP_KERNEL);
					}
#endif

					if (!update_firmware_kbuffer)
					{
						printk("[openair][IOCTL] Could NOT allocate %u bytes from kernel memory (kmalloc failed).\n", lendian_length * 4);
						flag_is_ioctl_already_in_used	= 0;
						return (-1);
//						break; // TODO : delete
					}

					// update all cards at the same time
					for (c = 0; c < number_of_cards; c++)
					{
						fw_block	= (unsigned int *)exmimo_pci_kvirt[c].firmware_block_ptr;
						for (i = 0; i < 32; i++)	/* Initialized fw_block tab to 0x00000000 */
							fw_block[i]	= 0;
						/* Copy the data block from user space */
						fw_block[0]	= update_firmware_address;
						fw_block[1]	= update_firmware_length;
						/* CRC32 calculation : use of fw_block[2] and fw_block[3] to record invert4 values of fw_block[0] and fw_block[1] */
						fw_block[2]	= fw_block[0];			/* Initialized fw_block[2] : CRC Head */
						fw_block[3]	= fw_block[1];			/* Initialized fw_block[3] : CRC Body */
						invert4(fw_block[2]);
						invert4(fw_block[3]);
						fw_block[2]	= update_crc32_be( 0, &(fw_block[2]), (2 * sizeof(uint32_t)) );

						//printk("copy_from_user %p => %p (pci) => fw[0]=fw_addr=%08x (ahb), fw[1]=fw_length=%d DW\n",update_firmware_ubuffer,&fw_block[16],update_firmware_address,update_firmware_length);

						tmp			= copy_from_user(update_firmware_kbuffer,
													update_firmware_ubuffer,		/* from */
													(update_firmware_length * 4)	/* in bytes */);
						if (tmp)
						{
							printk("[openair][IOCTL] Could NOT copy all data from user-space to kernel-space (%d bytes remained uncopied).\n", tmp);
							flag_is_ioctl_already_in_used	= 0;
							return (-1);
						}

						// pci_map_single(pdev[0],(void*)fw_block, update_firmware_length*4,PCI_DMA_BIDIRECTIONAL);
						fw_block[3]	= 0;			/* Initialized fw_block[2] : CRC Body */
						for (i = 0; i < update_firmware_length; i++)
						{
							fw_block[32 + i]	= ((unsigned int *)update_firmware_kbuffer)[i];
							fw_block[3]			= update_crc32_be( fw_block[3], &(fw_block[32 + i]), sizeof(uint32_t) );
							// Endian flipping is done in user-space so undo it
							invert4(fw_block[32 + i]);
						}
#if 0	/* TEST Firmware at 0 */
						for (i = 0; i < 4; i++)
							printk("fw_block[%d]=0x%.8x\n", (i), fw_block[i]);
#endif
#if 0	/* TEST Firmware at 0 */
						if (update_firmware_length >= 20480)
							fw_block[3]	= ~(fw_block[3]) + 1 + 0xFF;
#endif
#if 0	/* TEST Firmware at 0 */
						fw_block[1]	= 2;
						fw_block[2]	= update_firmware_address;			/* Initialized fw_block[2] : CRC Head */
						invert4(fw_block[2]);
						fw_block[3]	= fw_block[1];						/* Initialized fw_block[3] : CRC Body */
						invert4(fw_block[3]);
						fw_block[2]	= update_crc32_be( 0, &(fw_block[2]), (2 * sizeof(uint32_t)) );
						fw_block[32]	= 1;
						fw_block[33]	= 2;
						fw_block[3]	= update_crc32_be( 0, &(fw_block[32]), (2 * sizeof(uint32_t)) ) + 1;
						invert4(fw_block[32]);
						invert4(fw_block[33]);
#endif
#if 0	/* TEST Firmware at i */
						for (i = 0; i < 32000; i++)
						{
							fw_block[i]	= i;
//							printk("fw_block[%d]=0x%.8x\n", (i), fw_block[i]);
						}
						fw_block[0]	= update_firmware_address;
						fw_block[1]	= 32000;						
						/* CRC32 calculation : use of fw_block[2] and fw_block[3] to record invert4 values of fw_block[0] and fw_block[1] */
						fw_block[2]	= fw_block[0];			/* Initialized fw_block[2] : CRC Head */
						fw_block[3]	= fw_block[1];			/* Initialized fw_block[3] : CRC Body */
						invert4(fw_block[2]);
						invert4(fw_block[3]);
						fw_block[2]	= update_crc32_be( 0, &(fw_block[2]), (2 * sizeof(uint32_t)) );
#endif
						printk("[openair][IOCTL] UPFW_TRANSBLK fwblk[0] = 0x%.8x, fwblk[1] = 0x%.8x, fwblk[2] = 0x%.8x, fwblk[3] = 0x%.8x\n",
								fw_block[0],	// update_firmware_address
								fw_block[1],	// update_firmware_length
								fw_block[2],	// CRC header
								fw_block[3]);	// CRC section_content (.text | .data)

						exmimo_send_pccmd(c, EXMIMO_FW_INIT);

						printk("[openair][IOCTL] card%d: ok %u DW copied to address 0x%08x  (fw_block_ptr %p)\n",
								c,
								fw_block[1],	// update_firmware_length
								fw_block[0],	// update_firmware_address
								fw_block);
					}
					kfree(update_firmware_kbuffer);
					break;

				case UPDATE_FIRMWARE_CLEAR_BSS:
					update_firmware_bss_address	= ((unsigned int*)arg)[1];
					update_firmware_bss_size	= ((unsigned int*)arg)[2];
					sparc_tmp_0					= update_firmware_bss_address;
					sparc_tmp_1					= update_firmware_bss_size;
					printk("[openair][IOCTL] ok asked Leon to clear .bss (addr 0x%08x, size %d bytes)\n", sparc_tmp_0, sparc_tmp_1);
					// update all cards at the same time
					for (c = 0; c < number_of_cards; c++)
					{
						fw_block	= (unsigned int *)exmimo_pci_kvirt[c].firmware_block_ptr;
						fw_block[0]	= update_firmware_bss_address;
						fw_block[1]	= update_firmware_bss_size;
						/* CRC32 calculation : use of fw_block[2] and fw_block[3] to record invert4 values of fw_block[0] and fw_block[1] */
						fw_block[2]	= update_firmware_bss_address;			/* Initialized fw_block[2] : CRC Head */
						invert4(fw_block[2]);
						fw_block[3]	= update_firmware_bss_size;			/* Initialized fw_block[3] : CRC Body */
						invert4(fw_block[3]);
//						fw_block[2]	+= fw_block[0];
//						fw_block[2]	+= fw_block[1];
//						fw_block[2]	= ~(fw_block[2]) + 1;
						fw_block[2]	= update_crc32_be( 0, &(fw_block[2]), (2 * sizeof(uint32_t)) );
						printk("[openair][IOCTL] UPFW_CLEARBSS fwblk[0] = 0x%.8x, fwblk[1] = 0x%.8x, fwblk[2] = 0x%.8x\n",
								fw_block[0],	// update_firmware_address
								fw_block[1],	// update_firmware_length
								fw_block[2]);	// CRC section_content (.text | .data)
						exmimo_send_pccmd(c, EXMIMO_FW_CLEAR_BSS);
					}
					break;

				case UPDATE_FIRMWARE_START_EXECUTION:
					update_firmware_start_address	= ((unsigned int*)arg)[1];
					update_firmware_stack_pointer	= ((unsigned int*)arg)[2];
					sparc_tmp_0						= update_firmware_start_address;
					sparc_tmp_1						= update_firmware_stack_pointer;
					printk("[openair][IOCTL] ok asked Leon to set stack and start execution (addr 0x%08x, stackptr %08x)\n", sparc_tmp_0, sparc_tmp_1);
					for (c = 0; c < number_of_cards; c++)
					{
						fw_block	= (unsigned int *)exmimo_pci_kvirt[c].firmware_block_ptr;
						fw_block[0]	= update_firmware_start_address;
						fw_block[1]	= update_firmware_stack_pointer;
						/* CRC32 calculation : use of fw_block[2] and fw_block[3] to record invert4 values of fw_block[0] and fw_block[1] */
						fw_block[2]	= update_firmware_start_address;			/* Initialized fw_block[2] : CRC Head */
						invert4(fw_block[2]);
						fw_block[3]	= update_firmware_stack_pointer;			/* Initialized fw_block[3] : CRC Body */
						invert4(fw_block[3]);
//						fw_block[2]	+= fw_block[0];
//						fw_block[2]	+= fw_block[1];
//						fw_block[2]	= ~(fw_block[2]) + 1;
						fw_block[2]	= update_crc32_be( 0, &(fw_block[2]), (2 * sizeof(uint32_t)) );
						printk("[openair][IOCTL] UPFW_STARTEXE fwblk[0] = 0x%.8x, fwblk[1] = 0x%.8x, fwblk[2] = 0x%.8x\n",
								fw_block[0],	// update_firmware_address
								fw_block[1],	// update_firmware_length
								fw_block[2]);	// CRC section_content (.text | .data)
						exmimo_send_pccmd(c, EXMIMO_FW_START_EXEC);
						msleep(10);
						exmimo_firmware_init(c);
					}
					break;

				case UPDATE_FIRMWARE_FORCE_REBOOT:
					printk("[openair][IOCTL] ok asked Leon to reboot.\n");
					for (c = 0; c < number_of_cards; c++)
					{
						exmimo_send_pccmd(c, EXMIMO_REBOOT);
						exmimo_firmware_init(c);
					}
					break;

				case UPDATE_FIRMWARE_TEST_GOK:
					printk("[openair][IOCTL] TEST_GOK command doesn't work with ExpressMIMO. Ignored.\n");
					break;

				default:
					flag_is_ioctl_already_in_used	= 0;
					return (-1);
					break;
			}
			break;

		case syrtem_GET_CTRL0:
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			start1_ctrl0	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini1_ctrl0	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			copy_from_user((void *)&argp, (argument_t *)arg, sizeof(argument_t));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end1_ctrl0	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini2_ctrl0	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			((argument_t *)&argp)->value	= ioread32((bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL0));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end2_ctrl0	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini3_ctrl0	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			copy_to_user((argument_t *)arg, &argp, sizeof(argument_t));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end3_ctrl0	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			stop1_ctrl0	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			if (end1_ctrl0 > ini1_ctrl0)
				diff_ioread32_copy_from_user_ctrl0 = (end1_ctrl0 - ini1_ctrl0);
			if (end2_ctrl0 > ini2_ctrl0)
				diff_ioread32_ctrl0 = (end2_ctrl0 - ini2_ctrl0);
			if (end3_ctrl0 > ini3_ctrl0)
				diff_ioread32_copy_to_user_ctrl0 = (end3_ctrl0 - ini3_ctrl0);
			if (stop1_ctrl0 > start1_ctrl0)
				diff_ioread32_start_stop_ctrl0 = (stop1_ctrl0 - start1_ctrl0);
			printk("sh_cnt      =%d\n", sh_cnt);
			printk("ini1_ctrl0  =%llu\nend1_ctrl0 =%llu\n", ini1_ctrl0, end1_ctrl0);
			printk("ini2_ctrl0  =%llu\nend2_ctrl0 =%llu\n", ini2_ctrl0, end2_ctrl0);
			printk("ini3_ctrl0  =%llu\nend3_ctrl0 =%llu\n", ini3_ctrl0, end3_ctrl0);
			printk("start1_ctrl0=%llu\nstop1_ctrl0=%llu\n", start1_ctrl0, stop1_ctrl0);
			break;

		case syrtem_GET_CTRL1:
//			printk("syrtem_GET_CTRL1 001\n");
			start1_ctrl1	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini1_ctrl1	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			retval_user	= copy_from_user((void *)&argp, (argument_t *)arg, sizeof(argument_t));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end1_ctrl1	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

//			printk("syrtem_GET_CTRL1 002: argp.card_id=%d, argp.value=%d\n", argp.card_id, argp.value);

//			printk("syrtem_GET_CTRL1 002: retval_user=%ld, card=%d, ctrl1=0x%x\n", retval_user, ((argument_t *)&argp)->card_id, PCIE_CONTROL1);
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini2_ctrl1	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			if (!retval_user)
				((argument_t *)&argp)->value	= ioread32((bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL1));
			else
				((argument_t *)&argp)->value	= ioread32((bar[0] + PCIE_CONTROL1));

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end2_ctrl1	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

//			printk("syrtem_GET_CTRL1 003\n");
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini3_ctrl1	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			copy_to_user((argument_t *)arg, &argp, sizeof(argument_t));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end3_ctrl1	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

//			printk("syrtem_GET_CTRL1 004\n");
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			stop1_ctrl1	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			if (end1_ctrl1 > ini1_ctrl1)
				diff_ioread32_copy_from_user_ctrl1 = (end1_ctrl1 - ini1_ctrl1);
			if (end2_ctrl1 > ini2_ctrl1)
				diff_ioread32_ctrl1 = (end2_ctrl1 - ini2_ctrl1);
			if (end3_ctrl1 > ini3_ctrl1)
				diff_ioread32_copy_to_user_ctrl1 = (end3_ctrl1 - ini3_ctrl1);
			if (stop1_ctrl1 > start1_ctrl1)
				diff_ioread32_start_stop_ctrl1 = (stop1_ctrl1 - start1_ctrl1);
//			printk("ini1_ctrl1  =%llu\nend1_ctrl1 =%llu\n", ini1_ctrl1, end1_ctrl1);
//			printk("ini2_ctrl1  =%llu\nend2_ctrl1 =%llu\n", ini2_ctrl1, end2_ctrl1);
//			printk("ini3_ctrl1  =%llu\nend3_ctrl1 =%llu\n", ini3_ctrl1, end3_ctrl1);
//			printk("start1_ctrl1=%llu\nstop1_ctrl1=%llu\n", start1_ctrl1, stop1_ctrl1);
			break;

		case syrtem_GET_CTRL2:
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			start1_ctrl2	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini1_ctrl2	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			copy_from_user((void *)&argp, (argument_t *)arg, sizeof(argument_t));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end1_ctrl2	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini2_ctrl2	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			((argument_t *)&argp)->value	= ioread32((bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL2));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end2_ctrl2	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini3_ctrl2	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			copy_to_user((argument_t *)arg, &argp, sizeof(argument_t));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end3_ctrl2	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			stop1_ctrl2	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			if (end1_ctrl2 > ini1_ctrl2)
				diff_ioread32_copy_from_user_ctrl2 = (end1_ctrl2 - ini1_ctrl2);
			if (end2_ctrl2 > ini2_ctrl2)
				diff_ioread32_ctrl2 = (end2_ctrl2 - ini2_ctrl2);
			if (end3_ctrl2 > ini3_ctrl2)
				diff_ioread32_copy_to_user_ctrl2 = (end3_ctrl2 - ini3_ctrl2);
			if (stop1_ctrl2 > start1_ctrl2)
				diff_ioread32_start_stop_ctrl2 = (stop1_ctrl2 - start1_ctrl2);
			printk("ini1_ctrl2  =%llu\nend1_ctrl2 =%llu\n", ini1_ctrl2, end1_ctrl2);
			printk("ini2_ctrl2  =%llu\nend2_ctrl2 =%llu\n", ini2_ctrl2, end2_ctrl2);
			printk("ini3_ctrl2  =%llu\nend3_ctrl2 =%llu\n", ini3_ctrl2, end3_ctrl2);
			printk("start1_ctrl2=%llu\nstop1_ctrl2=%llu\n", start1_ctrl2, stop1_ctrl2);
			break;

		case syrtem_SET_CTRL0:
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			start	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			copy_from_user((void *)&argp, (argument_t *)arg, sizeof(argument_t));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			iowrite32(((argument_t *)&argp)->value, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL0));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			stop	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			if (end > ini)
				diff_iowrite32_copy_from_user_ctrl0 = (end - ini);
			if (end > ini)
				diff_iowrite32_ctrl0 = (end - ini);
			if (stop > start)
				diff_iowrite32_start_stop_ctrl0 = (stop - start);
			break;

		case syrtem_SET_CTRL1:
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			start	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			copy_from_user((void *)&argp, (argument_t *)arg, sizeof(argument_t));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			iowrite32(((argument_t *)&argp)->value, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL1));
			iowrite32(((argument_t *)&argp)->value+1, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL1));
			iowrite32(((argument_t *)&argp)->value+2, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL1));
			iowrite32(((argument_t *)&argp)->value+3, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL1));
			iowrite32(((argument_t *)&argp)->value+4, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL1));
			iowrite32(((argument_t *)&argp)->value+5, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL1));
			iowrite32(((argument_t *)&argp)->value+6, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL1));
			iowrite32(((argument_t *)&argp)->value+7, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL1));
			iowrite32(((argument_t *)&argp)->value+8, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL1));
			iowrite32(((argument_t *)&argp)->value+9, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL1));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			stop	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			if (end > ini)
				diff_iowrite32_copy_from_user_ctrl1 = (end - ini);
			if (end > ini)
				diff_iowrite32_ctrl1 = (end - ini);
			if (stop > start)
				diff_iowrite32_start_stop_ctrl1 = (stop - start);
			break;

		case syrtem_SET_CTRL2:
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			start	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			copy_from_user((void *)&argp, (argument_t *)arg, sizeof(argument_t));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			ini	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			iowrite32(((argument_t *)&argp)->value, (bar[((argument_t *)&argp)->card_id] + PCIE_CONTROL2));
			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			end	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

			__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
			stop	= ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
			if (end > ini)
				diff_iowrite32_copy_from_user_ctrl2 = (end - ini);
			if (end > ini)
				diff_iowrite32_ctrl2 = (end - ini);
			if (stop > start)
				diff_iowrite32_start_stop_ctrl2 = (stop - start);
			break;

		case syrtem_GET_COUNTERS:
			printk("[IOCTL] syrtem_GET_COUNTERS: \n");
			printk("[IOCTL]    diff_iowrite32_start_stop_ctrl0    =%lld\n", diff_iowrite32_start_stop_ctrl0);
			printk("[IOCTL]    diff_iowrite32_copy_from_user_ctrl0=%lld\n", diff_iowrite32_copy_from_user_ctrl0);
			printk("[IOCTL]    diff_iowrite32_ctrl0               =%lld\n\n", diff_iowrite32_ctrl0);

			printk("[IOCTL]    diff_ioread32_start_stop_ctrl0     =%lld\n", diff_ioread32_start_stop_ctrl0);
			printk("[IOCTL]    diff_ioread32_copy_from_user_ctrl0 =%lld\n", diff_ioread32_copy_from_user_ctrl0);
			printk("[IOCTL]    diff_ioread32_ctrl0                =%lld\n", diff_ioread32_ctrl0);
			printk("[IOCTL]    diff_ioread32_copy_to_user_ctrl0   =%lld\n\n", diff_ioread32_copy_to_user_ctrl0);

			printk("[IOCTL]    diff_iowrite32_start_stop_ctrl1    =%lld\n", diff_iowrite32_start_stop_ctrl1);
			printk("[IOCTL]    diff_iowrite32_copy_from_user_ctrl1=%lld\n", diff_iowrite32_copy_from_user_ctrl1);
			printk("[IOCTL]    diff_iowrite32_ctrl1               =%lld\n\n", diff_iowrite32_ctrl1);

			printk("[IOCTL]    diff_ioread32_start_stop_ctrl1     =%lld\n", diff_ioread32_start_stop_ctrl1);
			printk("[IOCTL]    diff_ioread32_copy_from_user_ctrl1 =%lld\n", diff_ioread32_copy_from_user_ctrl1);
			printk("[IOCTL]    diff_ioread32_ctrl1                =%lld\n", diff_ioread32_ctrl1);
			printk("[IOCTL]    diff_ioread32_copy_to_user_ctrl1   =%lld\n\n", diff_ioread32_copy_to_user_ctrl1);

			printk("[IOCTL]    diff_iowrite32_start_stop_ctrl2    =%lld\n", diff_iowrite32_start_stop_ctrl2);
			printk("[IOCTL]    diff_iowrite32_copy_from_user_ctrl2=%lld\n", diff_iowrite32_copy_from_user_ctrl2);
			printk("[IOCTL]    diff_iowrite32_ctrl2               =%lld\n\n", diff_iowrite32_ctrl2);

			printk("[IOCTL]    diff_ioread32_start_stop_ctrl2     =%lld\n", diff_ioread32_start_stop_ctrl2);
			printk("[IOCTL]    diff_ioread32_copy_from_user_ctrl2 =%lld\n", diff_ioread32_copy_from_user_ctrl2);
			printk("[IOCTL]    diff_ioread32_ctrl2                =%lld\n", diff_ioread32_ctrl2);
			printk("[IOCTL]    diff_ioread32_copy_to_user_ctrl2   =%lld\n\n", diff_ioread32_copy_to_user_ctrl2);

			printk("[IOCTL]    flag_is_ioctl_already_in_used_cnt  =%lld\n\n", flag_is_ioctl_already_in_used_cnt);

			printk("[IOCTL]    spi_write_cnt                  =%lld\n", spi_write_cnt);
			printk("[IOCTL]    int_exec_cnt                   =%lld\n", int_exec_cnt);
			printk("[IOCTL]    openair_irq                    =%lld\n", openair_irq);
			printk("[IOCTL]    openair_irq_not_match          =%lld\n", openair_irq_not_match);
			printk("[IOCTL]    ioread32_cnt_loop              =%lld\n", ioread32_cnt_loop);
			printk("[IOCTL]    ioread32_cnt_max               =%lld\n", ioread32_cnt);
			printk("[IOCTL]    irqval_asserted_cnt            =%lld\n", irqval_asserted_cnt);
			printk("[IOCTL]    irqval_not_asserted_cnt        =%lld\n", irqval_not_asserted_cnt);
			printk("[IOCTL]    tasklet_argument_done          =%lld\n", tasklet_argument_done);
			printk("[IOCTL]    tasklet_argument_missed        =%lld\n\n", tasklet_argument_missed);

			break;

		case syrtem_SPI_READ:
//			printk("syrtem_SPI_READ 001\n");
			retval_user	= copy_from_user((void *)&argspi, (ioctl_arg_spi_t *)arg, sizeof(ioctl_arg_spi_t));
			if ( retval_user != 0 )
			{
				printk("syrtem_SPI_READ retval_user != 0\n");
				flag_is_ioctl_already_in_used	= 0;
				return (-EFAULT);
			}
//			printk("syrtem_SPI_READ 002\n");
			/* ioread32 for SPI CMD: [SPI_FLAG][READ|WRITE][REGISTER][N/A] */
			spicmd	=  0x00000000;
			spicmd	|= ((SYR_PIO_CMD_SPI							<< PIO_COMMAND_SHIFT)		& PIO_COMMAND_MASK);
			spicmd	|= ((SYR_PIO_SPI_DATAREADY						<< PIO_SPI_DREADY_SHIFT)	& PIO_SPI_DREADY_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_rqrs		<< PIO_SPI_REQRESP_SHIFT)	& PIO_SPI_REQRESP_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_rw		<< PIO_SPI_RW_SHIFT)		& PIO_SPI_RW_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_tid		<< PIO_SPI_TID_SHIFT)		& PIO_SPI_TID_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_reg		<< PIO_SPI_REG_SHIFT)		& PIO_SPI_REG_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_data		<< PIO_SPI_DATA_SHIFT)		& PIO_SPI_DATA_MASK);
//			printk("syrtem_SPI_READ 003\n");
//			printk("syrtem_SPI_READ: retval_user=%ld, card=%d, spicmd=0x%.4x, ctrl1=0x%x\n", retval_user, ((ioctl_arg_spi_t *)&argspi)->card_id, spicmd, PCIE_CONTROL1);
			iowrite32(spicmd, (bar[((ioctl_arg_spi_t *)&argspi)->card_id] + PCIE_CONTROL1));
//			printk("syrtem_SPI_READ 004\n");
			((ioctl_arg_spi_t *)&argspi)->drv_comp	= IOCTL_ARG_SUCCESS;
			if ( copy_to_user((ioctl_arg_spi_t *)arg, &argspi, sizeof(ioctl_arg_spi_t)) != 0 )
			{
				flag_is_ioctl_already_in_used	= 0;
				return (-EFAULT);
			}
//			printk("syrtem_SPI_READ 005\n");
//			printk("[openair][IOCTL] card%d: SPI READ REG(0x%02X) COMP(%d)\n", ((ioctl_arg_spi_t *)&argspi)->card_id, ((ioctl_arg_spi_t *)&argspi)->spi_reg, ((ioctl_arg_spi_t *)&argspi)->drv_comp);
			break;

		case syrtem_SPI_WRITE:
			spi_write_cnt++;
			if ( copy_from_user((void *)&argspi, (ioctl_arg_spi_t *)arg, sizeof(ioctl_arg_spi_t)) != 0 )
			{
				flag_is_ioctl_already_in_used	= 0;
				return (-EFAULT);
			}
			/* iowrite32 for SPI CMD: [SPI_FLAG][READ|WRITE][REGISTER][DATA] */
			spicmd	=  0x00000000;
			spicmd	|= ((SYR_PIO_CMD_SPI		<< PIO_COMMAND_SHIFT)		& PIO_COMMAND_MASK);
			spicmd	|= ((SYR_PIO_SPI_DATAREADY	<< PIO_SPI_DREADY_SHIFT)	& PIO_SPI_DREADY_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_rqrs		<< PIO_SPI_REQRESP_SHIFT)	& PIO_SPI_REQRESP_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_rw			<< PIO_SPI_RW_SHIFT)		& PIO_SPI_RW_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_tid			<< PIO_SPI_TID_SHIFT)		& PIO_SPI_TID_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_reg			<< PIO_SPI_REG_SHIFT)		& PIO_SPI_REG_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_data		<< PIO_SPI_DATA_SHIFT)		& PIO_SPI_DATA_MASK);
			iowrite32(spicmd, (bar[((ioctl_arg_spi_t *)&argspi)->card_id] + PCIE_CONTROL1));
			((ioctl_arg_spi_t *)&argspi)->drv_comp	= IOCTL_ARG_SUCCESS;
			
			if ( copy_to_user((ioctl_arg_spi_t *)arg, &argspi, sizeof(ioctl_arg_spi_t)) != 0 )
			{
				flag_is_ioctl_already_in_used	= 0;
				return (-EFAULT);
			}
//			printk("[openair][IOCTL] card%d: SPI WRITE REG(0x%02X) DATA(0x%02X) COMP(%d)\n", ((ioctl_arg_spi_t *)&argspi)->card_id, ((ioctl_arg_spi_t *)&argspi)->spi_reg, ((ioctl_arg_spi_t *)&argspi)->spi_data, ((ioctl_arg_spi_t *)&argspi)->drv_comp);
			break;

		case syrtem_CFG_WRITE:
			spi_write_cnt++;
			if ( copy_from_user((void *)&argspi, (ioctl_arg_spi_t *)arg, sizeof(ioctl_arg_spi_t)) != 0 )
			{
				flag_is_ioctl_already_in_used	= 0;
				return (-EFAULT);
			}
			/* iowrite32 for SPI CMD: [SPI_FLAG][READ|WRITE][REGISTER][DATA] */
			spicmd	=  0x00000000;
			spicmd	|= ((SYR_PIO_CMD_CFG		<< PIO_COMMAND_SHIFT)		& PIO_COMMAND_MASK);
			spicmd	|= ((SYR_PIO_SPI_DATAREADY	<< PIO_SPI_DREADY_SHIFT)	& PIO_SPI_DREADY_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_rqrs		<< PIO_SPI_REQRESP_SHIFT)	& PIO_SPI_REQRESP_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_rw			<< PIO_SPI_RW_SHIFT)		& PIO_SPI_RW_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_tid			<< PIO_SPI_TID_SHIFT)		& PIO_SPI_TID_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_reg			<< PIO_SPI_REG_SHIFT)		& PIO_SPI_REG_MASK);
			spicmd	|= ((((ioctl_arg_spi_t *)&argspi)->spi_data		<< PIO_SPI_DATA_SHIFT)		& PIO_SPI_DATA_MASK);
			iowrite32(spicmd, (bar[((ioctl_arg_spi_t *)&argspi)->card_id] + PCIE_CONTROL1));
			((ioctl_arg_spi_t *)&argspi)->drv_comp	= IOCTL_ARG_SUCCESS;
			
			if ( copy_to_user((ioctl_arg_spi_t *)arg, &argspi, sizeof(ioctl_arg_spi_t)) != 0 )
			{
				flag_is_ioctl_already_in_used	= 0;
				return (-EFAULT);
			}
//			printk("[openair][IOCTL] card%d: SPI WRITE REG(0x%02X) DATA(0x%02X) COMP(%d)\n", ((ioctl_arg_spi_t *)&argspi)->card_id, ((ioctl_arg_spi_t *)&argspi)->spi_reg, ((ioctl_arg_spi_t *)&argspi)->spi_data, ((ioctl_arg_spi_t *)&argspi)->drv_comp);
			break;

		case syrtem_INT_EXEC:
//			printk("[IOCTL]    syrtem_INT_EXEC\n");
			int_exec_cnt++;
			if ( copy_from_user((void *)&argint, (ioctl_arg_int_t *)arg, sizeof(ioctl_arg_int_t)) != 0 )
			{
				flag_is_ioctl_already_in_used	= 0;
				return (-EFAULT);
			}
			/* iowrite32 for INT CMD: [INT_FLAG][READ|WRITE][REGISTER][DATA] */
			intcmd	=  0x00000000;
			intcmd	|= ((SYR_PIO_CMD_INT		<< PIO_COMMAND_SHIFT)		& PIO_COMMAND_MASK);
			intcmd	|= ((SYR_PIO_INT_DATAREADY	<< PIO_INT_DREADY_SHIFT)	& PIO_INT_DREADY_MASK);
			intcmd	|= ((((ioctl_arg_int_t *)&argint)->int_rqrs		<< PIO_INT_REQRESP_SHIFT)	& PIO_INT_REQRESP_MASK);
			intcmd	|= ((((ioctl_arg_int_t *)&argint)->int_rw			<< PIO_INT_RW_SHIFT)		& PIO_INT_RW_MASK);
			intcmd	|= ((((ioctl_arg_int_t *)&argint)->int_tid			<< PIO_INT_TID_SHIFT)		& PIO_INT_TID_MASK);
			intcmd	|= ((((ioctl_arg_int_t *)&argint)->int_cnt			<< PIO_INT_CNT_SHIFT)		& PIO_INT_CNT_MASK);
//			printk("[IOCTL] iowrite32(0x%08X, bar[%d])\n", intcmd, ((ioctl_arg_int_t *)&argint)->card_id);
			spin_lock_irqsave(&mLock, state_flags);
			iowrite32(intcmd, (bar[((ioctl_arg_int_t *)&argint)->card_id] + PCIE_CONTROL1));
			spin_unlock_irqrestore(&mLock, state_flags);
			((ioctl_arg_int_t *)&argint)->drv_comp	= IOCTL_ARG_SUCCESS;
			
			if ( copy_to_user((ioctl_arg_int_t *)arg, &argint, sizeof(ioctl_arg_int_t)) != 0 )
			{
				flag_is_ioctl_already_in_used	= 0;
				return (-EFAULT);
			}
//			printk("[openair][IOCTL] card%d: INT WRITE REG(0x%02X) DATA(0x%02X) COMP(%d)\n", ((ioctl_arg_int_t *)&argint)->card_id, ((ioctl_arg_int_t *)&argint)->int_reg, ((ioctl_arg_int_t *)&argint)->int_data, ((ioctl_arg_int_t *)&argint)->drv_comp);
			break;

		default:
			printk("[IOCTL] openair_IOCTL unknown: basecmd = %i  (cmd=%X)\n", _IOC_NR(cmd), cmd);
			flag_is_ioctl_already_in_used	= 0;
			return (-EPERM);
			break;
	}
	flag_is_ioctl_already_in_used	= 0;
	return 0;
}
