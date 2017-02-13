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

/** irq.c
     - IRQ Handler: IRQ from Leon to PCIe/Kernel: exmimo_irq_handler
     - sends received packets to userspace

     - send command from PC to Leon and trigger IRQ on Leon using CONTROL1 register
     - commands are defined in $OPENAIR0/express-mimo/software/pcie_interface.h

     - added: pass card_id as parameter to tasklet and irq handler

  Authors:
      Raymond Knopp <raymond.knopp@eurecom.fr>
      Matthias Ihmig <matthias.ihmig@mytum.de>, 2011, 2013
  */

#include <linux/pci.h>
#include <linux/interrupt.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include <linux/swab.h>
//#include <linux/spinlock.h>

//#include <asm/system.h>

#include "openair_device.h"
#include "extern.h"

#include "syr_pio.h"

typedef struct tasklet_args_s
{
	unsigned long	ta_cardid;
	unsigned int	ta_irqcmd;
	unsigned int	ta_pciectrl;
	unsigned char	ta_in_progress;
} tasklet_args_t;

//volatile tasklet_args_t	tasklet_arguments;
tasklet_args_t	tasklet_arguments;

void pcie_printk(int card_id);
void openair_do_tasklet (unsigned long ta_args);

DECLARE_TASKLET(openair_tasklet, openair_do_tasklet, (unsigned long)&tasklet_arguments);

//unsigned int					openair_bh_cnt;
volatile int					get_frame_done					= 0;
//volatile unsigned int			syrtem_pio_counter				= 0;

volatile unsigned long long		openair_irq						= 0;
volatile unsigned long long		openair_irq_not_match			= 0;
volatile unsigned long long		ioread32_cnt					= 0;
volatile unsigned long long		ioread32_cnt_loop				= 0;
volatile unsigned long long		irqval_not_asserted_cnt			= 0;
volatile unsigned long long		irqval_asserted_cnt				= 0;
volatile unsigned long long		tasklet_argument_done			= 0;
volatile unsigned long long		tasklet_argument_missed			= 0;

// deprecated
//spinlock_t					mLock							= SPIN_LOCK_UNLOCKED;
DEFINE_SPINLOCK(mLock);

irqreturn_t openair_irq_handler(int irq, void *cookie)
{
	unsigned int	irqval;
	unsigned int	irqcmd				= EXMIMO_NOP;
	unsigned long	card_id;	//		= (unsigned long) cookie;
	unsigned char	is_openair_irq		= 0;
	unsigned long   ioread32_cnt_local	= 0;
	unsigned long	state_flags;

	openair_irq++;

	// find card_id. cookie is set by request_irq and static, so we will always find it
	for (card_id = 0; card_id < MAX_CARDS; card_id++)
	{
		if (pdev[card_id] == cookie)
		{
			is_openair_irq	= 1;
			break;
		}
	}
	if (!is_openair_irq)
	{
		openair_irq_not_match++;
		return IRQ_NONE;
	}

	spin_lock_irqsave(&mLock, state_flags);
	// get AHBPCIE interrupt line (bit 7) to determine if IRQ was for us from ExMIMO card, or from a different device
	// reading CONTROL0 will also clear this bit and the LEON-to-PC IRQ line
	irqval	= ioread32(bar[card_id] + PCIE_CONTROL0);
	barrier();
	rmb();
	wmb();
	irqcmd	= ioread32(bar[card_id] + PCIE_CONTROL2);
	barrier();
	rmb();
	wmb();
	while ( (ioread32(bar[card_id] + PCIE_CONTROL0) & 0x80) && (ioread32_cnt_local < 1000) )
	{
		barrier();
		rmb();
		wmb();
//		irqcmd	= ioread32(bar[card_id] + PCIE_CONTROL2);
//		rmb();
		ioread32_cnt_local++;
	}
	if (ioread32_cnt_local)
		ioread32_cnt_loop++;
	if (ioread32_cnt_local > ioread32_cnt)
		ioread32_cnt	= ioread32_cnt_local;

	barrier();
	rmb();
	wmb();
	spin_unlock_irqrestore(&mLock, state_flags);

	if (irqval & 0x80)	// IRQ_ASSERTED
	{
		irqval_asserted_cnt++;

		switch(irqcmd)
		{
			case GET_FRAME_DONE:
				get_frame_done = 1;
				break;
			case PCI_PRINTK:
				if (!tasklet_arguments.ta_in_progress)
				{
					tasklet_arguments.ta_cardid			= card_id;
					tasklet_arguments.ta_irqcmd			= irqcmd;
					tasklet_arguments.ta_pciectrl		= PCIE_CONTROL2;
					tasklet_arguments.ta_in_progress	= 1;
					tasklet_schedule(&openair_tasklet);
//					openair_bh_cnt++;
					tasklet_argument_done++;
				}
				else
				{
					tasklet_argument_missed++;
				}
				break;
			case EXMIMO_NOP:
			default:
				break;
		}
		iowrite32(EXMIMO_NOP, bar[card_id] + PCIE_CONTROL2);

		return IRQ_HANDLED;
	}
	else					// IRQ_NOT_ASSERTED : CTRL0.bit7 is no set -> IRQ is not from ExMIMO i.e. not for us
	{
		irqval_not_asserted_cnt++;
		return IRQ_NONE;
	}
}

void openair_do_tasklet (unsigned long ta_args)
{
/*	printk("((tasklet_args_t *)ta_args)->ta_cardid=%ld, ((tasklet_args_t *)ta_args)->ta_irqcmd=0x%02x\n",
		((tasklet_args_t *)ta_args)->ta_cardid,
		((tasklet_args_t *)ta_args)->ta_irqcmd );
*/
	switch(((tasklet_args_t *)ta_args)->ta_irqcmd)
	{
		case PCI_PRINTK:
			// printk("Got PCIe interrupt for printk ...\n");
			pcie_printk((int)((tasklet_args_t *)ta_args)->ta_cardid);
			break;
		case GET_FRAME_DONE:
		case EXMIMO_NOP:
		default:
/*			if ( (irqcmd >= SYR_TEST) && (irqcmd <= (SYR_TEST+0x2000)) )
			{
				syrtem_pio_counter++;
				if ( (syrtem_pio_counter%1000) == 0)
					printk("[openair][IRQ tasklet] : Got %d PIO from LEON3\n", syrtem_pio_counter);
				if (syrtem_pio_counter >= 0x2000)
					printk("[openair][IRQ tasklet] : Got 8192 PIO from LEON3\n");
			}
			else if ( (irqcmd == SYR_TEST_NOK) || (irqcmd == SYR_TEST_OK) )	// End of PIO test
			{
				if (irqcmd == SYR_TEST_NOK)
				{
					printk("[openair][IRQ tasklet] : PIO Test failed\n");
				}
				else
				{
					printk("[openair][IRQ tasklet] : PIO Test ended successfully\n");
				}
				printk("[openair][IRQ tasklet] :   irqval_0x08_cnt               =%lld\n	\
						[openair][IRQ tasklet] :   board_swrev_cmdreg_cnt        =%lld\n	\
						[openair][IRQ tasklet] :   nop_and_cookie_cmdreg_cnt     =%lld\n	\
						[openair][IRQ tasklet] :   nop_and_cookie_cmdreg_else_cnt=%lld\n	\
						[openair][IRQ tasklet] :   board_swrev_cmdreg_else_cnt   =%lld\n	\
						[openair][IRQ tasklet] :   irqval_0x08_else_cnt          =%lld\n",
						irqval_0x08_cnt,
						board_swrev_cmdreg_cnt,
						nop_and_cookie_cmdreg_cnt,
						nop_and_cookie_cmdreg_else_cnt,
						board_swrev_cmdreg_else_cnt,
						irqval_0x08_else_cnt);
			}
			else
			{
				printk("[openair][IRQ tasklet] : Got unknown PCIe cmd: card_id = %li, irqcmd(CONTROL1) = %i (0x%X)\n", card_id, irqcmd, irqcmd);
			}
*/			break;
	}
	tasklet_arguments.ta_in_progress	= 0;
}

void pcie_printk(int card_id)
{
	char			*buffer = exmimo_pci_kvirt[card_id].printk_buffer_ptr;
	unsigned int	len		= ((unsigned int *)buffer)[0];
	unsigned int	off		= 0;
	unsigned int	i;
	unsigned char	*dword;
	unsigned char	tmp;

	//printk("In pci_fifo_printk : buffer %p, len %d: \n",buffer,len);
	printk("[LEON card%d]: ", card_id);

	if (len < 1024)
	{
		if ( (len&3) > 0 )
			off	= 1;

		for (i = 0; i < (off + (len >> 2)); i++)
		{
			dword		= &((unsigned char *)buffer)[(1 + i) << 2];
			tmp			= dword[3];
			dword[3]	= dword[0];
			dword[0]	= tmp;
			tmp			= dword[2];
			dword[2]	= dword[1];
			dword[1]	= tmp;
		}

		for (i = 0; i < len; i++)
		{
			printk( "%c", ((char*)&buffer[4])[i] );
		}
	}
}
