CCC = gcc

#include /home/albatross/frobinet/workspaceOAItrunk/syrpciekdm/common/utils/Makefile.inc

KERNEL_MAIN_TYPE:=$(shell echo `uname -r | cut -d. -f-2  | tr "." "_"`)
export KERNEL_MAIN_TYPE

KERNEL_DIR:=/usr/src/linux-headers-3.17.0-031700-lowlatency
#EXTRA_CFLAGS = -ggdb -D__KERNEL__ -DMODULE -D_LOOSE_KERNEL_NAMES -I$(KERNEL_DIR)/build/include -I$(KERNEL_DIR)/build/include/asm/mach-default -include $(KERNEL_DIR)/build/include/linux/autoconf.h 
EXTRA_CFLAGS = -ggdb -D__KERNEL__ -DMODULE -D_LOOSE_KERNEL_NAMES -I$(KERNEL_DIR)/include -I$(KERNEL_DIR)/include/asm-generic -include $(KERNEL_DIR)/include/linux/autoconf.h 

EXTRA_CFLAGS += -I$(PWD)/../../DEFS

ccflags-y= $(CFLAGS) $(EXTRA_CFLAGS)
CFLAGS=

obj-m += openair_rf.o

openair_rf-objs += module_main.o irq.o fileops.o exmimo_fw.o

all:
	make -C $(KERNEL_DIR) M=$(PWD) modules

clean:
	rm -rf *.o
	make -C $(KERNEL_DIR) M=$(PWD) clean

