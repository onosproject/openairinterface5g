# compiles the example

OPENAIRTARGETS_DIR ?=../../../..

#EXT	= 160216
EXT		= 

OPENAIROBJS += $(OPENAIRTARGETS_DIR)/ARCH/UED/USERSPACE/CALIB$(EXT)/openair0_lib.o
OPENAIROBJS += $(OPENAIRTARGETS_DIR)/ARCH/UED/USERSPACE/CALIB$(EXT)/lime_spi_cmd.o
OPENAIROBJS += $(OPENAIRTARGETS_DIR)/ARCH/UED/USERSPACE/CALIB$(EXT)/lime_reg_cmd.o
OPENAIROBJS += $(OPENAIRTARGETS_DIR)/ARCH/UED/USERSPACE/CALIB$(EXT)/lime_cfg_cmd.o
OPENAIROBJS += $(OPENAIRTARGETS_DIR)/ARCH/UED/USERSPACE/CALIB$(EXT)/lime_cal_cmd.o

CFLAGS += -DUED -I$(OPENAIRTARGETS_DIR)/ARCH/COMMON -I$(OPENAIRTARGETS_DIR)/ARCH/UED/USERSPACE/CALIB$(EXT) -I$(OPENAIRTARGETS_DIR)/ARCH/UED/DEFS

all: lime_calib

lime_calib: lime_calib.o openair0_lib.o lime_spi_cmd.o lime_reg_cmd.o lime_cfg_cmd.o lime_cal_cmd.o
	gcc -o $@ $(CFLAGS) $(OPENAIROBJS) -lm $<

clean:
	rm -f *.o *~
	rm -f lime_calib
