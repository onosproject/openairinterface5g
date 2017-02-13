# compiles the example

OPENAIRTARGETS_DIR ?=../../../..

OPENAIROBJS += $(OPENAIRTARGETS_DIR)/ARCH/UED/USERSPACE/LIB/openair0_lib.o
CFLAGS += -DUED -I$(OPENAIRTARGETS_DIR)/ARCH/COMMON -I$(OPENAIRTARGETS_DIR)/ARCH/UED/USERSPACE/LIB -I$(OPENAIRTARGETS_DIR)/ARCH/UED/DEFS

example: example.o openair0_lib.o
	gcc -o $@ $(CFLAGS) -lm $(OPENAIROBJS) $<


clean:
	rm -f *.o *~
	rm -f example
