CC=gcc
CFLAGS=-Wall -g -pthread

CFLAGS += -O3 -ffast-math -march=native

PROG=channel_simulator
OBJS=main.o channel_simulator.o connection_manager.o utils.o

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(PROG) *.o core
