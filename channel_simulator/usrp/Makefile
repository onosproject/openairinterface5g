CC=gcc
CPP=g++
CFLAGS=-Wall -g -pthread -march=native -O3 -ffast-math

PROG=usrp
OBJS=main.o usrp.o ../utils.o

$(PROG): $(OBJS)
	$(CPP) $(CFLAGS) -o $@ $^ -luhd -lboost_system

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cc
	$(CPP) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(PROG) *.o core
