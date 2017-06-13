CC=gcc
CFLAGS=-Wall -g -pthread -I.. -I../logger
CFLAGS += -O3 -ffast-math -fomit-frame-pointer

OBJS=stdout.o textlist.o xy.o tti.o time.o ticktime.o scrolltti.o

view.a: $(OBJS)
	ar cr view.a $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -f *.a *.o
