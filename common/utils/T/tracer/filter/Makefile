CC=gcc
CFLAGS=-Wall -g -pthread -I..
CFLAGS += -O3 -ffast-math -fomit-frame-pointer

OBJS=filter.o

filter.a: $(OBJS)
	ar cr filter.a $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -f *.a *.o
