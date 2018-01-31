#define __USE_GNU
#define _GNU_SOURCE
#include <stdio.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <stdint.h>
#include <sys/stat.h>
#include <fcntl.h>

// OAI includes

uint64_t cpuCyclesMicroSec;
static __inline__ uint64_t rdtsc(void) {
    uint64_t a, d;
    __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
    return (d<<32) | a;
}

#include "thread-pool.h"
int main(int argc, char* argv[]) {

    uint64_t deb=rdtsc();
    usleep(100000);
    cpuCyclesMicroSec=(rdtsc()-deb)/100000;
    printf("Cycles per µs: %lu\n",cpuCyclesMicroSec);
#define SEP "\t"
    printf("Frame" SEP "SubFrame" SEP "CodeBlock" SEP "RNTI"  SEP "Iterations" SEP
           "PreparationTime" SEP "StartTime" SEP "RunTime" SEP "ReturnTime" SEP "CumulSubFrame"
           SEP "CPUcore" SEP "ThreadID" "\n");

    mkfifo("/tmp/test-tcri",0666);
    int fd=open("/tmp/test-tcri", O_RDONLY);
    if ( fd == -1 ) {
        perror("open read mode trace file:");
        exit(1);
    }
    request_t doneRequest;
    int s=sizeof(request_t) -2*sizeof(void*);
    while ( 1 ) {
        if ( read(fd,&doneRequest, s) == s ) {
            union turboReqUnion id= {.p=doneRequest.id};
            doneRequest.processedBy[15]='\0';
            printf("%u" SEP "%u" SEP "%u" SEP "%u" SEP "%lu" SEP
                   "%lu" SEP "%lu" SEP "%lu" SEP "%lu" SEP
                   "%lu" SEP "%u" SEP "%s" "\n",
                   id.s.frame,
                   id.s.subframe,
                   id.s.codeblock,
                   id.s.rnti,
                   doneRequest.decodeIterations,
                   (doneRequest.creationTime-doneRequest.startUELoop)/cpuCyclesMicroSec,
                   (doneRequest.startProcessingTime-doneRequest.creationTime)/cpuCyclesMicroSec,
                   (doneRequest.endProcessingTime-doneRequest.startProcessingTime)/cpuCyclesMicroSec,
                   (doneRequest.returnTime-doneRequest.endProcessingTime)/cpuCyclesMicroSec,
                   doneRequest.cumulSubframe/cpuCyclesMicroSec,
                   doneRequest.coreId,
                   doneRequest.processedBy
                  );
        } else {
            printf("no measurements\n");
            sleep(1);
        }
    }
}
