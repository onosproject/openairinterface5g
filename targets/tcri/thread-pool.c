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
#include <stdbool.h>
// OAI includes
#include <assertions.h>
#include <log.h>
#include "PHY/TOOLS/time_meas.h"
#include "PHY/CODING/defs.h"
#include "PHY/CODING/extern.h"
#include <thread-pool.h>

#ifdef DEBUG
#define THREADINIT   PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP
#else
#define THREADINIT   PTHREAD_MUTEX_INITIALIZER
#endif


request_t * createRequest(enum request_t type,int size) {
    request_t* request;
    AssertFatal( (request = (request_t*)aligned_alloc(32,sizeof(request_t)+size)) != NULL,"");
    memset(request,0,sizeof(request_t));
    request->id = 0;
    request->type=type;
    request->next = NULL;
    request->creationTime=rdtsc();
    request->data=(void*)(request+1);
    return request;
}

void freeRequest(request_t* request) {
    //printf("freeing: %ld, %p\n", request->id, request);
    free(request);
}

volatile int ii=0;
int add_request(request_t* request, tpool_t * tp) {
    mutexlock(tp->lockRequests);
    if (tp->oldestRequests == NULL)
        tp->oldestRequests = request;
    else {
        AssertFatal(tp->newestRequests != NULL, "");
        tp->newestRequests->next = request;
    }
    tp->newestRequests = request;
    mutexlock(tp->lockReportDone);
    tp->notFinishedJobs++;
    mutexunlock(tp->lockReportDone);
    condbroadcast(tp->notifRequest);
    mutexunlock(tp->lockRequests);
    return 0;
}

int add_requests(uint64_t request_num, tpool_t * tp) {
    request_t* request;
    int nbToAdd=((uint32_t)lrand48())%20+1;
    mutexlock(tp->lockRequests);
    for (int i=0; i<nbToAdd; i++) {
        // simulate request
        request=createRequest(DECODE,sizeof(turboDecode_t));
        union turboReqUnion id= {.s={request_num,1000,i*10,111,222}};
        request->id= id.p;
        turboDecode_t * rdata=(turboDecode_t *) request->data;
        rdata->function=phy_threegpplte_turbo_decoder8;
        rdata->Kr=6144;
        rdata->iind=0; // not used, OAI code need cleanup!!!
        rdata->Fbits=0;
        rdata->maxIterations=6;
        if (tp->oldestRequests == NULL)
            tp->oldestRequests = request;
        else
            tp->newestRequests->next = request;
        tp->newestRequests = request;
    }

    mutexlock(tp->lockReportDone);
    tp->notFinishedJobs+=nbToAdd;
    mutexunlock(tp->lockReportDone);
    condbroadcast(tp->notifRequest);
    mutexunlock(tp->lockRequests);
    return nbToAdd;
}

request_t * get_request(tpool_t * tp, uint16_t threadID ) {
    int nb=0;
    request_t* r=tp->oldestRequests;
    while (r!=NULL) {
        nb++;
        r=r->next;
    }
    request_t* request=tp->oldestRequests;
    if (request == NULL)
        return NULL;

    if ( tp->restrictRNTI ) {
        request_t** start=&tp->oldestRequests;
        request = NULL;
        while(*start!=NULL && request==NULL) {
            union turboReqUnion id= {.p=(*start)->id};
            if ( id.s.rnti % tp->nbThreads ==  threadID ) {
                request=*start;
                *start=(*start)->next;
            } else
                start=&((*start)->next);
        }
    } else
        tp->oldestRequests = request->next;

    if ( tp->oldestRequests == NULL)
        tp->newestRequests=NULL;

    int nnb=0;
    r=tp->oldestRequests;
    while (r!=NULL) {
        nnb++;
        r=r->next;
    }
    /*
    if ( ! ( nb == nnb && request == NULL))
      printf("getr:was=%d,is=%d,gotit=%p\n",nb,nnb,request);
    */
    return request;
}

request_t * searchRNTI(tpool_t * tp, rnti_t rnti) {
    request_t * result=NULL;
    request_t ** start=&tp->oldestRequests;
    while(*start!=NULL && result==NULL) {
        union turboReqUnion id= {.p=(*start)->id};
        if ( id.s.rnti == rnti ) {
            result=*start;
            *start=(*start)->next;
            if ( tp->oldestRequests == NULL)
                tp->newestRequests=NULL;
        } else
            start=&((*start)->next);
    }
    return result;
}



void process_request(request_t* request) {
    //printf("S:%s...",request->type==DECODE?"D":"E");
    switch(request->type) {
    case DECODE : {
        time_stats_t oaitimes[7];
        turboDecode_t * rdata=(turboDecode_t*) request->data;
        rdata->decodeIterations=rdata->function(rdata->soft_bits+96,
                                                rdata->decoded_bytes,
                                                rdata->Kr,
                                                f1f2mat_old[rdata->iind*2],
                                                f1f2mat_old[(rdata->iind*2)+1],
                                                rdata->maxIterations,
                                                rdata->nbSegments == 1 ? CRC24_A: CRC24_B,
                                                rdata->Fbits,
                                                oaitimes+0,
                                                oaitimes+1,
                                                oaitimes+2,
                                                oaitimes+3,
                                                oaitimes+4,
                                                oaitimes+5,
                                                oaitimes+6);
    };
    break;
    case ENCODE :  {
        turboEncode_t * rdata=(turboEncode_t*) request->data;
        memset(rdata->output,LTE_NULL,TURBO_SIMD_SOFTBITS);
        threegpplte_turbo_encoder(rdata->input,
                                  rdata->Kr_bytes,
                                  rdata->output+96,//&dlsch->harq_processes[harq_pid]->d[r][96],
                                  rdata->filler,
                                  f1f2mat_old[rdata->iind*2],   // f1 (see 36121-820, page 14)
                                  f1f2mat_old[(rdata->iind*2)+1]  // f2 (see 36121-820, page 14)
                                 );
    };
    break;
    default:
        AssertFatal(false,"");
    }
    //printf("..End\n");
}

void handle_request(tpool_t * tp, request_t* request) {
    request->startProcessingTime=rdtsc();
    process_request(request);
    request->endProcessingTime=rdtsc();
    mutexlock(tp->lockReportDone);
    tp->notFinishedJobs--;
    request->next=tp->doneRequests;
    tp->doneRequests=request;
    condsignal(tp->notifDone);
    mutexunlock(tp->lockReportDone);

}

void* one_thread(void* data) {
    struct  one_thread * myThread=(struct  one_thread *) data;
    struct  thread_pool* tp=myThread->pool;

    // configure the thread core assignment
    // TBD: reserve the core for us exclusively
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(myThread->coreID, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    //Configure the thread scheduler policy for Linux
    struct sched_param sparam= {0};
    sparam.sched_priority = sched_get_priority_max(SCHED_RR);
    pthread_setschedparam(pthread_self(), SCHED_RR, &sparam);

    // set the thread name for debugging
    sprintf(myThread->name,"Tcodec_%d",myThread->coreID);
    pthread_setname_np(pthread_self(), myThread->name );

    // Infinite loop to process requests
    do {
        mutexlock(tp->lockRequests);

        request_t* request = get_request(tp, myThread->id);
        if (request == NULL) {
            condwait(tp->notifRequest,tp->lockRequests);
            request = get_request(tp, myThread->id);
        }

        mutexunlock(tp->lockRequests);

        if (request!=NULL) {
            strncpy(request->processedBy,myThread->name, 15);
            request->coreId=myThread->coreID;
            handle_request(tp, request);
        }

    } while (true);
}

void init_tpool(char * params,tpool_t * pool) {
    mkfifo("/tmp/test-tcri",0666);
    pool->dummyTraceFd=open("/tmp/test-tcri", O_RDONLY| O_NONBLOCK);
    if ( pool->dummyTraceFd == -1 ) {
        perror("open read mode trace file:");
        exit(1);
    }
    pool->traceFd=open("/tmp/test-tcri", O_WRONLY|O_APPEND|O_NOATIME|O_NONBLOCK);
    if ( pool->traceFd == -1 ) {
        perror("open trace file:");
        exit(1);
    }

    //Configure the thread scheduler policy for Linux
    struct sched_param sparam= {0};
    sparam.sched_priority = sched_get_priority_max(SCHED_RR)-1;
    pthread_setschedparam(pthread_self(), SCHED_RR, &sparam);
    pool->activated=true;
    mutexinit(pool->lockRequests);
    condinit (pool->notifRequest);
    pool->notifCount=0;
    mutexinit(pool->lockReportDone);
    condinit (pool->notifDone);
    pool->oldestRequests=NULL;
    pool->newestRequests=NULL;
    pool->doneRequests=NULL;
    pool->notFinishedJobs=0;
    pool->allthreads=NULL;
    char * saveptr, * curptr;
    pool->nbThreads=0;
    pool->restrictRNTI=false;
    curptr=strtok_r(params,",",&saveptr);
    while ( curptr!=NULL ) {
        if (curptr[0] == 'u' || curptr[0] == 'U') {
            pool->restrictRNTI=true;
        } else if ( curptr[0]>='0' && curptr[0]<='9' ) {
            struct one_thread *tmp=pool->allthreads;
            pool->allthreads=(struct one_thread *)malloc(sizeof(struct one_thread));
            pool->allthreads->next=tmp;
            printf("create a thread for core %d\n", atoi(curptr));
            pool->allthreads->coreID=atoi(curptr);
            pool->allthreads->id=pool->nbThreads;
            pool->allthreads->pool=pool;
            pthread_create(&pool->allthreads->threadID, NULL, one_thread, (void*)pool->allthreads);
            pool->nbThreads++;
        } else if (curptr[0] == 'n' || curptr[0] == 'N') {
            pool->activated=false;
        } else
            printf("Error in options for thread pool: %s\n",curptr);
        curptr=strtok_r(NULL,",",&saveptr);
    }

    if (pool->activated && pool->nbThreads==0) {
        printf("No servers created in the thread pool, exit\n");
        exit(1);
    }

    uint64_t deb=rdtsc();
    usleep(100000);
    pool->cpuCyclesMicroSec=(rdtsc()-deb)/100000;
    printf("Cycles per µs: %lu\n",pool->cpuCyclesMicroSec);

}


void displayList(request_t*start, request_t*end) {
    int n=0;
    while(start!=NULL) {
        n++;
        union turboReqUnion id= {.p=start->id};
        printf("rnti:%u frame:%u-%u codeblock:%u\n",
               id.s.rnti,
               id.s.frame,
               id.s.subframe,
               id.s.codeblock);
        if ( start->next==NULL && start!=end)
            printf("Error of end pointer");
        start=start->next;
    }
    printf("End of list: %d elements\n",n);
}

#ifdef TESTMAIN
#include "PHY/CODING/lte_interleaver.h"
#include "PHY/CODING/lte_interleaver2.h"

int main(int argc, char* argv[]) {

    if (argc<2) {
        printf("Usage: %s core,core,...\n",argv[0]);
        exit(1);
    }

    // configure the thread core assignment: client thread on core 0
    // TBD: reserve the core for us exclusively
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    tpool_t  pool;
    init_tpool(argv[1], &pool);
    //initialize turbo decoder tables
    init_td8();

    uint64_t i=1;
    // Test the lists
    srand48(time(NULL));
    int nbRequest=add_requests(i, &pool);
    printf("These should be: %d elements in the list\n",nbRequest);
    displayList(pool.oldestRequests, pool.newestRequests);
    // remove in middle
    request_t *req106=searchRNTI(&pool, 106);
    if (req106) {
        union turboReqUnion id= {.p=req106->id};
        printf("Removed: rnti:%u frame:%u-%u codeblock:%u, check it\n",
               id.s.rnti,
               id.s.frame,
               id.s.subframe,
               id.s.codeblock);
        freeRequest(req106);
    }  else
        printf("no rnti 106\n");
    displayList(pool.oldestRequests, pool.newestRequests);
    request_t *reqlast=searchRNTI(&pool, 100+nbRequest-1);
    if (reqlast) {
        printf("Removed last item, check it\n");
        freeRequest(reqlast);
    }  else
        printf("tried to removed from empty list\n");
    displayList(pool.oldestRequests, pool.newestRequests);
    printf("Remove all jobs\n");
    while(pool.oldestRequests!=NULL)
        get_request(&pool,0);
    printf("List should be empty now\n");
    displayList(pool.oldestRequests, pool.newestRequests);

    sleep(1);
    mutexlock(pool.lockReportDone);
    pool.notFinishedJobs=0;
    pool.doneRequests=NULL;
    mutexunlock(pool.lockReportDone);

    while (1) {
        uint64_t now=rdtsc();
        /* run a loop that generates a lot of requests */
        AssertFatal(pool.notFinishedJobs==0,"");
        int n=add_requests(i, &pool);
        printf("Added %d requests\n",n);

        /*
            // The main thread also process the queue
            mutexlock(pool.lockRequests);
            request_t* request= NULL;
            while ( (request=get_request(&pool,0)) != NULL ) {
                mutexunlock(pool.lockRequests);
                strcpy(request->processedBy,"MainThread");
                handle_request(&pool, request);
                mutexlock(pool.lockRequests);
            }
            mutexunlock(pool.lockRequests);
        */

        // Wait all other threads finish to process
        mutexlock(pool.lockReportDone);
        while ( pool.notFinishedJobs > 0 ) {
            condwait(pool.notifDone,pool.lockReportDone);
        }
        mutexunlock(pool.lockReportDone);

        int i=0;
        for (request_t* ptr=pool.doneRequests; ptr!=NULL; ptr=ptr->next)  {
            i++;
            //printf("in return: %ld, %p\n", ptr->id, ptr);
        }
        AssertFatal(i==n,"%d/%d\n",i,n);

        while (pool.doneRequests!=NULL) {
            pool.doneRequests->returnTime=rdtsc();
            if(write(pool.traceFd,pool.doneRequests,sizeof(request_t)- 2*sizeof(void*))) {};
            request_t* tmp=pool.doneRequests;
            pool.doneRequests=pool.doneRequests->next;
            freeRequest(tmp);
        }

        printf("Requests %d Done in %ld µsec\n",i, (rdtsc()-now)/pool.cpuCyclesMicroSec);
        i++;
    };
    return 0;
}
#endif
