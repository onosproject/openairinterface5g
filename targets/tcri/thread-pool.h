#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <stdbool.h>
#include <openair2/COMMON/platform_types.h>

enum request_t {
    DECODE,
    ENCODE
};

struct turboReqId {
    uint16_t rnti;
    uint16_t frame;
    uint8_t  subframe;
    uint8_t  codeblock;
    uint16_t spare;
} __attribute__((packed));

union turboReqUnion {
    struct turboReqId s;
    uint64_t p;
};

typedef struct request {
    uint64_t id;
    enum request_t type;
    uint64_t creationTime;
    uint64_t startProcessingTime;
    uint64_t endProcessingTime;
    uint64_t returnTime;
    uint64_t decodeIterations;
    int coreId;
    char processedBy[16];
    struct request* next;
    void * data __attribute__((aligned(32)));
} request_t;

struct one_thread {
    pthread_t  threadID;
    int id;
    int coreID;
    char name[256];
    struct thread_pool * pool;
    struct one_thread* next;
};

typedef struct thread_pool {
    int activated;
    pthread_mutex_t lockRequests;
    pthread_cond_t  notifRequest;
    pthread_mutex_t lockReportDone;
    pthread_cond_t  notifDone;
    request_t* oldestRequests;
    request_t* newestRequests;
    request_t* doneRequests;
    int notFinishedJobs;
    int traceFd;
    int dummyTraceFd;
    uint64_t cpuCyclesMicroSec;
    int nbThreads;
    bool restrictRNTI;
    struct one_thread * allthreads;
} tpool_t;

void init_tpool(char*,tpool_t* );
request_t * createRequest(enum request_t type,int size);
void freeRequest(request_t* request);
int add_request(request_t* request, tpool_t * tp);
request_t * searchRNTI(tpool_t*,  rnti_t rnti);

#endif
