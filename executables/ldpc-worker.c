#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "thpool.h"
#define DEFAULT_PORT 8000
#define MAXLIN 4096


void dumytask1(){
	printf("Thread #%u working on task1\n", (int)pthread_self());
}

void dummytask2(){
	printf("Thread #%u working on task2\n", (int)pthread_self());
}

int main(int argc,char **argv)
{
    int socket_fd,connect_fd,N_sockets;
    struct sockaddr_in servaddr;
    char buff[4096]; //

    int n;
    int N = argv[2];
    //Init
    


    if((socket_fd=socket(AF_INET,SOCK_STREAM,0))==-1)
    {
        printf("create socket error:%s(errno :%d)\n",strerror(errno),errno);
        exit(0);
    }

    memset(&servaddr,0,sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr=htonl(INADDR_ANY);//IP Address as INADDR_ANY,get IP automaticcaly 
    servaddr.sin_port=htons(DEFAULT_PORT);
    //Port set as DEFAULT_PORT

    //Bind
    if(bind(socket_fd,(struct sockaddr*)&servaddr,sizeof(servaddr))==-1)
    {
        printf("bind socket error:%s(errno:%d)\n",strerror(errno),errno);
        exit(0);
    }

    //listen to client connections，10 being the maximum listening number
    if(listen(socket_fd,10)==-1)
    {
        printf("listen socket error:%s(errno:%d)\n",strerror(errno),errno);
        exit(0);
    }

    printf("======waiting for client's request=====\n");


    while(1)
    {
        if((connect_fd=accept(socket_fd,(struct sockaddr*)NULL,NULL))==-1){
            printf("accept socket error :%s(errno:%d)\n",strerror(errno),errno);
            continue;
        }

  		// select free buffers for, shared mem between this thread and 
        // N worker threads, using thpool_init(N)
		puts("Making threadpool with N worker threads");
		threadpool thpool = thpool_init(N);

        n=recv(connect_fd,buff,MAXLIN,0);

        // dispatch to worker threads, use cond_signal

        if(!fork()){
            if(send(connect_fd,"hello man\n",26,0)==-1)
            perror("send error");
            close(connect_fd);
            exit(0);
        }
	    buff[n]='\n';
	    
	    printf("recv client : %s\n",buff);
	    puts("Adding tasks to threadpool");
		int i;
		for (i=0; i<20; i++){
			thpool_add_work(thpool, (void*)task1, NULL);
			thpool_add_work(thpool, (void*)task2, NULL);
		};
    close(connect_fd);
    }
    close(socket_fd);

	puts("Killing threadpool");
	thpool_destroy(thpool);
}