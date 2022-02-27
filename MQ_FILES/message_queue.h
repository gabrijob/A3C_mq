#ifndef MQ
#define MQ

#include <math.h>
#include <stdio.h>
#include <zmq.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <time.h>
#include <inttypes.h>
#include <omp.h>
#include <string.h>
#include <inttypes.h>
#include <ml_module.h>

#define ONE_SECOND           1000


//Sockets status
#define SOCKET_READY_TO_SEND 2
#define SOCKET_WAITING_ACK   4

//buffer
typedef struct item{
    zmq_msg_t * input_message;
    struct item* next;
} hanger;

hanger*  input_hanger         = NULL;

int      nb_input_messages_waiting_to_be_polled = 0;
int      need_output_socket   = 0;
int      input_hanger_size    = 0;

uint64_t last_second_received = 0;
uint64_t last_second_sent     = 0;

#endif

