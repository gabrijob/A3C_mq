#ifndef ML_MODULE_HEADER
#define ML_MODULE_HEADER

#include <Python.h>

/* Module Global Variables*/
extern PyObject* pmodule;
extern wchar_t *program;
extern PyObject* agent;
extern PyObject* worker;
extern PyObject* p_server;

extern int ml_init_counts[8];

/* Function for importing Cython declarations */
void ml_agent_import(const char* argv_0, int argc); 

/* Function responsible for initializing the agent's environment */
int ml_agent_init(int controll, float qosmin, float qosmax, int split_ipqueue_size, char** split_ipqueue, int duration);

/* Function to control the cache value and flow status */
int ml_caching (void * API_puller, int msgsize, int msgperbatch, int qosmin, int nb_sending_sockets, char** split_ipqueue, int controll, void ** split_sockets_state, void ** split_sockets, float second,int qosmax,int loss, int window,
int ttpersocket, int ttpersocketsec, int *flagState, float *RecMQTotal, float *avgRecMQTotal, float *RecSparkTotal, uint64_t *ackSent, int *RecTotal, int *cREC, int *cDELAY, int *cTIMEP, float *last_second,
float *global_avg_spark, float *lastonespark, float *state, float *qosbase, int* vector, float maxth, float measure, int input_hanger_size);

/* Function responsible for closing th agent's environment */
int ml_agent_finish(int controll); 

#endif
