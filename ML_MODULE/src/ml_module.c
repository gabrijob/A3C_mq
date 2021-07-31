#ifndef ML_MODULE_CODE
#define ML_MODULE_CODE

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

#include "SAQNAgent.h"
#include "A3CProcesses.h"
#include "ml_module.h"


#define SAQN_RUN

PyObject* pmodule;
wchar_t *program;
PyObject* agent = NULL;
PyObject* worker = NULL;
PyObject* p_server = NULL;

float last_global_avg=0;
float last_local_thpt=0;
float throughput_var=0;
//FILE* saqn_time_csv=NULL; 


/* Agent's code to run on the master MQ */
void ml_agent_master_action(int msgsize, int split_ipqueue_size, void ** split_sockets_state, void ** split_sockets, int qosmin, int qosmax, float second,
float *last_second, int flagState, float RecTotal, float RecMQTotal, float avgRecMQTotal, float RecSparkTotal, int cDELAY, int cTIMEP,
float *global_avg_spark, float *state, int *qosbase, int* vector, float *lastonespark, float maxth, float measure); 

/* Agent's code to run on the worker MQ */
void ml_agent_worker_action(int controll, int msgsize, void ** split_sockets_state, void ** split_sockets, float second,
int ttpersocket, int ttpersocketsec, int input_hanger_size, int cDELAY, int cTIMEP, float state, int *qosbase, int* vector, float *last_second);


void free_data (void *data, void *hint){
    free (data);
}


void ml_agent_init(int controll, float qosmin, float qosmax, int split_ipqueue_size, char** split_ipqueue, int duration) {

   if (controll == 0) {
	printf("\nControll Agent Start \n");
#ifdef SAQN_RUN
	printf("\nStarting Agent\n");
	float start_state[8] = {0.0,0.0,0.0,0.0,0,0,0,0};
	agent = createAgent(start_state, qosmin, qosmax);
#endif
#ifdef A3C_RUN
	printf("\nStarting Parameter Server\n");
	for(int i=0; i < split_ipqueue_size; i++) {
		printf("\n%s", split_ipqueue[i]);
	}
	p_server = parameter_server_proc(duration + 60, split_ipqueue, split_ipqueue_size);
#endif

#ifdef RANDOM_ACTION
	time_t t;
	srand((unsigned) time(&t));
#endif
    }	
#ifdef A3C_RUN
    printf("\nStarting Worker %d\n", controll);
    float start_state[8] = {0.0,0.0,0.0,0,0,0,0,0};
    worker = create_worker(start_state,0,split_ipqueue, split_ipqueue_size);
#endif

}


void ml_agent_master_action(int msgsize, int split_ipqueue_size, void ** split_sockets_state, void ** split_sockets, int qosmin, int qosmax, float second,
float *last_second, int flagState, float RecTotal, float RecMQTotal, float avgRecMQTotal, float RecSparkTotal, int cDELAY, int cTIMEP,
float *global_avg_spark, float *state, int *qosbase, int* vector, float *lastonespark, float maxth, float measure ) {

	if (flagState == 1 && (second != *last_second)) {
#ifndef A3C_RUN
		//printf("\nNot A3C master\n");
			
		//throughput mean
		if ((((float)RecTotal*msgsize)/1024/1024)/second > 0){
		 	*global_avg_spark = (((float)RecTotal*msgsize)/1024/1024)/second;
		}


		// Gabriel's TCC - A Bachpessure-based
		//RANDOM_ACTION - off  	
		#ifndef RANDOM_ACTION
			//saqn_time_csv = fopen("/tmp/SAQN_TIME.csv", "a");
				
			clock_t start, end;
			throughput_var = *global_avg_spark - last_global_avg;
			/* Get new action from agent */
			//[total_thpt, thpt_var, proc_t, sche_t, msgs_to_spark, msgs_in_gb, ready_mem, spark_thresh]
			float curr_env_state[8] = {*global_avg_spark, throughput_var, cDELAY , cTIMEP, RecSparkTotal, RecMQTotal, *state, *qosbase};	
			start = clock();
			float action = infer(agent, curr_env_state);
			//printf("\nNew action is %f. ", action);
			end = clock();

			*qosbase = *qosbase + action;
			double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
			//printf("Agent took %f seconds to execute \n", cpu_time_used);
			//printf("\n Second is: %f || Last second is: %f", second, *last_second);
			//fprintf(saqn_time_csv, "%f \t,\t %f\n", second, cpu_time_used);
			*last_second = second;
			last_global_avg = *global_avg_spark;
		
			//fflush(saqn_time_csv);
			//fclose(saqn_time_csv);
		#else
			*qosbase = rand() % qosmax + qosmin; // random between 4 and 30
		#endif

    	if (*qosbase < qosmin) *qosbase = qosmin;
		if (*qosbase > qosmax ) *qosbase = qosmax-1;

		/*
		*Engine: static memory's global state orchestration
		*/	
		if ( (avgRecMQTotal - RecSparkTotal) > *qosbase )
		{
			vector[0] = 1000;
			*state = avgRecMQTotal - RecSparkTotal;
			*lastonespark=*state;
		}
		else
		{
			if (avgRecMQTotal - RecSparkTotal < 0 )
			{
				*state = 0;
				*lastonespark = 0;
			}
			else
			{

				*state= avgRecMQTotal - RecSparkTotal ;
				*lastonespark=*state;
			}
			vector[0] = 0;
						
			
		}
			
		if (cTIMEP == 0)
		{
			*qosbase = qosmin;
			vector[0] = 0;
		}
		if (*state < qosmin ) vector[0] = 0;

		printf("++PT %d SD %d Total Delay %d G-AVG %.2f  Max-TH %.2f TH-loss %.2f State %.2f, Global-Limit %.2f \n", cTIMEP, cDELAY,cTIMEP + cDELAY, global_avg_spark, maxth, measure,*state, *qosbase);
#endif

	}else{
		*state = *lastonespark;
		if (*state > *qosbase){
			vector[0] = 1000;
		}else{
			vector[0] = 0;
		}
	}

	for (int i = 1; i<= split_ipqueue_size; i++) {
		zmq_msg_t msgstate;
        assert(zmq_msg_init(&msgstate) == 0);
        char recMaster[3];
        gcvt(*state, 3, recMaster);
		char cat[8];
		gcvt(*qosbase, 3, cat);
		strcat(cat,"@");
		strcat(cat,recMaster);
		void * butter = malloc(strlen(cat+1));
 		memcpy(butter,cat,strlen(cat));
        zmq_msg_init_data (&msgstate, butter,strlen(cat), free_data, NULL);

		if(zmq_msg_send(&msgstate, split_sockets_state[i], ZMQ_DONTWAIT) == -1)
        {
			//printf ("error in zmq_connect sent state: %s \n", zmq_strerror (errno));
        }
        else
        {
  			//printf(" MQ %d, valor %s \n",i, recMaster);
        }
        assert(zmq_msg_close(&msgstate) != -1);
	}
//	printf("  AVG Throughput %.2f Spark %.2f State %.2f - last %.2f  \n", avgRecMQTotal, RecSparkTotal, *state, *lastonespark );
}


void ml_agent_worker_action(int controll, int msgsize, void ** split_sockets_state, void ** split_sockets, float second,
int ttpersocket, int ttpersocketsec, int input_hanger_size, int cDELAY, int cTIMEP, float state, int *qosbase, int* vector, float *last_second) {

	zmq_msg_t msg;
    assert(zmq_msg_init(&msg) == 0);
	char SentMaster[15];
    gcvt((float)ttpersocket, 15, SentMaster);
    void * butter = malloc(strlen(SentMaster+1));
    memcpy(butter,SentMaster,strlen(SentMaster));
    zmq_msg_init_data (&msg, butter,strlen(SentMaster), free_data, NULL);
    if(zmq_msg_send(&msg,split_sockets[controll], ZMQ_DONTWAIT) == -1) {
      	//printf ("error in zmq_connect: - mq sending data %s \n", zmq_strerror (errno));
    }
    else {
       	//printf (" Sent total of rec %s from MQ %s \n", SentMaster, split_ipqueue[split_ipqueue_size]);
	}
    assert(zmq_msg_close(&msg) != -1);
	zmq_msg_t msgsw;
    assert(zmq_msg_init(&msgsw) == 0);
    if(zmq_msg_recv(&msgsw, split_sockets_state[controll], ZMQ_NOBLOCK) ==-1) {
		//printf ("error in zmq_connect: %s \n", zmq_strerror (errno)); 
    }
    else {
        void*  data_adj      = zmq_msg_data(&msgsw);
        size_t adj_size = zmq_msg_size(&msgsw);
		char*  adj_string = malloc(adj_size+1);
        memcpy(adj_string, data_adj, adj_size);
        adj_string[adj_size] = 0x00;
		char * dados[2];
		char * token = strtok(adj_string, "@");
		
		int si = 0;
        while( token != NULL ) {
			dados[si] = token;
			//printf("d[%d] %s \n", si,dados[si]);
			token = strtok(NULL, "@");
            si++;
		}

		float cslavestate = atof(dados[1]);
#ifndef A3C_RUN		
		//printf("\nNot A3C worker\n");
		float qosslave = atof(dados[0]);
		if (  cslavestate  >=  qosslave ) {
			vector[0] = 1000;
		}
		else {
			vector[0] = 0;
		}
#else
		if (second != *last_second) {
			clock_t start, end;
			float local_thpt = ((float)ttpersocketsec*msgsize)/1024/1024/1024;
			throughput_var = local_thpt - last_local_thpt;
			/* Get new action from worker */
			//[local_thpt, l_thpt_var, proc_t, sche_t, input_hanger_size, ttpersocket, l_ready_mem, l_spark_trsh]
			float curr_env_state[8] = {local_thpt, throughput_var, *cDELAY , *cTIMEP, input_hanger_size, ttpersocket, state, qosbase};	
			start = clock();
			int* actions = worker_infer(worker, curr_env_state);
    		printf("\nCache action is %d", actions[0]);
    		printf("\nFlow action is %d", actions[1]);
			end = clock();

			qosbase = qosbase + actions[0];
			vector[0] = actions[1] * 1000;
			double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
			//printf("Agent took %f seconds to execute \n", cpu_time_used);
			//printf("\n Second is: %f || Last second is: %f", second, *last_second);
			*last_second = second;
			last_local_thpt = local_thpt;
		}
#endif
				
		//printf("control %d qos %.2f state %.2f - [1] %s \n",controll, qosslave, cslavestate,dados[1] );
		free(adj_string);
	}
	assert(zmq_msg_close(&msgsw) != -1);
	
}


void ml_caching (void * API_puller, int msgsize, int msgperbatch, int qosmin, int nb_sending_sockets, char** split_ipqueue, int controll, void ** split_sockets_state, void ** split_sockets, float second,int qosmax,int loss, int window,
int ttpersocket, int ttpersocketsec, int *flagState, float *RecMQTotal, float *avgRecMQTotal, float *RecSparkTotal, uint64_t *ackSent, int *RecTotal, int *cREC, int *cDELAY, int *cTIMEP, float *last_second,
float *global_avg_spark, float *lastonespark, float *state, int *qosbase, int* vector, float maxth, float measure, int input_hanger_size)
{
	int split_ipqueue_size=0;
	vector[0]=0;
  	while (split_ipqueue[split_ipqueue_size] != NULL)
   	{
		split_ipqueue_size++;
   	}
   	split_ipqueue_size--;
	//receive data from Spark API 
  	if (controll == 0) {
        *RecMQTotal=0;
	zmq_msg_t msg;
	assert(zmq_msg_init(&msg) == 0);
        //socket to receive data from spark listner 
	if(zmq_msg_recv(&msg, API_puller, ZMQ_NOBLOCK) == -1) {
	//printf ("error in zmq_connect: %s \n", zmq_strerror (errno));
	//assert(errno == EAGAIN);
        }
        else {
            *ackSent++;
            void*  data      = zmq_msg_data(&msg);
            size_t data_size = zmq_msg_size(&msg);
            char*  rebuilt_string = malloc(data_size+1);
        	memcpy(rebuilt_string, data, data_size);
        	rebuilt_string[data_size] = 0x00;
        	char * dados[3]; 
        	char * token = strtok(rebuilt_string, ";");
        	// loop through the string to extract all other tokens
        	int i = 0;
        	while( token != NULL ) {
                dados[i]=       token;
                token = strtok(NULL, ";");
                i++;
        	}
			free(rebuilt_string);
        	assert(zmq_msg_close(&msg) != -1);
        	*RecTotal += atoi(dados[1]);
        	*cREC  = atoi(dados[1]);
			*RecSparkTotal = ((float)*RecTotal*msgsize)/1024/1024/1024;	
			*cDELAY= atoi(dados[3]);
			*cTIMEP= atoi(dados[2]);
   		}

   		*RecMQTotal = ((float)ttpersocket*msgsize)/1024/1024/1024;
   		if(split_ipqueue_size == 0) {
			*flagState=1;
			*avgRecMQTotal=*RecMQTotal;
		}

		//loop to receive the total of msgs sent from workers (other MQs available)
		if (controll == 0 && split_ipqueue_size != 0)
		{
			//printf("MQ %d, [%.2f] ", controll, *RecMQTotal);
		        int cont = 1;
			for (int i = 1; i<= split_ipqueue_size; i++)   
        		{
				zmq_msg_t msg;
                		assert(zmq_msg_init(&msg) == 0);
		        	if(zmq_msg_recv(&msg, split_sockets[i], ZMQ_NOBLOCK) == -1)
                		{
			                //	printf ("error in zmq_connect Master Rec: %s \n", zmq_strerror (errno));
		                 	assert(errno == EAGAIN);
		                }
                		else
                		{
                       			void*  data     = zmq_msg_data(&msg);
                       			size_t data_size = zmq_msg_size(&msg);
                       			char*  rebuilt_string = malloc(data_size+1);
                       			memcpy(rebuilt_string, data, data_size);
                       			rebuilt_string[data_size] = 0x00;
                   			//printf(" - MQ %d, [%.2f]  ", i, (atof(rebuilt_string)*msgsize)/1024/1024/1024);
					*RecMQTotal += (atof(rebuilt_string)*msgsize)/1024/1024/1024;
					cont++;
					free(rebuilt_string);
                       			assert(zmq_msg_close(&msg) != -1);
                		}

   			}
			//check if all data were retrivied from executors
			//it help to stabilize the current state if there are failures during data collecting;
			if(cont == split_ipqueue_size+1)
			{
				*avgRecMQTotal = *RecMQTotal;	
				*flagState = 1;
			}else{

				*flagState = 0;

			}
		}

		// MASTER ACTION
		ml_agent_master_action(msgsize, split_ipqueue_size, split_sockets_state, split_sockets, qosmin, qosmax, second,
		last_second, *flagState, *RecTotal, *RecMQTotal, *avgRecMQTotal, *RecSparkTotal, *cDELAY, *cTIMEP, 
		global_avg_spark, state, qosbase, vector, lastonespark, maxth, measure);

	} else	if (controll > 0) {
		// WORKER ACTION
		ml_agent_worker_action(controll, msgsize, split_sockets_state, split_sockets, second, ttpersocket, ttpersocketsec,
		input_hanger_size, *cDELAY, *cTIMEP, *state, qosbase, vector, last_second);
	}     
}	


void ml_agent_finish(int controll) {
    
    /* Close Agent */
#ifdef A3C_RUN
	printf("\nClosing Worker");
 	float last_state[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    	worker_finish(worker, last_state);
#endif	

#ifndef RANDOM_ACTION
        if (controll == 0) 
	{
	# ifdef SAQN_RUN
		//saqn_time_csv = fopen("/tmp/SAQN_TIME.csv", "a");
		
		clock_t start, end;
		printf("\nClosing Agent");
		float last_state[8] = {0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0};
		start = clock();
		finish(agent, last_state);
		end = clock();
		double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		//printf("Agent took %f seconds to finish \n", cpu_time_used);
		//fprintf(saqn_time_csv, "%f \t,\t %f\n", second, cpu_time_used);			
		
		//fflush(saqn_time_csv);
		//fclose(saqn_time_csv);
	# endif
	# ifdef A3C_RUN
    		parameter_server_kill(p_server);  
	# endif
		/* Clean up after using CPython. */
		PyMem_RawFree(program);
		Py_Finalize();

	}
#endif

}

void saqn_init() {}
void saqn_master() {}
void saqn_worker() {}
void saqn_finish() {}

void a3c_init() {}
void a3c_master() {}
void a3c_worker() {}
void a3c_finish() {}

#endif

