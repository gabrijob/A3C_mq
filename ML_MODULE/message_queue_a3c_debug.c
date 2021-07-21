#include "message_queue.h"
#include "string.h"

#define SAQN_RUN


char * ack_str;
char * nack_str;
char connect_string[100];


PyObject* pmodule;
wchar_t *program;
PyObject* agent = NULL;
PyObject* worker = NULL;
PyObject* p_server = NULL;

float last_global_avg=0;
float last_local_thpt=0;
float throughput_var=0;
//FILE* saqn_time_csv=NULL; 

zmq_msg_t ACK;
zmq_msg_t NACK;

int ttpersocket;
int ttpersocketsec;
int* sockets_status;
int* sockets_volume;
int* sockets_sent_sec;

int* sockets_nbmsg;
int* vector;
int* batch;
long long* sockets_accumulator_stuck;
long long* sockets_stuck_start;
int contup=1;
int RecTotal = 0;
int cREC = 0;
int cDELAY = 0; 
int cTIMEP = 0; 
float maxth = 0;
uint64_t ackSent=0;
float state=0;
int flagState=0;
float RecMQTotal = 0;
float RecSparkTotal = 0;
float prevmaster = 0;
float prev = 0;
float lastonespark=0;
float avgRecMQTotal=0;
float last_state = 0;
float qosbase = 0;
float global_avg_spark=0;
float measure =0;
float last_measure=1;
float last_second = 1;
float min_measure=1;
float mean_measure=1;
int stop = 0;
float mean_sdelay = 0;
float mean_ptime = 0;

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

void ml_agent_master_action(int msgsize, int split_ipqueue_size, void ** split_sockets_state, void ** split_sockets, int qosmin, int qosmax, float second) {
	if (flagState == 1 && (second != last_second)) {
#ifndef A3C_RUN
		printf("\nNot A3C master\n");
			
		//throughput mean
		if ((((float)RecTotal*msgsize)/1024/1024)/second > 0){
		 	global_avg_spark = (((float)RecTotal*msgsize)/1024/1024)/second;
		}


		// Gabriel's TCC - A Bachpessure-based
		//RANDOM_ACTION - off  	
		#ifndef RANDOM_ACTION
			//saqn_time_csv = fopen("/tmp/SAQN_TIME.csv", "a");
				
			clock_t start, end;
			throughput_var = global_avg_spark - last_global_avg;
			/* Get new action from agent */
			//[total_thpt, thpt_var, proc_t, sche_t, msgs_to_spark, msgs_in_gb, ready_mem, spark_thresh]
			float curr_env_state[8] = {global_avg_spark, throughput_var, cDELAY , cTIMEP, RecSparkTotal, RecMQTotal, state, qosbase};	
			start = clock();
			float action = infer(agent, curr_env_state);
			//printf("\nNew action is %f. ", action);
			end = clock();

			qosbase = qosbase + action;
			double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
			//printf("Agent took %f seconds to execute \n", cpu_time_used);
			//printf("\n Second is: %f || Last second is: %f", second, last_second);
			//fprintf(saqn_time_csv, "%f \t,\t %f\n", second, cpu_time_used);
			last_second = second;
			last_global_avg = global_avg_spark;
		
			//fflush(saqn_time_csv);
			//fclose(saqn_time_csv);
		#else
			qosbase = rand() % qosmax + qosmin; // random between 4 and 30
		#endif

    	if (qosbase < qosmin) qosbase = qosmin;
		if (qosbase > qosmax ) qosbase = qosmax-1;

		/*
		*Engine: static memory's global state orchestration
		*/	
		if ( (avgRecMQTotal - RecSparkTotal) > qosbase )
		{
			vector[0] = 1000;
			state = avgRecMQTotal - RecSparkTotal;
			lastonespark=state;
		}
		else
		{
			if (avgRecMQTotal - RecSparkTotal < 0 )
			{
				state = 0;
				lastonespark = 0;
			}
			else
			{

				state= avgRecMQTotal - RecSparkTotal ;
				lastonespark=state;
			}
			vector[0] = 0;
						
			
		}
			
		if (cTIMEP == 0)
		{
			qosbase = qosmin;
			vector[0] = 0;
		}
		if (state < qosmin ) vector[0] = 0;

		printf("++PT %d SD %d Total Delay %d G-AVG %.2f  Max-TH %.2f TH-loss %.2f State %.2f, Global-Limit %.2f \n", cTIMEP, cDELAY,cTIMEP + cDELAY, global_avg_spark, maxth, measure,state, qosbase);
#endif

	}else{
		state = lastonespark;
		if (state > qosbase){
			vector[0] = 1000;
		}else{
			vector[0] = 0;
		}
	}

	for (int i = 1; i<= split_ipqueue_size; i++) {
		zmq_msg_t msgstate;
        assert(zmq_msg_init(&msgstate) == 0);
        char recMaster[3];
        gcvt(state, 3, recMaster);
		char cat[8];
		gcvt(qosbase, 3, cat);
		strcat(cat,"@");
		strcat(cat,recMaster);
		void * butter = malloc(strlen(cat+1));
 		memcpy(butter,cat,strlen(cat));
        zmq_msg_init_data (&msgstate, butter,strlen(cat), my_free, NULL);

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
//	printf("  AVG Throughput %.2f Spark %.2f State %.2f - last %.2f  \n", avgRecMQTotal, RecSparkTotal, state, lastonespark );
}

void ml_agent_worker_action(int controll, int msgsize, void ** split_sockets_state, void ** split_sockets, float second) {

	zmq_msg_t msg;
    assert(zmq_msg_init(&msg) == 0);
	char SentMaster[15];
    gcvt((float)ttpersocket, 15, SentMaster);
    void * butter = malloc(strlen(SentMaster+1));
    memcpy(butter,SentMaster,strlen(SentMaster));
    zmq_msg_init_data (&msg, butter,strlen(SentMaster), my_free, NULL);
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
		printf("\nNot A3C worker\n");
		float qosslave = atof(dados[0]);
		if (  cslavestate  >=  qosslave ) {
			vector[0] = 1000;
		}
		else {
			vector[0] = 0;
		}
			
#else
		if (second != last_second) {
			clock_t start, end;
			float local_thpt = ((float)ttpersocketsec*msgsize)/1024/1024/1024;
			throughput_var = local_thpt - last_local_thpt;
			/* Get new action from worker */
			//[local_thpt, l_thpt_var, proc_t, sche_t, input_hanger_size, ttpersocket, l_ready_mem, l_spark_trsh]
			float curr_env_state[8] = {local_thpt, throughput_var, cDELAY , cTIMEP, input_hanger_size, ttpersocket, state, qosbase};	
			start = clock();
			int* actions = worker_infer(worker, curr_env_state);
    		printf("\nCache action is %d", actions[0]);
    		printf("\nFlow action is %d", actions[1]);
			end = clock();

			qosbase = qosbase + actions[0];
			vector[0] = actions[1] * 1000;
			double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
			//printf("Agent took %f seconds to execute \n", cpu_time_used);
			//printf("\n Second is: %f || Last second is: %f", second, last_second);
			last_second = second;
			last_local_thpt = local_thpt;
		}
#endif
				
		//printf("control %d qos %.2f state %.2f - [1] %s \n",controll, qosslave, cslavestate,dados[1] );
		free(adj_string);
	}
	assert(zmq_msg_close(&msgsw) != -1);
	
}

void ml_caching (void * API_puller, int msgsize, int msgperbatch, int qosmin, int nb_sending_sockets, char** split_ipqueue, int controll, void ** split_sockets_state, void ** split_sockets, float second,int qosmax,int loss, int window)
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
        RecMQTotal=0;
		zmq_msg_t msg;
		assert(zmq_msg_init(&msg) == 0);
        //socket to receive data from spark listner 
		if(zmq_msg_recv(&msg, API_puller, ZMQ_NOBLOCK) == -1) {
            //printf ("error in zmq_connect: %s \n", zmq_strerror (errno));
			//assert(errno == EAGAIN);
        }
        else {
            ackSent++;
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
        	RecTotal += atoi(dados[1]);
        	cREC  = atoi(dados[1]);
			RecSparkTotal = ((float)RecTotal*msgsize)/1024/1024/1024;	
			cDELAY= atoi(dados[3]);
			cTIMEP= atoi(dados[2]);
   		}

   		RecMQTotal = ((float)ttpersocket*msgsize)/1024/1024/1024;
   		if(split_ipqueue_size == 0) {
			flagState=1;
			avgRecMQTotal=RecMQTotal;
		}

		// MASTER ACTION
		ml_agent_master_action(msgsize, split_ipqueue_size, split_sockets_state, split_sockets, qosmin, qosmax, second);

	} else	if (controll > 0) {
		// WORKER ACTION
		ml_agent_worker_action(controll, msgsize, split_sockets_state, split_sockets, second);
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



char** str_split(char* a_str, const char a_delim){
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (a_delim == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);

    /* Add space for terminating null string so caller
       knows where the list of returned strings ends. */
    count++;

    result = malloc(sizeof(char*) * count);

    if (result)
    {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token)
        {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}

long long current_timestamp()
{
    struct timeval te; 
    gettimeofday(&te, NULL); 
    long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; 
    return milliseconds;
}

void my_free (void *data, void *hint){
    free (data);
}

/*
 * Put the current msg into the buffer and return the size of
 */
int hang_msg(zmq_msg_t* msg)
{
    hanger* my_hang = malloc(sizeof(hanger));
    my_hang->input_message = msg;
    my_hang->next = NULL;    
    input_hanger_size += 1;
	if (input_hanger == NULL)
	{
        input_hanger = my_hang;
    }else
    {
        hanger * this_hang = input_hanger;
        //current msg
        //printf("\nRank %d, Step %d, Content %d \n", *(int *)zmq_msg_data (my_hang->input_message),*(((int *)zmq_msg_data (my_hang->input_message))+1),*(((int *)zmq_msg_data (my_hang->input_message))+2));        
        while(this_hang->next != NULL)
        {
            this_hang = this_hang->next;
        }
        this_hang->next = my_hang;
    }
    return input_hanger_size;
}

/*
 * Retrieve a msg without remove it from the queue
 */
zmq_msg_t* gimme_first()
{
    hanger* my_hang = input_hanger;
    if(my_hang == NULL)
    {
        return NULL;
    }
    else
    {
        zmq_msg_t * msg = my_hang->input_message;
        return msg;
    }
}

/*
 * Retrieve a msg and remove it from the queue.
 */
zmq_msg_t* pick_up_msg()
{
    hanger* my_hang = input_hanger;
    if(my_hang == NULL)
    {
        return NULL;
    }else
    {
        input_hanger = my_hang->next;
        zmq_msg_t * msg = my_hang->input_message;
        input_hanger_size -= 1;
        free(my_hang);
        return msg;
    }
}

/*
 * Retrieve a message from producer
 */
void get_request_message(void** items, int nb_sending_sockets)
{
    zmq_msg_t* msg = malloc(sizeof(zmq_msg_t));
    assert(zmq_msg_init(msg) == 0);
    if(zmq_msg_recv(msg, items[0], ZMQ_NOBLOCK) == -1)
    {
        assert(errno == EAGAIN);
        free(msg);
    }
    else
    {
        hang_msg(msg);
    }
}

/*
 * API to get information from Spark
 */
void caching (void * API_puller, int msgsize, int msgperbatch, int qosmin, int nb_sending_sockets, char** split_ipqueue, int controll, void ** split_sockets_state, void ** split_sockets, float second,int qosmax,int loss, int window)
{
	int split_ipqueue_size=0;
	vector[0]=0;
  	while (split_ipqueue[split_ipqueue_size] != NULL)
   	{
		split_ipqueue_size++;
   	}
   	split_ipqueue_size--;
	//receive data from Spark API 
  	if (controll == 0)
   	{
        	RecMQTotal=0;
		zmq_msg_t msg;
		assert(zmq_msg_init(&msg) == 0);
        	//socket to receive data from spark listner 
		if(zmq_msg_recv(&msg, API_puller, ZMQ_NOBLOCK) == -1)
        	{
              		//printf ("error in zmq_connect: %s \n", zmq_strerror (errno));
			//assert(errno == EAGAIN);
        	}
        	else
        	{
               		ackSent++;
               		void*  data      = zmq_msg_data(&msg);
               		size_t data_size = zmq_msg_size(&msg);
               		char*  rebuilt_string = malloc(data_size+1);
        		memcpy(rebuilt_string, data, data_size);
        		rebuilt_string[data_size] = 0x00;
        		char * dados[3]; 
        		char * token = strtok(rebuilt_string, ";");
        		// loop through the string to extract all other tokens
        		int i = 0;
        		while( token != NULL ) 
        		{
                		dados[i]=       token;
                		token = strtok(NULL, ";");
                		i++;
        		}
			free(rebuilt_string);
        		assert(zmq_msg_close(&msg) != -1);
        		RecTotal += atoi(dados[1]);
        		cREC  = atoi(dados[1]);
			RecSparkTotal = ((float)RecTotal*msgsize)/1024/1024/1024;	
			cDELAY= atoi(dados[3]);
			cTIMEP= atoi(dados[2]);

   		}

   		RecMQTotal = ((float)ttpersocket*msgsize)/1024/1024/1024;
   		if(split_ipqueue_size == 0)
		{
			flagState=1;
			avgRecMQTotal=RecMQTotal;
		}
		//loop to receive the total of msgs sent from workers (other MQs available)
		if (controll == 0 && split_ipqueue_size != 0)
		{
			//printf("MQ %d, [%.2f] ", controll, RecMQTotal);
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
					RecMQTotal += (atof(rebuilt_string)*msgsize)/1024/1024/1024;
					cont++;
					free(rebuilt_string);
                       			assert(zmq_msg_close(&msg) != -1);
                		}

   			}
			//check if all data were retrivied from executors
			//it help to stabilize the current state if there are failures during data collecting;
			if(cont == split_ipqueue_size+1)
			{
				avgRecMQTotal = RecMQTotal;	
				flagState = 1;
			}else{

				flagState = 0;

			}
		}



		if (flagState == 1 && (second != last_second)) {
#ifndef A3C_RUN
			printf("\nNot A3C master\n");
			
	   			//throughput mean
				if ((((float)RecTotal*msgsize)/1024/1024)/second > 0){
				       	global_avg_spark = (((float)RecTotal*msgsize)/1024/1024)/second;

				}


				// Gabriel's TCC - A Bachpessure-based
				//RANDOM_ACTION - off  	
				#ifndef RANDOM_ACTION
					//saqn_time_csv = fopen("/tmp/SAQN_TIME.csv", "a");
		
				
					clock_t start, end;
					throughput_var = global_avg_spark - last_global_avg;
					/* Get new action from agent */
					//[total_thpt, thpt_var, proc_t, sche_t, msgs_to_spark, msgs_in_gb, ready_mem, spark_thresh]
					float curr_env_state[8] = {global_avg_spark, throughput_var, cDELAY , cTIMEP, RecSparkTotal, RecMQTotal, state, qosbase};	
					start = clock();
					float action = infer(agent, curr_env_state);
					//printf("\nNew action is %f. ", action);
					end = clock();

					qosbase = qosbase + action;
					double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
					//printf("Agent took %f seconds to execute \n", cpu_time_used);
					//printf("\n Second is: %f || Last second is: %f", second, last_second);
					//fprintf(saqn_time_csv, "%f \t,\t %f\n", second, cpu_time_used);
					last_second = second;
					last_global_avg = global_avg_spark;
				
					//fflush(saqn_time_csv);
					//fclose(saqn_time_csv);
				#else
					qosbase = rand() % qosmax + qosmin; // random between 4 and 30
				#endif

    
				if (qosbase < qosmin) qosbase = qosmin;
				if (qosbase > qosmax ) qosbase = qosmax-1;

			/*
			*Engine: static memory's global state orchestration
			*/	
		
			
			if ( (avgRecMQTotal - RecSparkTotal) > qosbase )
			{
				vector[0] = 1000;
				state = avgRecMQTotal - RecSparkTotal;
				lastonespark=state;
			}
			else
			{

				if (avgRecMQTotal - RecSparkTotal < 0 )
				{
					state = 0;
					lastonespark = 0;
				}
				else
				{

					state= avgRecMQTotal - RecSparkTotal ;
					lastonespark=state;
				}
				vector[0] = 0;
			
				
			
			}
				

				
			if (cTIMEP == 0)
			{
				qosbase = qosmin;
				vector[0] = 0;
			}
			if (state < qosmin ) vector[0] = 0;
#endif
				printf("++PT %d SD %d Total Delay %d G-AVG %.2f  Max-TH %.2f TH-loss %.2f State %.2f, Global-Limit %.2f \n", cTIMEP, cDELAY,cTIMEP + cDELAY, global_avg_spark, maxth, measure,state, qosbase);




		

		}else{
			state = lastonespark;
			if (state > qosbase){
				vector[0] = 1000;
			}else{
				vector[0] = 0;
			}
		}

		for (int i = 1; i<= split_ipqueue_size; i++)
                {
			zmq_msg_t msgstate;
                	assert(zmq_msg_init(&msgstate) == 0);
               		char recMaster[3];
                	gcvt(state, 3, recMaster);
			char cat[8];
			gcvt(qosbase, 3, cat);
			strcat(cat,"@");
			strcat(cat,recMaster);
			void * butter = malloc(strlen(cat+1));
 		        memcpy(butter,cat,strlen(cat));
                        zmq_msg_init_data (&msgstate, butter,strlen(cat), my_free, NULL);
    

			if(zmq_msg_send(&msgstate, split_sockets_state[i], ZMQ_DONTWAIT) == -1)
                        {
	      //                        printf ("error in zmq_connect sent state: %s \n", zmq_strerror (errno));
               
			}
                        else
                        {
  			//	     printf(" MQ %d, valor %s \n",i, recMaster);
                        }
                	assert(zmq_msg_close(&msgstate) != -1);
		}
//		printf("  AVG Throughput %.2f Spark %.2f State %.2f - last %.2f  \n", avgRecMQTotal, RecSparkTotal, state, lastonespark );
	}
	else
	{

		if (controll > 0){
                	zmq_msg_t msg;
                	assert(zmq_msg_init(&msg) == 0);
                	char SentMaster[15];
               		 gcvt((float)ttpersocket, 15, SentMaster);
                	void * butter = malloc(strlen(SentMaster+1));
                	memcpy(butter,SentMaster,strlen(SentMaster));
                	zmq_msg_init_data (&msg, butter,strlen(SentMaster), my_free, NULL);
                	if(zmq_msg_send(&msg,split_sockets[controll], ZMQ_DONTWAIT) == -1)
                	{
                              //      printf ("error in zmq_connect: - mq sending data %s \n", zmq_strerror (errno));
               		}
               		else
               		{
              		//                      printf (" Sent total of rec %s from MQ %s \n", SentMaster, split_ipqueue[split_ipqueue_size]);
			}
               		assert(zmq_msg_close(&msg) != -1);
			zmq_msg_t msgsw;
               		 assert(zmq_msg_init(&msgsw) == 0);
                	if(zmq_msg_recv(&msgsw, split_sockets_state[controll], ZMQ_NOBLOCK) ==-1)
                	{
           	       		//printf ("error in zmq_connect: %s \n", zmq_strerror (errno)); 
                	}
                	else
                	{
                        	void*  data_adj      = zmq_msg_data(&msgsw);
                        	size_t adj_size = zmq_msg_size(&msgsw);
	                        char*  adj_string = malloc(adj_size+1);
        	                memcpy(adj_string, data_adj, adj_size);
                	        adj_string[adj_size] = 0x00;
				char * dados[2];
				char * token = strtok(adj_string, "@");

				
				int si = 0;
	                        while( token != NULL )
        	                {
					dados[si] = token;
//					printf("d[%d] %s \n", si,dados[si]);
				 	token = strtok(NULL, "@");

                                	si++;
                        	}

				float cslavestate = atof(dados[1]);
			#ifndef A3C_RUN		
				printf("\nNot A3C worker\n");
				float qosslave = atof(dados[0]);
                        	if (  cslavestate  >=  qosslave )
                        	{
                                	vector[0] = 1000;
                        	}
				else	
                        	{
                               		vector[0] = 0;
                        	}
			
			#else
				 if (second != last_second) {
					clock_t start, end;
					float local_thpt = ((float)ttpersocketsec*msgsize)/1024/1024/1024;
					throughput_var = local_thpt - last_local_thpt;
					/* Get new action from worker */
					//[local_thpt, l_thpt_var, proc_t, sche_t, input_hanger_size, ttpersocket, l_ready_mem, l_spark_trsh]
					float curr_env_state[8] = {local_thpt, throughput_var, cDELAY , cTIMEP, input_hanger_size, ttpersocket, state, qosbase};	
					start = clock();
					int* actions = worker_infer(worker, curr_env_state);
    					printf("\nCache action is %d", actions[0]);
    					printf("\nFlow action is %d", actions[1]);
					end = clock();

					qosbase = qosbase + actions[0];
					vector[0] = actions[1] * 1000;
					double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
					//printf("Agent took %f seconds to execute \n", cpu_time_used);
					//printf("\n Second is: %f || Last second is: %f", second, last_second);
					last_second = second;
					last_local_thpt = local_thpt;
				 }
			#endif
				
				//printf("control %d qos %.2f state %.2f - [1] %s \n",controll, qosslave, cslavestate,dados[1] );
                        	free(adj_string);
		                }
               			assert(zmq_msg_close(&msgsw) != -1);
			}
	}	

}

/*
 * Forward messages to the consumer as a message part until the lenght reaches max_effective_data_to_send_per_window
 */
int write_message_to_consumer(int msgperbatch, int msgsize, void** items, int i, int max_effective_data_to_send_per_window, int threshold,int qosmin,void * API_puller,float second, int nb_sending_sockets, char** split_ipqueue,int controll, void** split_sockets_state, void** split_sockets,int qosmax,int loss,int window)
{
    int sent = 0;
    if(sockets_status[i] != SOCKET_READY_TO_SEND)
    {
		printf("Socket busy: sockets_status[%d]",sockets_status[i] );
        return 0;
    }



		 zmq_msg_t * input_message = gimme_first();



   if(input_message != NULL)
   {
        zmq_msg_t output_message;
        zmq_msg_init(&output_message);
        assert(zmq_msg_move(&output_message, input_message) !=-1);
        int v = zmq_msg_size(&output_message);
        sockets_volume[i] += v;
	//vector[0]=0;
       	caching(API_puller,msgsize,msgperbatch,qosmin,nb_sending_sockets,split_ipqueue,controll,split_sockets_state,split_sockets,second,qosmax,loss,window);
         /** once this socket has sent enough data, it will need to send
         * the last message part and change the socket status
         */
if (vector[0] == 0){
		if(sockets_volume[i] >=  max_effective_data_to_send_per_window  )
		{
//		printf(" ack number %d \n", batch[i]);
	
		if (batch[i]==threshold)
			{
			  //printf("control %d ack total %d - %d \n",controll, threshold,i);
				if(zmq_msg_send(&output_message, items[1+(i*2)], ZMQ_DONTWAIT) == -1)
				{
					assert(errno > EAGAIN);
				}
				else
				{
					sockets_status[i] = SOCKET_WAITING_ACK;
					sockets_stuck_start[i] = current_timestamp();
					sent++;         
					batch[i]=1;     
				}
				
			}
			else
			{
				if(zmq_msg_send(&output_message, items[1+(i*2)],ZMQ_DONTWAIT) == -1)
				{
					assert(errno == EAGAIN);
				}
				else
				{
					sockets_volume[i] = 0;
					sent++;
					batch[i]++;
				}
			}
	}   
        else
        {
				if(zmq_msg_send(&output_message, items[1+(i*2)],ZMQ_DONTWAIT|ZMQ_SNDMORE) == -1)
				{
					assert(errno == EAGAIN);
				}
				else
				{
					sent++;
				}
        }
	    
}
        if(sent)
        {

            pick_up_msg();
            sockets_nbmsg[i]++;
            assert(zmq_msg_close(input_message) != -1);
            assert(zmq_msg_close(&output_message) != -1);
            free(input_message);
            sockets_status[i] = SOCKET_READY_TO_SEND;//add to fix bug (all sockets stuck)

        }
        else
        {
            sockets_volume[i] -= v;
            assert(zmq_msg_move(input_message, &output_message) !=-1);
	    sockets_status[i] = SOCKET_READY_TO_SEND;//add to fix bug (all sockets stuck)
        }

    }

    return sent;
}

/*
 * Finalize the sending. If the socket was sending parts, send a zero length
 * message. Then send a 2byte message to the consumer indicating that the work
 * is done.
 */
void end_connection(void** items, int i, int max_effective_data_to_send_per_window)
{
    zmq_msg_t output_message;
    assert(zmq_msg_init_size(&output_message, 0) != -1);
    if(sockets_volume[i] > 0)
    {
        assert(zmq_msg_send(&output_message, items[1+(i*2)], 0) != -1);
    }
    assert(zmq_msg_close(&output_message) != -1);
    zmq_msg_t output_message2;
    char* end = malloc(2);
    assert(zmq_msg_init_data(&output_message2, end, 2, my_free, NULL) != -1);
    if(sockets_volume[i] > 0)
    {
        assert(zmq_msg_send(&output_message2, items[1+(i*2)], 0) != -1);
    }
    assert(zmq_msg_close(&output_message2) != -1);
}

/*
 * Acquire the ack from receiver and update the socket state
 */
void get_ack_from_consumer(void** items, int i)
{
	zmq_msg_t msg;
    assert(zmq_msg_init(&msg) == 0);
    if(zmq_msg_recv(&msg, items[2+(i*2)], ZMQ_NOBLOCK) == -1)
    {
        assert(errno == EAGAIN);
    }
    else
    {
        void*  data      = zmq_msg_data(&msg);
        size_t data_size = zmq_msg_size(&msg);
        char*  rebuilt_string = malloc(data_size+1);
        memcpy(rebuilt_string, data, data_size);
        rebuilt_string[data_size] = 0x00;
        if(strncmp(rebuilt_string, ack_str, 3) == 0)
        {
            sockets_status[i] = SOCKET_READY_TO_SEND;
            sockets_volume[i] = 0;
            sockets_accumulator_stuck[i]+=current_timestamp() - sockets_stuck_start[i];
        }  
        free(rebuilt_string);
        assert(zmq_msg_close(&msg) != -1);
    }
}



void do_queue(int msgperbatch,int max_effective_data_to_send_per_window, int duration, int sending_port, int receiving_port, int io_threads,int nb_sending_sockets, int msgsize, int threshold, int qosmin, char** split_ipqueue, int controll, int qosmax, int loss, int window)
{
    char* output_csv = malloc(100);
    sprintf(output_csv, "/tmp/LOG.csv");
    FILE* outcsv = fopen(output_csv, "w");

    ack_str = malloc(4);
    memcpy(ack_str, "ACK", 3);
    ack_str[3] = 0x00;
    assert(zmq_msg_init_data(&ACK, ack_str, 4, my_free, NULL) != -1);
    nack_str = malloc(5);
    memcpy(nack_str, "NACK", 5);
    nack_str[4] = 0x00;
    assert(zmq_msg_init_data(&ACK, nack_str, 5, my_free, NULL) !=-1);

    float second = 0; 

    batch                      = malloc(nb_sending_sockets*sizeof(int));
    vector                     = malloc(2*sizeof(int));
    sockets_status             = malloc(nb_sending_sockets*sizeof(int));
    sockets_volume             = malloc(nb_sending_sockets*sizeof(int));
    sockets_nbmsg              = malloc(nb_sending_sockets*sizeof(int));
	sockets_sent_sec 		   = malloc(nb_sending_sockets*sizeof(int));
    sockets_accumulator_stuck  = malloc(nb_sending_sockets*sizeof(long long));
    sockets_stuck_start        = malloc(nb_sending_sockets*sizeof(long long));
    
    memset(batch, 					   0, nb_sending_sockets*sizeof(int));
    memset(vector,                     0, 2*sizeof(int));
    memset(sockets_sent_sec,           0, nb_sending_sockets*sizeof(int));
    memset(sockets_status,             0, nb_sending_sockets*sizeof(int));
    memset(sockets_volume,             0, nb_sending_sockets*sizeof(int));
    memset(sockets_nbmsg,              0, nb_sending_sockets*sizeof(int));
    memset(sockets_accumulator_stuck,  0, nb_sending_sockets*sizeof(long long));
    memset(sockets_stuck_start,        0, nb_sending_sockets*sizeof(long long));

    fprintf(outcsv, "timestamp,");
	for(int i=0; i<nb_sending_sockets; i++)
	{
		sockets_status[i] = SOCKET_READY_TO_SEND;
		batch[i]=1;
	    fprintf(outcsv, "socket_%d,socket_%d_TSENT,socket_%d_TSENTSEC,socket_%d_TTHSENTSEC,socket_%d_TTHSENT,socket_status_%d,",i,i,i,i,i,i);
	}
	fprintf(outcsv,"TMSG,TOSEND,SENT,TMSGSEC,TTHMSGSEC,AVG,THtotalsent,ackSent,state,globalState,THloss\n");

    /*if(controll == 0){	
    	saqn_time_csv = fopen("/tmp/SAQN_TIME.csv", "w");
    	fprintf(saqn_time_csv, "TIMESTAMP \t,\t SAQN_PROC_TIME\n");
    	fflush(saqn_time_csv);
    	fclose(saqn_time_csv);
    }*/
	
	
    void ** sockets  = malloc(sizeof(void**)*(1+(nb_sending_sockets*2)));
    
    void * context = zmq_ctx_new();
    
    // Setting the number of threads per queue
    zmq_ctx_set(context, ZMQ_IO_THREADS, io_threads);
    assert (zmq_ctx_get (context, ZMQ_IO_THREADS) == io_threads);
    
    // Connecting to the producer
    void * input_socket = zmq_socket(context, ZMQ_PULL); 
    sprintf (connect_string, "tcp://*:%d", receiving_port);
    if (zmq_bind (input_socket, connect_string) == -1){
        printf("failure binding REP %s", connect_string);
        fflush(stdout);
        exit(-1);
    }
    else
    {
	 printf("Consumer connected at port %s\n", connect_string);
    }
	
    sockets[0] = input_socket;
    for(int i=0; i<nb_sending_sockets; i++)
    {
        void* pusher = zmq_socket(context, ZMQ_PUSH);
        void* puller = zmq_socket(context, ZMQ_PULL);
        sockets[1+(i*2)] = pusher;
        sockets[2+(i*2)] = puller;
        sprintf (connect_string, "tcp://*:%d", sending_port);
        if (zmq_bind (pusher, connect_string) == -1)
        {
            printf("failure connecting puller %d %s \n", i, connect_string);
            fflush(stdout);
            exit(-1);
        }
	else
	{
        	printf("pusher bound %s\n", connect_string);
	}
        sprintf (connect_string, "tcp://*:%d", sending_port+1);
        if(zmq_bind (puller, connect_string) == -1)
        {
            printf("failure connecting pusher %d %s", i, connect_string);
            fflush(stdout);
            exit(-1);
        }
	else
	{
        	printf("puller bound %s\n", connect_string);
	}
        sending_port+=2;
    }
    int API_port = 5000;

    void * API_puller;


    if (controll == 0 ){
    	API_puller = zmq_socket(context, ZMQ_PULL); 
    	sprintf (connect_string, "tcp://*:%d", API_port);
    	if (zmq_bind (API_puller, connect_string) == -1)
    	{
        	printf("failure binding REP %s", connect_string);
	        fflush(stdout);
        	exit(-1);
    	}
    	else
    	{
		printf(" Spark API connected %s\n", connect_string);
    	}
    }



    int split_ipqueue_size = 0;
    while (split_ipqueue[split_ipqueue_size] != NULL)
    {
	split_ipqueue_size++;
    }
    split_ipqueue_size--;

    void ** split_sockets = malloc(sizeof(void**)*(split_ipqueue_size+1));
    void ** split_sockets_state = malloc(sizeof(void**)*(split_ipqueue_size+1));

    int pp = 50000;
    int pps = 35001;
    int pps1 = 35000;
    for(int i=0; i<=split_ipqueue_size; i++)
    {
	if( controll==0 && i!=0)
	{
		printf("\nConnections on Master side!\n\n");
		printf ("[Getting number of sent messages from workers] \n");
        	void * wpuller = zmq_socket(context, ZMQ_PULL);
                split_sockets[i] = wpuller;
		sprintf (connect_string, "tcp://*:%d", pp);
        	if(zmq_bind (wpuller, connect_string) == -1)
       		{
			printf("Getting data error in zmq_connect: %s \n", zmq_strerror (errno));
            		printf("Failure connecting puller in MQ %d %s", controll, connect_string);
            		fflush(stdout);
            		exit(-1);
        	}
        	else
        	{
			
                	printf("Puller bound in MQ %d (%s:%d)\n\n", controll,split_ipqueue[0],pp);
        	}
	}
	if ( controll==0 && i==0){
		printf ("[Sent Cache state from master to workers] \n");		
		for(int b=1; b<=split_ipqueue_size; b++)
		{
                	void * spusher = zmq_socket(context, ZMQ_PUSH);
                	split_sockets_state[b] = spusher;
                	//sprintf (connect_string, "tcp://%s:%d",split_ipqueue[0], pps);
                	sprintf (connect_string, "tcp://*:%d" ,pps);
			if(zmq_bind (spusher, connect_string) == -1)
                	{
                        	printf ("Error in zmq_connect sent cache state: %s \n", zmq_strerror (errno));
                        	printf("Failure connecting pusher in %d %s", b, connect_string);
                        	fflush(stdout);
                        	exit(-1);
                	}
                	else
                	{
                 
                        	printf("Pusher ? Connected to MQ %d (%s) from %s:%d \n", b, connect_string,split_ipqueue[0],pps);
                	}
			pps++;
		}
	}

	if( controll == i && i != 0)
	{
		printf("\nConnections on worker side!\n\n");
                printf ("[Sending number of sent messages to Master node] \n");
		void * wpusher = zmq_socket(context, ZMQ_PUSH);
		split_sockets[i] = wpusher;
		sprintf (connect_string, "tcp://%s:%d", split_ipqueue[0],pp);  
		if (zmq_connect (wpusher, connect_string) == -1)
        	{
		        printf ("Pushing data error in zmq_connect: %s \n", zmq_strerror (errno));
                        printf("Failure connecting pusher from MQ %d to pull socket in MQ at %s", i, connect_string);
            		fflush(stdout);
            		exit(-1);
        	}
        	else
        	{
                        printf("pusher connnected to %s from MQ %d (%s:%d) \n\n", connect_string, i, split_ipqueue[i],pp);

        	}
		
		pps1 = pps1 + controll;
                printf ("[Getting Cache state from master] \n");
		void * spuller = zmq_socket(context, ZMQ_PULL);
                split_sockets_state[i] = spuller;
                //sprintf (connect_string, "tcp://*:%d" ,pps1);
		sprintf (connect_string, "tcp://%s:%d",split_ipqueue[0], pps1);
                if (zmq_connect (spuller, connect_string) == -1)
                {
                        printf ("Error in zmq_connect get cache state: %s \n", zmq_strerror (errno));
                        printf("Failure connecting puller %d %s \n", i, connect_string);
                        fflush(stdout);
                        exit(-1);
                }
                else
                {
                        printf("Puller bound in MQ %d (%s:%d)\n", i, split_ipqueue[0],pps1);
                }

		
	}
	pp++;
//	pps1++;
    }

    /* Start ML Agent */
    ml_agent_init(controll, qosmin, qosmax, split_ipqueue_size, split_ipqueue, duration);

    // Time tracking to work only for Duration
    int keepup = 1;
    long long start_time     = current_timestamp();
    long long second_tracker = start_time;

    /**************************************************************************
     *
     *                          Main Event Loop
     *
     *************************************************************************/
    /*
     * get message from queue
     */
    int* free_to_send = malloc(nb_sending_sockets*4);
    int last_start = 0;
    long long whithout_write_start = current_timestamp();
    long long total_without_write = 0;

  
    printf("Starting\n");
    while(keepup)
    {
        /*
         * Creates a buffer (item 0) and returns the current size
         */
        get_request_message(sockets, nb_sending_sockets);
        /**
         * Process which output socket is ready to send and try to see if an ack
         * has been received for the other ones.
         */
        for(int i=0; i<nb_sending_sockets;i++)
        {
		get_ack_from_consumer(sockets, i);
		//printf("sockets_status[%d]\n",i);	
        }

        // if there is some message...    
		if(input_hanger_size >= 0 )
		{     
			//for each socket
            for(int i=last_start; i<=nb_sending_sockets+last_start;i++)
            {
                int k = (i % nb_sending_sockets);
                if(sockets_status[k] == SOCKET_READY_TO_SEND)
                {
					int ret = write_message_to_consumer(msgperbatch,msgsize, sockets, k, max_effective_data_to_send_per_window,threshold,qosmin,API_puller,second,nb_sending_sockets,split_ipqueue, controll, split_sockets_state,split_sockets,qosmax,loss,window);
					if (ret > 0)
					{
						long long now = current_timestamp();
						total_without_write += now - whithout_write_start;
						whithout_write_start = now;
					}

				}
            }
            last_start++;
            if(last_start == nb_sending_sockets)
            {
                last_start = 0;
            }
        }
        
        /**
         * This loop save some statiscs every second
         * **/
		long long now = current_timestamp();
        keepup = now - start_time < duration;
        // Every seconds write CSV
        if(!keepup || now - second_tracker >= ONE_SECOND)
        {
			second += ((now - second_tracker) / ONE_SECOND);
			second_tracker = now;          
            fprintf(outcsv, "%d,",(int)second);
			ttpersocketsec = 0;
			for(int i=0; i<nb_sending_sockets;i++)
			{
				if ( i >= 1)
				{
					fprintf(outcsv, ",");
				}
                fprintf(outcsv, "%d,%d,%d,%.2f,%.2f,%d",i,sockets_nbmsg[i],sockets_nbmsg[i]-sockets_sent_sec[i],((((float)sockets_nbmsg[i]-sockets_sent_sec[i])*msgsize)/1024/1024), (((float)sockets_nbmsg[i]*msgsize)/1024/1024),sockets_status[i]);
				ttpersocket += (sockets_nbmsg[i]-sockets_sent_sec[i]);                
				ttpersocketsec += (sockets_nbmsg[i]-sockets_sent_sec[i]);
				sockets_sent_sec[i] = sockets_nbmsg[i];
            
			}
			float THtotalsent = ((float)ttpersocket*msgsize)/1024/1024; 
			fprintf(outcsv, ",%d,%d,%d,%d,%.2f,%.2f,%.2f,%ld,%.2f,%.2f,%.2f\n", ttpersocket+input_hanger_size, input_hanger_size, ttpersocket, ttpersocketsec,((float)ttpersocketsec*msgsize)/1024/1024 ,THtotalsent/(float)second,THtotalsent,ackSent,state,qosbase,measure);
			ackSent=0;
		}
	}	

	for(int i=0; i<nb_sending_sockets; i++)
	{
		printf("Ending connections..\n");
		end_connection (sockets, i, max_effective_data_to_send_per_window);
    }
      
    free(batch);
    free(vector);
    free(sockets_status);
    free(sockets_volume);
    free(sockets_nbmsg);
    free(ack_str);
    free(nack_str);
    free(free_to_send);
    free(sockets_sent_sec);

    
    /* Close ML Agent */
    ml_agent_finish(controll);


    printf("closing sockets\n");
    for(int i=0; i<1+(nb_sending_sockets*2); i++)
    {
        zmq_close(sockets[i]);
        printf("socket %d closed\n", i);
    }
   zmq_close(API_puller);
    //zmq_close(MASTER);
    //zmq_close(WORKER);
    split_ipqueue_size = 0;
    while (split_sockets[split_ipqueue_size] != NULL)
    {
        split_ipqueue_size++;
    }
    for (int i=0;i<=split_ipqueue_size; i++){
	zmq_close(split_sockets_state[i]);
        zmq_close(split_sockets[i]);
    } 

    fflush(outcsv);	
    fclose(outcsv);

    printf("terminateing context\n");
    fflush(stdout);
    zmq_ctx_term(context);
    printf("destroying context context\n");
    fflush(stdout);
    zmq_ctx_destroy(context);
    printf("Can die\n");
	    
}

/*
 * int   max_effective_data_to_send_per_window    : batch size msg * amoun of msg...in bytes
 * int   duration           					  : execution time in seconds
 * int   sending_port       					  : port used to connect with spark workes - starts in 9999, plus 2, every time
 * char* receiving_port    						  : Port to be used to receive data from the producer
 * int   nb_sending_sockets 					  : How many Worker nodes
 * int   io_threads								  : How many io threads at the MQ level
 * int   threshold								  : ack frequency - ack sent after how many batches? 
 * int	 qos									  : max value in GB to send spark
 */
int main(int argc, char** argv)
{
	
	if( argc < 15 || argc > 15 )
    {
        printf("The list of arguments is not enought.\n");
        printf("usage: ./message_queue <execution time (s)> <data input port> <data output ports p,p,p> <IO threads> <#msgs per batch> <#sockets> <MSG size> <ACK freq> <qos (GB)>\n");
		printf(" ./message_queue 30 5050 9999,1001 8 10 2 10000 1000 20'\n");
        return 1;
    }
    
	int   duration            = atoi(argv[1]) * 1000;
	int   receiving_port      = atoi(argv[2]);
	int   sending_port        = atoi(argv[3]);
	int   io_threads          = atoi(argv[4]);
	int   msgperbatch         = atoi(argv[5]);
	int   nb_sending_sockets  = atoi(argv[6]);
	int   msgsize		  = atoi(argv[7]);
	int   threshold		  = atoi(argv[8]);
	int   qosmin		  = atoi(argv[9]);
	char* list_ipqueue        = argv[10];	
	int   controll		  = atoi(argv[11]);
	int   qosmax		  = atoi(argv[12]);
	int   loss		  = atoi(argv[13]);
	int   window              = atoi(argv[14]);

	char** split_ipqueue   = str_split(list_ipqueue, ',');
	//adjust msg size
	int aux = (int)floor(sqrt((msgsize-8)/4));
	msgsize = aux*aux*4;
	//start qosbase with the minimum qos
	qosbase=(float) qosmin;
	uint64_t max_effective_data_to_send_per_window = msgperbatch * msgsize;
	
	/* Import Agent */
    	program = Py_DecodeLocale(argv[0], NULL);
   	if (program == NULL) {
	    	fprintf(stderr, "Fatal error: cannot decode argv[0], got %d arguments\n", argc);
	    	exit(1);
    	}
 	/* Add a built-in module, before Py_Initialize */    
#ifdef SAQN_RUN
	if (PyImport_AppendInittab("SAQNAgent", PyInit_SAQNAgent) == -1) {   
		fprintf(stderr, "Error: could not extend in-built modules table\n");
		exit(1);
	}
#endif
#ifdef A3C_RUN
	if (PyImport_AppendInittab("A3CProcesses", PyInit_A3CProcesses) == -1) {   
		fprintf(stderr, "Error: could not extend in-built modules table\n");
		exit(1);
	}
#endif

	/* Pass argv[0] to the Python interpreter */
	Py_SetProgramName(program);
    	/* Initialize the Python interpreter.  Required.  If this step fails, it will be a fatal error. */
    	Py_Initialize();
    	/* Optionally import the module; alternatively,
	 **import can be deferred until the embedded script
	 **imports it. */ 
#ifdef SAQN_RUN
	pmodule = PyImport_ImportModule("SAQNAgent");
	if (!pmodule) {
	       	PyErr_Print();
		fprintf(stderr, "Error: could not import module 'SAQNAgent'\n");
        	PyMem_RawFree(program);
		Py_Finalize();
		exit(1);
	}
#endif
#ifdef A3C_RUN
	pmodule = PyImport_ImportModule("A3CProcesses");
	if (!pmodule) {
	       	PyErr_Print();
		fprintf(stderr, "Error: could not import module 'A3CProcesses'\n");
        	PyMem_RawFree(program);
		Py_Finalize();
		exit(1);
	}
#endif



	do_queue(msgperbatch,max_effective_data_to_send_per_window, duration, sending_port, receiving_port, io_threads, nb_sending_sockets, msgsize, threshold, qosmin,split_ipqueue, controll,qosmax,loss,window);
    return 0;
}
