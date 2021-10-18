#include "Python.h" 
//#include "AgentAPI.h"
#include "A3CProcesses.h"

int main(int argc, char *argv[]) {
    
    PyObject *pmodule;
    wchar_t *program;
    
    program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0], got %d arguments\n", argc);
        exit(1);
    }
    
    /* Add a built-in module, before Py_Initialize */    
    if (PyImport_AppendInittab("A3CProcesses", PyInit_A3CProcesses) == -1) {   
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }
    
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required.
       If this step fails, it will be a fatal error. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */ 
    pmodule = PyImport_ImportModule("A3CProcesses");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'A3CProcesses'\n");
        goto exit_with_error;
    }

    char* ip_list[1];
    ip_list[0] = "localhost";

    /* Now call into your module code. */
    printf("\nStarting Parameter Server");
    PyObject* p_server = parameter_server_proc(180, ip_list, 1);

    printf("\nStarting Worker");
    float start_state[8] = {0.0,0.0,0.0,0,0,0,0,0};
    PyObject* worker = create_worker(start_state,0, ip_list, 1);

    printf("\nGetting first action");
    //float middle_state[8] = {30.0, 5.3, 5.3, 10, 8, 8, 77, 7.7};
    int* actions;
    for(int i=1; i<1010; i++) {
    	float middle_state[8] = {i%30, i%5, i%5, i%10, i%8, i%8, i%77,i% 7};
	actions = worker_infer(worker, middle_state);
    	printf("\nCache action is %d", actions[0]);
    	printf("\nFlow action is %d", actions[1]);
    }
    printf("\nClosing Worker");
    float last_state[8] = {8000.0, 1.1, 1.1, 10, 8, 8, 77, 7.7};
    worker_finish(worker, last_state);

    parameter_server_kill(p_server);
    /* Clean up after using CPython. */
    PyMem_RawFree(program);
    Py_Finalize();

    return 0;

    /* Clean up in the error cases above. */

exit_with_error:
    PyMem_RawFree(program);
    Py_Finalize();
    return 1;

}
