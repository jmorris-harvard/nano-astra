#ifdef NS3_MODULE_COMPILATION 
    error "Do not include ns3 module aggregator headers from other modules these are meant only for end user scripts." 
#endif 
#ifndef NS3_MODULE_MPI
    // Module headers: 
    #include <ns3/mpi-interface.h>
    #include <ns3/mpi-receiver.h>
    #include <ns3/parallel-communication-interface.h>
#endif 