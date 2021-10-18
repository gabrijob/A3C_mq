#Machine learning module for Spark message queue applications
cmake_minimum_required(VERSION 3.7)
project(ml_module VERSION 1.0.1)
include(GNUInstallDirs)

find_package(PkgConfig REQUIRED)

# C++11
set                         (CMAKE_CXX_STANDARD        11)
set                         (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall")

option(BUILD_WITH_ZMQ     "Build with ZeroMQ"                             ON)
option(transport_mpi      "Build Decaf with MPI transport layer"          ON)
set                         (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

## use pkg-config to get hints for 0mq locations
pkg_check_modules(PC_ZeroMQ QUIET zmq)

find_path(ZeroMQ_INCLUDE_DIR
        NAMES zmq.hpp
        PATHS ${PC_ZeroMQ_INCLUDE_DIRS}
        )

find_library(ZeroMQ_LIBRARY
        NAMES zmq
        PATHS ${PC_ZeroMQ_LIBRARY_DIRS}
        )
set  (libraries    ${libraries} ${})

# MPI
if                          (transport_mpi)
  find_package(MPI REQUIRED)
  include_directories(${MPI_INCLUDE_PATH})
  set                       (libraries    ${libraries} ${MPI_LIBRARIES})
  set                       (TRANSPORT_MPI ON)
endif                       (transport_mpi)

# Libm
#find_package(LIBM REQUIRED)
#include_directories(${LIBM_INCLUDE_PATH})
#set (libraries    ${libraries} ${LIBM_LIBRARIES})


## Python libraries
include_directories("/usr/include/python3.5m")
set (libraries	${libraries} -L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu -L/usr/lib -lpython3.5m -lpthread -ldl  -lutil -lm -lcrypt)

add_library(ml_module 
	"${PROJECT_SOURCE_DIR}/src/ml_module.c"
	"${PROJECT_SOURCE_DIR}/src/SAQNAgent_thpt_var_min_mem.c"
	"${PROJECT_SOURCE_DIR}/src/A3CProcesses.c"
)

set_target_properties(ml_module PROPERTIES
    VERSION ${PROJECT_VERSION}
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/api/ml_module.h)


target_link_libraries(ml_module ${libraries})
target_link_libraries(ml_module ${ZeroMQ_LIBRARY})

target_include_directories(ml_module PRIVATE include)
target_include_directories(ml_module PRIVATE src)
target_include_directories(ml_module PUBLIC api)

if(MPI_COMPILE_FLAGS)
	set_target_properties(ml_module PROPERTIES 
		COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
	set_target_properties(ml_module PROPERTIES
		LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

install(TARGETS ml_module 
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	COMPONENT library)
