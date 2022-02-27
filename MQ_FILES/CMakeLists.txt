project(message_queue C)
cmake_minimum_required(VERSION 3.0)
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
find_package(LIBM REQUIRED)
include_directories(${LIBM_INCLUDE_PATH})

# ML LIBRARIES
include_directories("/usr/include/python3.5m")
set (libraries	${libraries} -L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu -L/usr/lib -lpython3.5m -lpthread -ldl  -lutil -lm -lcrypt)

# A3C
include_directories("/usr/local/include")
link_directories("/usr/local/lib")
set (libraries	ml_module ${libraries} ${ZeroMQ_LIBRARY})

# A3C EXECUTABLE
add_executable (message_queue_4 message_queue_a3c_debug.c)

target_link_libraries(message_queue_4 ${libraries})

if(MPI_COMPILE_FLAGS)
	set_target_properties(message_queue_4 PROPERTIES
		COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
	set_target_properties(message_queue_4 PROPERTIES
		LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
