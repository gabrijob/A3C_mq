#!/bin/bash

rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile ml_module
cd build
cmake demo ../CMakeLists.txt
make
