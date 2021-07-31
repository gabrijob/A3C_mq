#!/bin/bash

rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile ml_module
cmake demo CMakeLists.txt
make
