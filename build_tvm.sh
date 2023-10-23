#!/bin/bash

#Build tvm from scratch
mkdir tvm/build;
cp config.cmake tvm/build/config.cmake;
cd tvm/build;
cmake ..;
make -j8;