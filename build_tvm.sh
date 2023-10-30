#!/bin/bash

#Build tvm from scratch
mkdir tvm/build;
cp config.cmake tvm/build/config.cmake;
cd tvm/build;
cmake -DCMAKE_BUILD_TYPE=Debug ..;
make -j8;