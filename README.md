# TVMchecksumTest

## Requirements:
 - sudo apt update && sudo apt upgrade
 - sudo apt install python3 python3-dev python3-setuptools pip gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev llvm-14

## Development Setup
- Make a Python virtual environment: `python3 -m venv venv`
- Activate the virtual environment: `source venv/bin/activate`
- Install pip deps for tvm: `pip install -r requirements.txt`

## Run the bash script to build tvm for llvm
 - `./scripts/build_tvm.sh`
 - Get 2 environmental vars into your .bashrc file:
    `export TVM_HOME=/path/to/tvm`
 - For solely standard TVM usage:
    `export PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}`
 - Extend path for VTA:
    `export PYTHONPATH=/path/to/vta/python:${PYTHONPATH}`
    
## Use this python script for testing the checksum extension unit test pass
 - ./scripts/pass_test.sh


