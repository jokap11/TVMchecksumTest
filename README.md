# TVMchecksumTest

## Requirements:
 - sudo apt-get update
 - sudo apt-get install -y python3 python3-dev python3-setuptools pip gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

## Development Setup
- Make a Python `venv`: `python3 -m venv venv`
- Activate said `venv`: `source venv/bin/activate`
- Install pip deps for tvm: `pip install -r requirements.txt`

## Run the bash script to build tvm for llvm
 - ./build_tvm.sh
 - Get 2 environmental vars into your .bashrc file:
    export TVM_HOME=/path/to/tvm
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
    
## Use the python script to build a relay node and probably multiple passes
 - ./passes.sh


