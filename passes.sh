#Check if virtual env already exists

if [ -d "venv" ]; then
source venv/bin/activate
else
python3 -m venv venv;
source venv/bin/activate;
pip3 install -r requirements.txt;
fi

# Debug just works with VLOG informations unfortunately
#TVM_LOG_DEBUG=DEFAULT=2 python3 test/relay_cond.py
#Run real units test:
cd tvm/tests/python/relay;
pytest test_pass_conv2d_checksum_extension.py