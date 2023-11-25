# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=line-too-long

#Check if virtual env already exists
if [ -d "venv" ]; then
source venv/bin/activate
else
python3 -m venv venv;
source venv/bin/activate;
pip3 install -r requirements.txt;
fi

# Debug just works with VLOG informations unfortunately
# To debug/test the script with multiple differenct inputs use
#TVM_LOG_DEBUG=DEFAULT=2 python3 test/checksum_pass_func.py
cd tvm/tests/python/relay;
pytest test_pass_conv2d_checksum_extension.py