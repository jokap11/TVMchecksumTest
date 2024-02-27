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

import numpy as np
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
from tvm.topi.testing import strided_slice_python, conv2d_nchw_python

def easy_2conv_example():
    w1 = relay.var("w1", shape=(1, 64, 3, 3), dtype="int8")
    w2 = relay.var("w2", shape=(1, 64, 8, 8), dtype="int8")
    x  = relay.var("x", shape=(1, 64, 56, 56), dtype="int8")   
    x1 = relay.multiply(x, relay.const(2,dtype="int8"))
    conv = relay.nn.conv2d(x1, w1, out_dtype="int32")
    conv2 = relay.nn.conv2d(x, w2, out_dtype="int32")
    y = relay.sum(conv,[0])
    mult_out = relay.Tuple([y,conv2])
    return relay.Function([x,w1,w2], mult_out)


def constant_fold_example_weight_propagation():
    c_data = np.empty(shape=(1, 64, 3, 3)).astype("int8")
    w1 = relay.const(c_data,dtype="int8")
    x  = relay.var("data", shape=(1, 64, 56, 56), dtype="int8")
    conv = relay.nn.conv2d(x, w1, out_dtype="int32")
    return relay.Function([x,w1], conv)

        
        
def complex_conv_in_deeper_tree_structure():
        x  = relay.var("data", shape=(1, 4, 20, 20), dtype="int8")
        w1 = relay.var("kernel", shape=(1, 4, 16, 16), dtype="int8")
        args = [x,w1]
        y0 = relay.ones(shape=(4,1,5,5), dtype="int8") 
        y1 = relay.nn.conv2d(x, y0, padding=(0,0), groups=4, channels=4, kernel_size=[5,5], out_dtype="int32")
        y2 = relay.cast(w1, dtype="int32") 
        y3 = relay.sum(y1, axis=[0], keepdims=True) 
        y4 = relay.sum(y2, axis=[0], keepdims=True) 
        y5 = relay.nn.conv2d(y3, y4, padding=(0,0), channels=1, groups=1, kernel_size=[16, 16], out_layout="NCHW", out_dtype="int64") 
        y6 = relay.nn.conv2d(x, w1, padding=(0,0), out_dtype="int32") 
        y7 = relay.cast(y6, dtype="int64")
        y8 = relay.sum(y5, axis=[0, 1, 2, 3])
        y9 = relay.sum(y7, axis=[0, 1, 2, 3])
        y10 = relay.not_equal(y8, y9)
        y = relay.Tuple([y6, y10])
        return relay.Function(args, y)



def qnn_2conv():
    weight = relay.var("weight", relay.TensorType((1, 37, 11, 11),  dtype="int8"))
    weight2 = relay.var("weight2", relay.TensorType((1, 37, 11, 11),  dtype="int8"))
    x = relay.var("x", relay.TensorType((1, 37, 56, 56), dtype="int8"))
    conv = relay.qnn.conv2d(x, weight,
            input_zero_point=relay.const(0),
            kernel_zero_point=relay.const(0),      
            input_scale=relay.const(1.0),
            kernel_scale=relay.const(1.0),
            kernel_size=(11, 11),
            channels=1,
            strides=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32")
    conv2 = relay.qnn.conv2d(x, weight2,
            input_zero_point=relay.const(0),
            kernel_zero_point=relay.const(0),      
            input_scale=relay.const(1.0),
            kernel_scale=relay.const(1.0),
            kernel_size=(11, 11),
            channels=1,
            strides=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32")
    result = relay.add(conv,conv2)
    return relay.Function([x,weight,weight2], result)


def strided_slice_example():
    x  = relay.var("data", shape=( 12, 12), dtype="int8")
    conv = relay.strided_slice(x, begin=[ 1, 1], end=[ 7, 7], strides=[1, 1], axes=[1,0],slice_mode="end")
    return relay.Function([x], conv)


def evaluate():
    #c_data = np.zeros(shape=(1, 64, 3, 3)).astype("int8")
    input_shape = (1 ,2, 6, 6)
    weight_shape = (1,2,4,4)
    #np.random.seed(seed=0)
    data = np.random.randint(-128,127,input_shape, dtype="int8")
    w1 = np.random.randint(-128,127,weight_shape, dtype="int8")

    print(data)
    conv = conv2d_nchw_python(data.astype("int32"),w1.astype("int32"),(2,2), 0)


    #weight size
    c = 2
    w = 4
    #striding
    s = 2

    # position in weight
    # pos_c = 0
    # pos_y = 0
    # pos_x = 1
    res = []

    #print(c_data[0])
    #print(c_data[1])
    print(f" Das ist das weight{w1}")
    # minimize batch size with summation
    b_data = data[0]
    
    for pos_c in range(c):
        layer = []
        for pos_y in range(w):
            for pos_x in range(w):
                slice = strided_slice_python(b_data[pos_c], begin=[pos_y, pos_x], end=[int(input_shape[2]-(w-pos_y)+1), int(input_shape[3]-(w-pos_x)+1)], strides=[s,s], axes=[0,1] ,slice_mode="end")
                #print(f"New Tensor: with c/x/y {pos_c}/{pos_x}/{pos_y}")
                #print(slice)
                slice = np.sum(slice)
                #print(slice)
                layer.append(slice)
        #print(layer)
        res.append(layer)
    input_checksum = np.reshape(res, (1,2,4,4))
    print(f"Input checksum {input_checksum}")
    tensor_dot =  conv2d_nchw_python(input_checksum.astype("int32"), w1.astype("int32"), (1,1), 0)
    #print(res)
    print(f"Das ist die COnv {conv}")
    print(f"Das ist das tensor dot {tensor_dot}")
    elem_wise = np.sum(conv)
    tensor_1dim = np.sum(tensor_dot)
    print(f"output checksum vs tensor tensor dot {elem_wise} == {tensor_1dim}")
    return res



result = evaluate()
#print("New Tensor:")
#print(result)

f = qnn_2conv()
mod = tvm.IRModule.from_expr(f)


inferType = relay.transform.InferType()
mod = inferType(mod)
#required to translated qnn.conv2d into legal relay operator
mod = relay.qnn.transform.CanonicalizeOps()(mod)
# Actual pass to add checksum calculation to each int8/int32 conv2D
mod = relay.transform.Extend2DConv()(mod)
#required to check folded constant
#mod = relay.transform.FoldConstant()(mod)
mod = inferType(mod)
print(mod)
