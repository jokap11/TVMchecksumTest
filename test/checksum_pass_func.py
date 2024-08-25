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
from tvm.relax.testing import relay_translator
import tvm.relax as relax


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


def fork_structure():

    const0 = relay.ones(shape=(3, 3, 3, 16), dtype="int8")
    const1 = relay.ones(shape=(1, 1, 1, 16), dtype="int32")
    const2 = relay.ones(shape=(16), dtype="int32")
    const6 = relay.ones(shape=(1), dtype="int32")
    const7 = relay.ones(shape=(3, 3, 3, 16), dtype="int8")
    const8 = relay.ones(shape=(1, 1, 1, 16), dtype="int32")
    const9 = relay.ones(shape=(16), dtype="int32")

    const13 = relay.ones(shape=(3, 3, 3, 16), dtype="int8")
    const14 = relay.ones(shape=(1, 1, 1, 16), dtype="int32")
    const15 = relay.ones(shape=(16), dtype="int32")
    const19 = relay.ones(shape=(1), dtype="int32")


  
    x  = relay.var("data", shape=(1, 32, 32, 3), dtype="int8")
    args = [x]  


    y0  = relay.nn.pad(x, -128, pad_width=[[0, 0], [1, 1], [1, 1], [0, 0]])
    y1  = relay.nn.conv2d(y0, const0, padding=[0, 0, 0, 0], channels=16, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")
    y2  = relay.subtract(y1, const1)
    y3  = relay.nn.bias_add(y2, const2, axis=3)
    y4  = relay.cast(y3, dtype="int32")
    y6  = relay.add(-128, y4)
    y7  = relay.clip(y6, a_min=-128, a_max=127)
    y8  = relay.cast(y7, dtype="int8")
    y9  = relay.clip(y8, a_min=-128, a_max=127)
    y10 = relay.cast(y9, dtype="int32")
    y11 = relay.subtract(y10, const6)
    y12 = relay.fixed_point_multiply(y11, multiplier=1660533717, shift=0)
    y13 = relay.nn.pad(y9, -128, pad_width=[[0, 0], [1, 1], [1, 1], [0, 0]])
    y14 = relay.nn.conv2d(y13, const7, padding=[0, 0, 0, 0], channels=16, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")
    y15 = relay.subtract(y14, const8)
    y16 = relay.nn.bias_add(y15, const9, axis=3)
    y17 = relay.cast(y16, dtype="int32")
    y19 = relay.add(-128 , y17)
    y20 = relay.clip(y19, a_min=-128, a_max=127)
    y21 = relay.cast(y20, dtype="int8")
    y22 = relay.clip(y21, a_min=-128, a_max=127)
    y23 = relay.nn.pad(y22, -128, pad_width=[[0, 0], [1, 1], [1, 1], [0, 0]])
    y24 = relay.nn.conv2d(y23, const13, padding=[0, 0, 0, 0], channels=16, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")
    y25 = relay.subtract(y24, const14)
    y26 = relay.nn.bias_add(y25, const15, axis=3)
    y27 = relay.cast(y26, dtype="int32")
    y29 = relay.add(4 , y27)
    y30 = relay.clip(y29, a_min=-128, a_max=127)
    y31 = relay.cast(y30, dtype="int8")
    y32 = relay.cast(y31, dtype="int32")
    y33 = relay.subtract(y32, const19)
    y34 = relay.fixed_point_multiply(y33, multiplier=1098017566, shift=2)
    y35 = relay.add(-128 , y12)
    y36 = relay.add(-128 , y34)
    y = relay.add(y35, y36)
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



#result = evaluate()
#print("New Tensor:")
#print(result)

f = fork_structure()
mod = tvm.IRModule.from_expr(f)


inferType = relay.transform.InferType()
mod = inferType(mod)
#required to translated qnn.conv2d into legal relay operator
mod = relay.qnn.transform.CanonicalizeOps()(mod)
# Actual pass to add checksum calculation to each int8/int32 conv2D
#mod = relay.transform.Extend2DConv()(mod)
#required to check folded constant
#mod = relay.transform.FoldConstant()(mod)
mod = inferType(mod)

print("Print relay modulse:")
print(mod)

print("Now transformed relax module:")

relax_mod = relay_translator.from_relay(
    mod["main"], 
    target="llvm",
    pass_config={
        "relay.backend.use_meta_schedule": True,
        "relay.FuseOps.max_depth": 1,  # Disable relay fusion
    })
# print(relax_mod)

print("dfs")

relax_dfs= relax.transform.TopologicalSort("depth-first", "from-inputs")(relax_mod)

print(relax_dfs)

print("bfs")

relax_bfs= relax.transform.TopologicalSort("breadth-first", "from-inputs")(relax_mod)

print(relax_bfs)
