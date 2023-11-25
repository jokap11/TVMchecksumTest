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
