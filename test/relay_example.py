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
"""
.. _tutorial-use-pass-infra:

How to Use TVM Pass Infra
=========================
**Author**: `Zhi Chen <https://github.com/zhiics>`_

As the number of optimization passes increases in Relay/tir, it becomes intractable to
execute them and maintain their dependencies manually. Therefore, we have
introduced an infrastructure to manage the optimization passes and make it
applicable to different layers of the IR in the TVM stack.

The optimizations of a Relay/tir program could be applied at various granularity,
namely function-level and module-level using :py:class:`tvm.relay.transform.FunctionPass`/
:py:class:`tvm.tir.transform.PrimFuncPass` and :py:class:`tvm.transform.ModulePass`
respectively. Or users can rely on :py:class:`tvm.transform.Sequential` to apply a sequence of passes
on a Relay/tir program where the dependencies between passes can be resolved by the
pass infra. For more details about each type of these passes, please refer to
the :ref:`pass-infra`

This tutorial mainly demonstrates how developers can use the pass infra to perform
a certain optimization and create an optimization pipeline for a Relay program.
The same approach can be used for tir as well.
"""


import numpy as np
import tvm
from tvm import te
import tvm.relay as relay

###############################################################################
# Create An Example Relay Program
# -------------------------------
# First of all, we create a simple Relay program for the tutorial. This program
# will be used by various optimizations of the examples in this tutorial.
# Similarly, users can write a tir primitive function and apply the tir passes.


def example():
    shape = (1, 64, 54, 54)
    c_data = np.empty(shape).astype("int8")
    c = relay.const(c_data,dtype="int8")
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    x = relay.var("x", relay.TensorType((1, 64, 56, 56), "int8"))
    conv = relay.nn.conv2d(x, weight)
    y = relay.add(c, c)
    y = relay.multiply(y, relay.const(2, "int8"))
    y = relay.add(conv, y)
    z = relay.add(y, c)
    z1 = relay.add(y, c)
    z2 = relay.add(z, z1)
    return relay.Function([x, weight], z2)


###############################################################################
# Optimize the Program
# --------------------
# Now we would like to optimize the program. Relay features a host of
# optimizations. We will select some of them to apply on this example program.
#
# There are multiple ways to optimize a Relay program. Below we will provide
# examples for each of them.
#
# Manually Apply Optimization Passes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Let's first create a relay Module which contains one or multiple Relay
# functions for optimization.
f = example()
mod = tvm.IRModule.from_expr(f)

# Now we can apply constant folding on the module.
# fold_const here is a callback that doesn't take any parameters.
fold_const = relay.transform.FoldConstant()
# Then, we can invoke the pass on the given module. Note that the constant
# folding pass works at the function-level. That being said, each function in
# the module will be applied with the optimization. Users don't need to iterate
# through individual functions manually to apply this pass.
mod = fold_const(mod)
# We can see from the updated program that the constants are folded.
print(mod)

###############################################################################
# More optimizations can be applied in the similar manner. For instance, we can
# eliminate the common expressions that used by `z` and `z1`.
mod = relay.transform.EliminateCommonSubexpr()(mod)
print(mod)
