import numpy as np
import scipy
from scipy import special
import tvm
import tvm.relay as relay

# def example():
#     shape = (1, 64, 54, 54)
#     c_data = np.empty(shape).astype("int8")
#     c = relay.const(c_data,dtype="int8")
#     weight = relay.var("weight", shape=(64, 64, 3, 3),dtype="int8")
#     x = relay.var("x", relay.TensorType((1, 64, 56, 56), "int8"))
#     conv = relay.nn.conv2d(x, weight)
#     #weight2 = relay.multiply(weight, relay.const(5, "int8"))
#     conv2 = relay.nn.conv2d(x, weight)
#     y = relay.add(c, c)
#     y = relay.multiply(y, relay.const(2, "int8"))
#     y = relay.add(conv, y)
#     z = relay.add(y, c)
#     z1 = relay.add(y, c)
#     z2 = relay.add(z, z1)
#     z2 = relay.exp(z2)
#     mult_out = relay.Tuple([z2, conv, conv2])
#     return relay.Function([x, weight], mult_out)

def example():
    shape = (1, 64, 54, 54)
    c_data = np.empty(shape).astype("int8")
    c = relay.const(c_data,dtype="int8")
    weight = relay.var("weight", shape=(64, 64, 3, 3),dtype="int8")
    x = relay.var("x", relay.TensorType((1, 64, 56, 56), "int8"))
    conv = relay.nn.conv2d(x, weight)
    return relay.Function([x,weight], conv)

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
#print(mod)

inferType = relay.transform.InferType()
mod = inferType(mod)
# Now we can apply constant folding on the module.
# fold_const here is a callback that doesn't take any parameters.
convExt = relay.transform.Extend2DConv()
commonSub = relay.transform.EliminateCommonSubexpr()
foldConst = relay.transform.FoldConstant()
#fastMath = relay.transform.ToANormalForm()
# Then, we can invoke the pass on the given module. Note that the constant
# folding pass works at the function-level. That being said, each function in
# the module will be applied with the optimization. Users don't need to iterate
# through individual functions manually to apply this pass.
#mod = common_sub(mod)
mod = commonSub(mod)
mod = convExt(mod)
mod = commonSub(mod)
print(mod)

mod = foldConst(mod)
print(mod)
