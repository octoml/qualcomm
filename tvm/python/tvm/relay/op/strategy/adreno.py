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
"""Definition of adreno operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import re
from tvm import topi
from .generic import *
from .. import op as _op

@conv2d_strategy.register("adreno")
def conv2d_strategy_adreno(attrs, inputs, out_type, target):
    """conv2d adreno strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    stride_h, stride_w = attrs.get_int_tuple("strides")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW4c" and kernel_layout == "OIHW4o":
            strategy.add_implementation(
                wrap_compute_conv2d(topi.adreno.conv2d_nchwc),
                wrap_topi_schedule(topi.adreno.schedule_conv2d_nchwc),
                name="conv2d_nchwc.opencl",
            )
    else:
        raise RuntimeError("group_conv2d is not yet supported for adreno")
    return strategy
