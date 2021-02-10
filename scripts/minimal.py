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

import os
import numpy as np

import tvm
from tvm import relay
from tvm.topi import testing

def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Tune and/or evaluate a curated set of models"
    )
    parser.add_argument(
        "-k", "--rpc_key", type=str, default="android", help="RPC key to use"
    )
    parser.add_argument(
        "-r",
        "--rpc_tracker_host",
        type=str,
        default=os.environ["TVM_TRACKER_HOST"],
        help="RPC tracker host IP address",
    )
    parser.add_argument(
        "-p",
        "--rpc_tracker_port",
        type=str,
        default=os.environ["TVM_TRACKER_PORT"],
        help="RPC tracker host port",
    )

    args = parser.parse_args()
    if args.rpc_tracker_port != None:
        args.rpc_tracker_port = int(args.rpc_tracker_port)
    return args


args = get_args()

if __name__ == "__main__":
    input_shape = (1, 128, 112, 112)
    filter_shape = (128, 128, 3, 3)
    A = relay.var("data", shape=input_shape, dtype="float32")
    B = relay.var("weight", shape=filter_shape, dtype="float32")
    C = relay.nn.conv2d(A, B, kernel_size=(3,3))
    func = relay.Function([A, B], C)
    mod, params = relay.testing.init.create_workload(func)
    input_shape = {"data": input_shape}

    # When the target_host != local system an assertion fires in VM
    # indicating as much even though the host binary should be run on
    # a remote system via RPC.
    target, target_host = "llvm", "llvm -mtriple=arm64-linux-android"
    with relay.build_config(opt_level=3):
        exe = relay.vm.compile(
            mod, target_host=target_host, target=target, params=params
        )

    from tvm import rpc
    tracker = rpc.connect_tracker(args.rpc_tracker_host, args.rpc_tracker_port)
    remote = tracker.request(
        args.rpc_key, priority=0, session_timeout=600
    )
    ctx = remote.cpu(0)
    from tvm.runtime import vm
    m = vm.VirtualMachine(exe, ctx)

    inputs = []
    if isinstance(input_shape, dict):
        input_dict = {}
        for name in input_shape:
            inputs.append(np.random.normal(size=input_shape[name]).astype("float32"))
            input_dict[name] = tvm.nd.array(inputs[-1])
    else:
        inputs.append(np.random.normal(size=input_shape).astype("float32"))
        input_dict = {"data": tvm.nd.array(inputs[-1])}
    m.set_input("main", **input_dict)
    time_f = m.module.time_evaluator("invoke", ctx, number=10)
    cost = time_f("main").mean
    print("%g secs/iteration\n" % cost)
