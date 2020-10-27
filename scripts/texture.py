# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os
import numpy as np

import tvm
import tvm.script
from tvm import te
from tvm import relay
from tvm.contrib import util, ndk

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Set test arguments')
    # parser.add_argument('-b', '--build', action="store_true", help='Whether to try to compile the test case; default is to lower only without compilation.')
    # parser.add_argument('-e', '--evaluate', action="store_true", help='Whether to evaluate the kernel and schedule.')
    # parser.add_argument('-v', '--verify', action="store_false", help='Whether to verify numerical results of evaluation.')
    # parser.add_argument('-B', '--batch_size', type=int, help='Batch size to use in batched gemm computation')
    # parser.add_argument('-M', type=int, help='Size of M for matrix A (MxK)')
    # parser.add_argument('-N', type=int, help='Size of N for matrix B (KxN)')
    # parser.add_argument('-K', type=int, help='Size of reduction axis K')
    # parser.add_argument('-r', '--relay', action="store_true", help='Use relay for testing')
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
    parser.add_argument(
        "-k", "--rpc_key", type=str, default="android", help="RPC key to use"
    )
    args = parser.parse_args()
    args.rpc_tracker_port = int(args.rpc_tracker_port)
    # if args.evaluate:
    #     args.build = True
    return args

args = get_args()


def get_remote():
    from tvm import rpc
    print(
        "Tracker attempting connection on {}:{}".format(
            args.rpc_tracker_host, args.rpc_tracker_port
        )
    )
    tracker = rpc.connect_tracker(args.rpc_tracker_host, args.rpc_tracker_port)
    remote = tracker.request(
        args.rpc_key, priority=0, session_timeout=600
    )
    print("Tracker connected to remote RPC server")
    return tracker, remote

def schedule(X, Y):
    s = te.create_schedule(Y.op)
    #Xc = s.cache_read(X, "global:texture", [Y])
    Xc = s.cache_read(X, "global", [Y])
    bx = te.thread_axis("blockIdx.x")
    tx = te.thread_axis("threadIdx.x")

    # copy to texture stage
    s[Xc].bind(s[Xc].op.axis[0], bx)
    s[Xc].bind(s[Xc].op.axis[1], tx)

    # the compute stage
    bx1 = te.thread_axis("blockIdx.x")
    tx1 = te.thread_axis("threadIdx.x")
    x, y, c = s[Y].op.axis
    xo, yo, xi, yi = s[Y].tile(x, y, 4, 4)
    s[Y].bind(xo, bx1)
    s[Y].bind(yo, tx1)
    s[Y].vectorize(yi)
    return s

def test_texture(target="opencl --device=mali", target_host="llvm -mtriple=arm64-linux-android"):
    shape =(32, 32, 4)
    X = te.placeholder(shape, name="X", dtype="float32")
    Y = te.compute(shape, lambda i, j, k: X[i, j, k] + 1, name="Compute_Y")

    s = schedule(X, Y)

    result = tvm.driver.lower(s, [X, Y])
    print("tvm.lower:\n", result)

    tracker, remote = get_remote()
    func = tvm.driver.build(s, [X, Y], target=target, target_host=target_host, name="texture_function")

    temp = util.tempdir()
    dso_binary = "dev_lib_cl.so"
    dso_binary_path = temp.relpath(dso_binary)
    func.export_library(dso_binary_path, ndk.create_shared)
    print("OpenCL source:\n", func.imported_modules[0].get_source())
    remote.upload(dso_binary_path)
    print("Uploading binary...")
    func = remote.load_module(dso_binary)

    ctx = remote.cl(0)
    x_np = np.random.uniform(size=shape).astype(X.dtype)
    x_tvm = tvm.nd.array(x_np, ctx)
    y_tvm = tvm.nd.array(np.zeros(shape, dtype=Y.dtype), ctx)
    func(x_tvm, y_tvm)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=3)
    print("time:", "%f ms" % (evaluator(x_tvm, y_tvm).mean * 1e3))

if __name__ == "__main__":
    test_texture()
