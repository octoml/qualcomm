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
    #parser.add_argument('-b', '--build', action="store_true", help='Whether to try to compile the test case; default is to lower only without compilation.')
    # parser.add_argument('-e', '--evaluate', action="store_true", help='Whether to evaluate the kernel and schedule.')
    # parser.add_argument('-v', '--verify', action="store_false", help='Whether to verify numerical results of evaluation.')
    # parser.add_argument('-B', '--batch_size', type=int, help='Batch size to use in batched gemm computation')
    parser.add_argument('-m', '--memory', type=str, default="texture", help='Use global or texture')
    parser.add_argument('-t', '--test', type=str, default="plus_one_rank3", choices=["plus_one_rank3", "plus_one_rank5", "matmul", "matmul_with_local"], help='Rank of the input tensor')
    # parser.add_argument('-N', type=int, help='Size of N for matrix B (KxN)')
    # parser.add_argument('-K', type=int, help='Size of reduction axis K')
    # parser.add_argument('-r', '--relay', action="store_true", help='Use relay for testing')
    parser.add_argument(
        "-r",
        "--rpc_tracker_host",
        type=str,
        default=None,
        help="RPC tracker host IP address",
    )
    parser.add_argument(
        "-p",
        "--rpc_tracker_port",
        type=str,
        default=None,
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

def compute(shape):
    X = te.placeholder(shape, name="X", dtype="float32")
    Y = te.compute(shape, lambda i, j, k: X[i, j, k] + 1, name="Compute_Y")
    return X, Y

def schedule(X, Y):
    s = te.create_schedule(Y.op)
    #Xt = s.cache_read(X, "texture", [Y])
    #Xt = s.cache_read(X, "global", [Y])
    Xt = s.cache_read(X, args.memory, [Y])

    # copy to texture stage
    x, y, c = s[Xt].op.axis
    s[Xt].bind(x, te.thread_axis("blockIdx.x"))
    s[Xt].bind(y, te.thread_axis("threadIdx.x"))
    s[Xt].vectorize(c)

    # the compute stage
    x, y, c = s[Y].op.axis
    xo, yo, xi, yi = s[Y].tile(x, y, 4, 4)
    s[Y].bind(xo, te.thread_axis("blockIdx.x"))
    s[Y].bind(yo, te.thread_axis("threadIdx.x"))
    s[Y].vectorize(c)
    return s

def compute5d(shape):
    X = te.placeholder(shape, name="X", dtype="float32")
    Y = te.compute(shape, lambda i, j, k, l, m: X[i, j, k, l, m] + 1, name="Compute_Y")
    return X, Y

def schedule5d(X, Y):
    s = te.create_schedule(Y.op)
    Xt = s.cache_read(X, args.memory, [Y])

    # copy to texture stage
    a, b, c, d, e = s[Xt].op.axis
    #ab = s[Xt].fuse(a, b)
    #cd = s[Xt].fuse(c, d)
    bcd = s[Xt].fuse(b, c, d)
    s[Xt].bind(a, te.thread_axis("blockIdx.x"))
    s[Xt].bind(bcd, te.thread_axis("threadIdx.x"))
    s[Xt].vectorize(e)

    # the compute stage
    a, b, c, d, e = s[Y].op.axis
    #ab = s[Y].fuse(a, b)
    #cd = s[Y].fuse(c, d)
    bcd = s[Y].fuse(b, c, d)
    #xo, yo, xi, yi = s[Y].tile(ab, cd, 4, 4)
    xo, yo, xi, yi = s[Y].tile(a, bcd, 4, 4)
    s[Y].bind(xo, te.thread_axis("blockIdx.x"))
    s[Y].bind(yo, te.thread_axis("threadIdx.x"))
    s[Y].vectorize(e)
    return s

def compute_matmul(shape):
    A = te.placeholder(shape, name="A", dtype="float32")
    B = te.placeholder(shape, name="B", dtype="float32")
    k = te.reduce_axis((0, shape[1]), name="k")
    C = te.compute(
        (shape[0]*shape[2], shape[0]*shape[2]),
        lambda i, j: te.sum(
            A[i//shape[2], k, i%shape[2]].astype("float") * B[j//shape[2], k, j%shape[2]].astype("float"), axis=[k]
        ),
        name="Compute_MatMul",
    )
    return A, B, C

def schedule_matmul(A, B, C, local=False):
    s = te.create_schedule(C.op)
    At = s.cache_read(A, args.memory, [C])
    Bt = s.cache_read(B, args.memory, [C])
    if local:
        Al = s.cache_read(At, "local", [C])
        Bl = s.cache_read(Bt, "local", [C])
    Cl = s.cache_write(C, "local")

    bx = te.thread_axis("blockIdx.x")
    tx = te.thread_axis("threadIdx.x")
    def copy_to_texture(stage):
        _io, _k, _ii = s[stage].op.axis
        s[stage].vectorize(_ii)
        s[stage].bind(_io, bx)
        s[stage].bind(_k, tx)

    copy_to_texture(At)
    copy_to_texture(Bt)

    # copy to global stage
    _i, _j = s[C].op.axis
    xo, yo, xi, yi = s[C].tile(_i, _j, 4, 4)
    s[C].unroll(xi)
    s[C].vectorize(yi)
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(yo, te.thread_axis("threadIdx.x"))

    # the compute stage
    s[Cl].compute_at(s[C], yo)
    (_k,) = Cl.op.reduce_axis
    _x, _y = s[Cl].op.axis
    s[Cl].reorder(_k, _x, _y)
    s[Cl].unroll(_x)
    s[Cl].vectorize(_y)

    if local:
        s[Al].compute_at(s[Cl], _k)
        s[Al].vectorize(s[Al].op.axis[-1])
        s[Bl].compute_at(s[Cl], _k)
        s[Bl].vectorize(s[Bl].op.axis[-1])

    return s


def test_texture(target="opencl", target_host="llvm -mtriple=arm64-linux-android"):
    if args.test == "plus_one_rank3":
        shape =(32, 32, 4)
        placeholders = compute(shape)
        s = schedule(*placeholders)
    elif args.test == "matmul":
        shape = (32, 64, 4)
        placeholders = compute_matmul(shape)
        s = schedule_matmul(*placeholders)
    elif args.test == "matmul_with_local":
        shape = (32, 64, 4)
        placeholders = compute_matmul(shape)
        s = schedule_matmul(*placeholders, local=True)
    elif args.test == "plus_one_rank5":
        shape =(32, 2, 4, 4, 4)
        placeholders = compute5d(shape)
        s = schedule5d(*placeholders)

    result = tvm.driver.lower(s, placeholders)
    print("tvm.lower:\n", result)

    func = tvm.driver.build(s, [*placeholders], target=target, target_host=target_host, name="TestFunction")
    temp = util.tempdir()
    dso_binary = "dev_lib_cl.so"
    dso_binary_path = temp.relpath(dso_binary)
    func.export_library(dso_binary_path, ndk.create_shared)
    print("OpenCL source:\n", func.imported_modules[0].get_source())
    print("Binary file located in: ", dso_binary_path)
    tracker, remote = get_remote()
    remote.upload(dso_binary_path)
    print("Uploading binary...")
    func = remote.load_module(dso_binary)
    ctx = remote.cl(0)

    args_tvm = []
    args_np = []
    for var in placeholders[:-1]:
        var_np = np.random.uniform(size=[i.value for i in var.shape]).astype(var.dtype)
        args_np.append(var_np)
        args_tvm.append(tvm.nd.array(var_np, ctx))
    args_tvm.append(tvm.nd.array(np.zeros([i.value for i in placeholders[-1].shape], dtype=placeholders[-1].dtype), ctx))
    func(*args_tvm)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=3)
    print("time:", "%f ms" % (evaluator(*args_tvm).mean * 1e3))
    if "plus_one" in args.test:
        np_result = args_np[0] + 1.0;
    elif "matmul" in args.test:
        np_result = np.matmul(args_np[0].transpose((0, 2, 1)).reshape(128, 64), args_np[1].transpose(1, 0, 2).reshape(64,128))
    np.testing.assert_allclose(args_tvm[-1].asnumpy(), np_result, rtol=1e-3, atol=1e-3)
    print("validation done")


if __name__ == "__main__":
    test_texture()
