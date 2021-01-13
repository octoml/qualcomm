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
from tvm import autotvm
from tvm import te
from tvm import relay
from tvm.contrib import util, ndk
from tvm.topi import testing
from tvm.topi.util import get_const_tuple, simplify
from tvm.topi import nn
from tvm.topi.mali import tile_and_bind3d

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Set test arguments')
    #parser.add_argument('-b', '--build', action="store_true", help='Whether to try to compile the test case; default is to lower only without compilation.')
    # parser.add_argument('-e', '--evaluate', action="store_true", help='Whether to evaluate the kernel and schedule.')
    # parser.add_argument('-v', '--verify', action="store_false", help='Whether to verify numerical results of evaluation.')
    # parser.add_argument('-B', '--batch_size', type=int, help='Batch size to use in batched gemm computation')
    parser.add_argument('-m', '--memory', type=str, default="texture", help='Use global or texture')
    parser.add_argument('-t', '--test', type=str, required=True, help='Selected test to run')
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

# TODO(csullivan): Improve executor class to be able to consume input values
# and return outputs, then refactor to use Executor in this test file
def tune_tasks(
        tasks,
        measure_option,
        tuner="xgb",
        n_trial=1024,
        early_stopping=None,
        log_filename="tuning.log",
        use_transfer_learning=True,
    ):
        from tvm.autotvm.tuner import XGBTuner
        from tvm.autotvm.tuner import GATuner

        tmp_log_file = log_filename + ".tmp"
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)

        for i, tsk in enumerate(reversed(tasks)):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            if tuner == "xgb" or tuner == "xgb-rank":
                tuner_obj = XGBTuner(tsk, loss_type="rank")
            elif tuner == "xgb_knob":
                tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
            elif tuner == "ga":
                tuner_obj = GATuner(tsk, pop_size=50)
            elif tuner == "random":
                tuner_obj = RandomTuner(tsk)
            elif tuner == "gridsearch":
                tuner_obj = GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            if use_transfer_learning:
                if os.path.isfile(tmp_log_file):
                    tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

            tsk_trial = min(n_trial, len(tsk.config_space))
            tuner_obj.tune(
                n_trial=tsk_trial,
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                    autotvm.callback.log_to_file(tmp_log_file),
                ],
            )

        autotvm.record.pick_best(tmp_log_file, log_filename)
        os.remove(tmp_log_file)

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
    abc = s[Xt].fuse(a, b, c)
    s[Xt].bind(abc, te.thread_axis("blockIdx.x"))
    s[Xt].bind(d, te.thread_axis("threadIdx.x"))
    s[Xt].vectorize(e)

    # the compute stage
    a, b, c, d, e = s[Y].op.axis
    abc = s[Y].fuse(a, b, c)
    xo, yo, xi, yi = s[Y].tile(abc, d, 4, 4)
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


def compute_matmul_inner(shape):
    A = te.placeholder(shape, name="A", dtype="float32")
    B = te.placeholder(shape, name="B", dtype="float32")
    k = te.reduce_axis((0, shape[1]*shape[2]), name="k")
    # (M, K) x (N, K)
    # (32, 256) x (32, 256)
    # (32, 64, 4) x (32, 64, 4)
    C = te.compute(
        (shape[0], shape[0]),
        lambda i, j: te.sum(
            A[i, k//shape[2], k%shape[2]].astype("float") * B[j, k//shape[2], k%shape[2]].astype("float"), axis=[k]
        ),
        name="Compute_MatMul",
    )
    return A, B, C

def schedule_matmul_inner(A, B, C, local=False):
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
        _i, _ko, _ki = s[stage].op.axis
        s[stage].vectorize(_ki)
        s[stage].bind(_i, bx)
        s[stage].bind(_ko, tx)

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
    s[Cl].reorder(_x, _y, _k)
    s[Cl].unroll(_x)
    # TODO(csullivan): consider whether the below error is worth resolving
    # s[Cl].vectorize(_y) # error

    if local:
        s[Al].compute_at(s[Cl], _x)
        s[Al].vectorize(s[Al].op.axis[-1])
        s[Bl].compute_at(s[Cl], _x)
        s[Bl].vectorize(s[Bl].op.axis[-1])

    return s

def compute_matmul_vector_accumulator(shapeA, shapeB):
    # A x B
    # (K/4, M, K%4) x (K, N/4, N%4) = (M, N)
    # (32, 64, 4) x (128, 16, 4) = (64, 64)
    A = te.placeholder(shapeA, name="A", dtype="float32")
    B = te.placeholder(shapeB, name="B", dtype="float32")
    k = te.reduce_axis((0, shapeB[0]), name="k")
    C = te.compute(
        (shapeA[1], shapeB[1]*shapeB[2]),
        lambda i, j: te.sum(
            A[k//shapeA[-1], i, k%shapeA[-1]].astype("float") * B[k, j//shapeB[-1], j%shapeB[-1]].astype("float"), axis=[k]
        ),
        name="Compute_MatMul",
    )
    return A, B, C

def schedule_matmul_vector_accumulator(A, B, C, local=False):
    s = te.create_schedule(C.op)
    At = s.cache_read(A, args.memory, [C])
    Bt = s.cache_read(B, args.memory, [C])
    if local:
        Al = s.cache_read(At, "local", [C])
        Bl = s.cache_read(Bt, "local", [C])
    Cl = s.cache_write(C, "local")

    def copy_to_texture(stage):
        _y, _x, _v = s[stage].op.axis
        # TODO(csullivan): removing this vectorize results in numerical errors, autovectorize
        s[stage].vectorize(_v)
        s[stage].bind(_y, te.thread_axis("blockIdx.x"))
        s[stage].bind(_x, te.thread_axis("threadIdx.x"))

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
    _a, _b = s[Cl].op.axis
    _ko, _ki = s[Cl].split(_k, factor=4)
    s[Cl].reorder(_ko, _a, _ki, _b)
    s[Cl].unroll(_ki)
    s[Cl].unroll(_a)
    s[Cl].vectorize(_b)

    if local:
        s[Al].compute_at(s[Cl], _a)
        _aa, _ka, _ba = s[Al].op.axis
        # TODO(csullivan)[BEFORE PR]: removing this vectorize command causes a crash. This needs to be autovectorized.
        s[Al].vectorize(_ba)
        s[Bl].compute_at(s[Cl], _ko)
        _ab, _kb, _bb = s[Bl].op.axis
        s[Bl].vectorize(_bb)
        s[Bl].unroll(_ab)

    return s

def schedule_matmul_vector_accumulator_autotvm(A, B, C):
    s = te.create_schedule(C.op)
    cfg = autotvm.get_config()

    At = s.cache_read(A, args.memory, [C])
    Bt = s.cache_read(B, args.memory, [C])
    Al = s.cache_read(At, "local", [C])
    Bl = s.cache_read(Bt, "local", [C])
    Cl = s.cache_write(C, "local")

    def copy_to_texture(stage):
        _y, _x, _v = s[stage].op.axis
        s[stage].vectorize(_v)
        s[stage].bind(_y, te.thread_axis("blockIdx.x"))
        s[stage].bind(_x, te.thread_axis("threadIdx.x"))

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
    _a, _b = s[Cl].op.axis
    _ko, _ki = s[Cl].split(_k, factor=4)

    s[Cl].reorder(_ko, _a, _ki, _b)
    cfg.define_knob("unroll", [0, 1])
    if cfg["unroll"] == 1:
        s[Cl].unroll(_ki)
        s[Cl].unroll(_a)
    s[Cl].vectorize(_b)

    s[Al].compute_at(s[Cl], _a)
    _aa, _ka, _ba = s[Al].op.axis
    s[Al].vectorize(_ba)
    s[Bl].compute_at(s[Cl], _ko)
    _ab, _kb, _bb = s[Bl].op.axis
    s[Bl].vectorize(_bb)
    s[Bl].unroll(_ab)


    return s

def compute_conv2d_1x1_NCHWc_RSCKk(input_shape, filter_shape):
    # conv2d( [N, C, H, W, c] , [1, 1, C, K, k]
    data = te.placeholder(input_shape, name="data", dtype="float32")
    filt = te.placeholder(filter_shape, name="filter", dtype="float32")
    c = te.reduce_axis((0, input_shape[1]), name="C")
    c4 = te.reduce_axis((0, input_shape[-1]), name="c4")
    kh = te.reduce_axis((0, filter_shape[0]), name="kh")
    kw = te.reduce_axis((0, filter_shape[1]), name="kw")
    conv = te.compute(
        (input_shape[0], filter_shape[-2], input_shape[2], input_shape[3], filter_shape[-1]),
        lambda n, ko, i, j, ki: te.sum(
            data[n, c, i, j, c4].astype("float") * filt[kh, kw, c*input_shape[-1] + c4, ko, ki].astype("float"), axis=[kh, kw, c, c4]
        ),
        #name="Compute_conv2d_1x1_NCHWc_RSCKk",
        name = "conv2d_1x1"
    )
    return data, filt, conv

def schedule_conv2d_1x1_NCHWc_RSCKk(data, filt, conv):
    # inputs: (1, 128//4, 56, 56, 4), (1, 1, 128, 128//4, 4)
    # outputs:
    s = te.create_schedule(conv.op)
    A, B, C = data, filt, conv
    At = s.cache_read(A, args.memory, [C])
    Bt = s.cache_read(B, args.memory, [C])
    Al = s.cache_read(At, "local", [C])
    Bl = s.cache_read(Bt, "local", [C])
    Cl = s.cache_write(C, "local")

    def copy_to_texture(stage):
        axes = s[stage].op.axis
        fused = s[stage].fuse(*axes[:-1])
        block, thread = s[stage].split(fused, factor=32)
        s[stage].vectorize(axes[-1])
        s[stage].bind(block, te.thread_axis("blockIdx.x"))
        s[stage].bind(thread, te.thread_axis("threadIdx.x"))
    copy_to_texture(At)
    copy_to_texture(Bt)

    _n, _ko, _h, _w, _ki = s[C].op.axis
    s[C].vectorize(_ki)
    s[C].bind(_n, te.thread_axis("blockIdx.x"))
    s[C].bind(_ko, te.thread_axis("threadIdx.x"))

    s[Cl].compute_at(s[C], _w)
    _nl, _kol, _hl, _wl, _kil = s[Cl].op.axis
    _khl, _kwl, _cl, _cl4 = s[Cl].op.reduce_axis
    _clo, _cli = s[Cl].split(_cl, factor=4)
    s[Cl].reorder(_clo, _cli, _cl4, _kil)
    s[Cl].unroll(_cli)
    s[Cl].unroll(_cl4)
    s[Cl].vectorize(_kil)

    s[Al].compute_at(s[Cl], _cli)
    s[Al].vectorize(s[Al].op.axis[-1])
    s[Bl].compute_at(s[Cl], _kwl)
    s[Bl].vectorize(s[Bl].op.axis[-1])

    return s


def compute_conv2d_1x1_WCHNc_CRSKk(input_shape, filter_shape):
    # input_shape = [W, C, H, N, c] -> [W, C, H*N, c]
    # filter_shape = [C, R, S, K, k] -> [C, R*S*K, k]
    # output_shape: [WK, HN, k] -> [W, K, H, N, k]
    data = te.placeholder(input_shape, name="data", dtype="float32")
    filt = te.placeholder(filter_shape, name="filter", dtype="float32")

    packed_data = te.compute(
        (input_shape[0], input_shape[1], input_shape[2] * input_shape[3], input_shape[4]),
        lambda i, j, k, l: data[i, j, k//input_shape[3], k%input_shape[3], l],
        name = "packed_data"
    )

    packed_filter = te.compute(
        (filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3], filter_shape[4]),
        lambda i, j, k: filt[i, j//(filter_shape[3] * filter_shape[2]), (j//filter_shape[3])%filter_shape[2], j%filter_shape[3], k],
        name = "packed_filter"
    )

    c = te.reduce_axis((0, input_shape[1]), name="C")
    c4 = te.reduce_axis((0, input_shape[-1]), name="c4")
    r = te.reduce_axis((0, filter_shape[1]), name="r")
    s = te.reduce_axis((0, filter_shape[2]), name="s")

    conv = te.compute(
        (input_shape[0], filter_shape[3], input_shape[2], input_shape[3], filter_shape[4]),
        lambda w, ko, h, n, ki: te.sum(
            packed_data[w, c, h * input_shape[3] + n, c4].astype("float")
            *
            packed_filter[c*input_shape[-1] + c4, ((r * filter_shape[2]) + s) * filter_shape[3] + ko, ki].astype("float"), axis=[r, s, c, c4]
        ),
        name = "conv2d_1x1"
    )
    return data, filt, packed_data, packed_filter, conv

def schedule_conv2d_1x1_WCHNc_CRSKk(data, filt, packed_data, packed_filter, conv):
    # data: [W, C, H*N, c]
    # filter: [C, R*S*K, k]
    # output: [W, K, H, N, k]

    # conv2d( [N, C, H, W, c] , [1, 1, C, K, k]
    # inputs: (1, 128//4, 56, 56, 4), (1, 1, 128, 128//4, 4)

    # data: (56, 128//4, 56*1, 4) = (56, 32, 56, 4)
    # filt: (128, 1*1*128//4, 4) = (128, 32, 4)
    # conv: (56, 32, 56, 1, 4)
    s = te.create_schedule(conv.op)
    cfg = autotvm.get_config()

    s[packed_data].compute_inline()
    s[packed_filter].compute_inline()
    A, B, C = packed_data, packed_filter, conv
    At = s.cache_read(A, args.memory, [C])
    Bt = s.cache_read(B, args.memory, [C])
    Al = s.cache_read(At, "local", [C])
    Bl = s.cache_read(Bt, "local", [C])
    Cl = s.cache_write(C, "local")

    def copy_to_texture(stage):
        axes = s[stage].op.axis
        fused = s[stage].fuse(*axes[:-1])
        block, thread = s[stage].split(fused, factor=32)
        s[stage].vectorize(axes[-1])
        s[stage].bind(block, te.thread_axis("blockIdx.x"))
        s[stage].bind(thread, te.thread_axis("threadIdx.x"))
    copy_to_texture(At)
    copy_to_texture(Bt)

    _w, _ko, _h, _n, _ki = s[C].op.axis
    kernel_scope, _n = s[C].split(_n, nparts=1)

    #_kh, _kw, _c, _c4 = s[C].op.reduce_axis
    # doesn't work as fused itervar has extent = -1
    #_hn = s[C].fuse(_h, _n)

    cfg.define_split("tile_f", _ko, num_outputs=4)
    cfg.define_split("tile_w", _w, num_outputs=4)
    cfg.define_split("tile_h", _h, num_outputs=4)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])


    bk, vk, tk, ki = cfg["tile_f"].apply(s, C, _ko)
    bw, vw, tw, wi = cfg["tile_w"].apply(s, C, _w)
    bh, vh, th, hi = cfg["tile_h"].apply(s, C, _h)
    s[C].reorder(bh, _n, vh, th, hi)
    bhn = s[C].fuse(bh, _n)

    s[C].bind(bk, te.thread_axis("blockIdx.z"))
    s[C].bind(bhn, te.thread_axis("blockIdx.y"))
    s[C].bind(bw, te.thread_axis("blockIdx.x"))
    s[C].bind(vk, te.thread_axis("vthread"))
    s[C].bind(vh, te.thread_axis("vthread"))
    s[C].bind(vw, te.thread_axis("vthread"))
    s[C].bind(tk, te.thread_axis("threadIdx.z"))
    s[C].bind(th, te.thread_axis("threadIdx.y"))
    s[C].bind(tw, te.thread_axis("threadIdx.x"))
    s[C].reorder(bw, bk, bhn, vw, vk, vh, tw, tk, th, ki, hi, wi, _ki)
    s[C].vectorize(_ki)

    # TODO(csullivan): Try uneven workgroup split
    # _wo, _wi = s[C].split(_w, factor=4)
    # #_hno, _hni = s[C].split(_hn, factor=8)
    # #s[C].reorder(_wo, _wi, _ko, _hno, _hni, _ki)
    # s[C].reorder(_wo, _ko, _hn, _ki, _wi)
    # s[C].unroll(_wi)

    # # mace:
    # # const int out_ch_blk = get_global_id(0);
    # # const int out_w_blk = get_global_id(1);
    # # const int out_hb = get_global_id(2);

    # bx = te.thread_axis("blockIdx.x")
    # by = te.thread_axis("blockIdx.y")
    # bz = te.thread_axis("blockIdx.z")
    # s[C].bind(_ko, bx)
    # s[C].bind(_wo, by)
    # s[C].bind(_hn, bz)

    #s[Cl].compute_at(s[C], _hn)
    s[Cl].compute_at(s[C], th)

    _wl, _kol, _hl, _nl, _kil = s[Cl].op.axis
    _khl, _kwl, _cl, _cl4 = s[Cl].op.reduce_axis

    cfg.define_split("tile_c", _cl, num_outputs=2)
    cfg.define_split("tile_kh", _khl, num_outputs=2)
    cfg.define_split("tile_kw", _kwl, num_outputs=2)



    _clo, _cli = cfg["tile_c"].apply(s, Cl, _cl)
    _khlo, _khli = cfg["tile_kh"].apply(s, Cl, _khl)
    _kwlo, _kwli = cfg["tile_kw"].apply(s, Cl, _kwl)
    #s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x)
    s[Cl].reorder(_clo, _khlo, _kwlo, _cli, _cl4, _khli, _kwli, _kol, _hl, _nl, _kil, _wl)
    #s[Cl].reorder(_clo, _khlo, _kwlo, _cli, _cl4, _khli, _kwli)
    # s[Cl].reorder(_cl, _cl4, _kil, _wl)
    s[Cl].unroll(_cl4)
    s[Cl].unroll(_wl)
    s[Cl].vectorize(_kil)


    _wla, _cla, _hnla, _cl4a = s[Al].op.axis
    s[Al].compute_at(s[Cl], _cli)
    s[Al].vectorize(_cl4a)
    s[Al].unroll(_wla)

    _clb, _rskolb, _kilb = s[Bl].op.axis
    s[Bl].compute_at(s[Cl], _cli)
    s[Bl].vectorize(_kilb)
    s[Bl].unroll(_clb)

    s[C].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[C].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    WO, K, HO, N, K4 = get_const_tuple(C.shape)
    RSC, _, _ = get_const_tuple(B.shape)
    cfg.add_flop(2 * N * K * K4 * HO * WO * RSC)

    return s


def compute_conv2d_mali_NCHW_KCRS(cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile):
    """compute define for Conv2d Spatial Pack with NCHW layout"""
    out_dtype = out_dtype or data.dtype
    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(N, tvm.tir.Any):
        N = tvm.te.size_var("n")
    if not isinstance(IH, int) or not isinstance(IW, int):
        raise RuntimeError("ARM winograd conv2d doesn't support dynamic input height or width.")

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:
        pre_packed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        pre_packed = True
        CO, _, KH, KW, VC = get_const_tuple(kernel.shape)
        CO = CO * VC

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1
    pad_top, pad_left, pad_bottom, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    OH = (IH + pad_top + pad_bottom - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1
    data_pad = nn.pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_bottom, pad_right])

    # ==================== define configuration space ====================
    # TODO(@kevinthesun): Support tuning/optimization for dynamic shape.
    n_tuning_axis = N if isinstance(N, int) else 1
    n, co, oh, ow = cfg.axis(n_tuning_axis), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    if num_tile == 2:  # for arm cpu
        co, vc = cfg.define_split("tile_co", co, num_outputs=2)
        oh, vh = cfg.define_split("tile_oh", oh, num_outputs=2)
        ow, vw = cfg.define_split("tile_ow", ow, num_outputs=2)
    elif num_tile == 3:  # for mali gpu
        co, _, vc = cfg.define_split("tile_co", co, num_outputs=3)
        oh, _, vh = cfg.define_split("tile_oh", oh, num_outputs=3)
        ow, _, vw = cfg.define_split("tile_ow", ow, num_outputs=3)
    else:
        raise RuntimeError("Invalid num_tile")

    cfg.define_reorder(
        "reorder_0",
        [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
        policy="candidate",
        candidate=[
            [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
            [n, co, oh, ow, ci, kh, kw, vc, vh, vw],
        ],
    )

    cfg.define_annotate("ann_reduce", [kh, kw], policy="try_unroll")
    cfg.define_annotate("ann_spatial", [vh, vw, vc], policy="try_unroll_vec")

    # fallback support
    if cfg.is_fallback:
        if num_tile == 2:  # arm cpu
            ref_log = autotvm.tophub.load_reference_log(
                "arm_cpu", "rk3399", "conv2d_nchw_spatial_pack.arm_cpu"
            )
            cfg.fallback_with_reference_log(ref_log)
        elif num_tile == 3:  # mali gpu
            ref_log = autotvm.tophub.load_reference_log(
                "mali", "rk3399", "conv2d_nchw_spatial_pack.mali"
            )
            cfg.fallback_with_reference_log(ref_log)
    # ====================================================================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    kvshape = (CO // VC, CI, KH, KW, VC)
    ovshape = (N, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (N, CO, OH, OW)

    if dilation_h != 1 or dilation_w != 1:
        # undilate input data
        dvshape = (N, OH // VH, OW // VW, CI, KH, KW, VH, VW)
        data_vec = te.compute(
            dvshape,
            lambda n, h, w, ci, kh, kw, vh, vw: data_pad[n][ci][
                (h * VH + vh) * HSTR + kh * dilation_h
            ][(w * VW + vw) * WSTR + kw * dilation_w],
            name="data_vec_undilated",
        )
    else:
        dvshape = (N, OH // VH, OW // VW, CI, VH * HSTR + KH - 1, VW * WSTR + KW - 1)
        data_vec = te.compute(
            dvshape,
            lambda n, h, w, ci, vh, vw: data_pad[n][ci][h * VH * HSTR + vh][w * VW * WSTR + vw],
            name="data_vec",
        )

    if autotvm.GLOBAL_SCOPE.in_tuning:
        # use "kernel_autotvm" instead of "kernel" to avoid naming conflict with OpenCL keyword
        kernel_vec = tvm.te.placeholder(kvshape, kernel.dtype, name="kernel_autotvm")
    else:
        if pre_packed:
            kernel_vec = kernel
        else:
            kernel_vec = te.compute(
                kvshape,
                lambda co, ci, kh, kw, vc: kernel[co * VC + vc][ci][kh][kw],
                name="kernel_vec",
            )

    ci = te.reduce_axis((0, CI), name="ci")
    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    if dilation_h != 1 or dilation_w != 1:
        conv = te.compute(
            ovshape,
            lambda n, co, h, w, vh, vw, vc: te.sum(
                data_vec[n, h, w, ci, kh, kw, vh, vw].astype(out_dtype)
                * kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                axis=[ci, kh, kw],
            ),
            name="conv",
        )
    else:
        conv = te.compute(
            ovshape,
            lambda n, co, h, w, vh, vw, vc: te.sum(
                data_vec[n, h, w, ci, vh * HSTR + kh, vw * WSTR + kw].astype(out_dtype)
                * kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                axis=[ci, kh, kw],
            ),
            name="conv",
        )

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    output = te.compute(
        oshape,
        lambda n, co, h, w: conv[
            n,
            idxdiv(co, VC),
            idxdiv(h, VH),
            idxdiv(w, VW),
            idxmod(h, VH),
            idxmod(w, VW),
            idxmod(co, VC),
        ],
        name="output_unpack",
        tag="spatial_conv2d_output",
    )
    return data_vec, kernel_vec, output, conv


def schedule_conv2d_mali_NCHW_KCRS(cfg, s, output, conv, data_vec, kernel_vec):
    """schedule the spatial packing for conv2d"""
    inputs = s[data_vec].op.input_tensors
    if len(inputs) == 0:
        data = data_vec
    else:
        data = inputs[0]

    max_unroll = 16
    vec_size = [1, 2, 4, 8, 16]
    # get tunable parameters (they are defined in compute)
    BC, TC, VC = cfg["tile_co"].size
    BH, TH, VH = cfg["tile_oh"].size
    BW, TW, VW = cfg["tile_ow"].size

    # schedule padding
    if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
        data_pad = data
        s[data_pad].compute_inline()

    # schedule data packing
    if isinstance(data_vec.op, tvm.te.ComputeOp) and data_vec.op.name == "data_vec_undilated":
        _, h, w, ci, _, _, vh, vw = s[data_vec].op.axis
    else:
        _, h, w, ci, vh, vw = s[data_vec].op.axis
    tile_and_bind3d(s, data_vec, h, w, ci, 1)
    if vh.dom.extent.value < max_unroll:
        s[data_vec].unroll(vh)
    if vw.dom.extent.value < max_unroll:
        s[data_vec].unroll(vw)

    if isinstance(kernel_vec.op, tvm.te.ComputeOp) and kernel_vec.name == "kernel_vec":
        if not autotvm.GLOBAL_SCOPE.in_tuning:
            max_threads = tvm.target.Target.current(allow_none=False).max_num_threads
            co, ci, kh, kw, vc = s[kernel_vec].op.axis
            fused = s[kernel_vec].fuse(co, ci, kh, kw, vc)
            fused, vec = s[kernel_vec].split(fused, VC)
            bb, tt = s[kernel_vec].split(fused, max_threads)
            s[kernel_vec].bind(bb, te.thread_axis("blockIdx.x"))
            s[kernel_vec].bind(tt, te.thread_axis("threadIdx.x"))
            if VC in vec_size:
                s[kernel_vec].vectorize(vec)

    # schedule convolution
    n, c, h, w, vh, vw, vc = s[conv].op.axis
    kc, kh, kw = s[conv].op.reduce_axis

    cfg["reorder_0"].apply(s, conv, [n, c, h, w, kc, kh, kw, vh, vw, vc])
    tile_and_bind3d(s, conv, c, h, w, TC, TH, TW)

    cfg["ann_reduce"].apply(
        s,
        conv,
        [kh, kw],
        axis_lens=[nn.get_const_int(kernel_vec.shape[2]), nn.get_const_int(kernel_vec.shape[3])],
        max_unroll=max_unroll,
    )

    cfg["ann_spatial"].apply(
        s,
        conv,
        [vh, vw, vc],
        axis_lens=[VH, VW, VC],
        max_unroll=max_unroll,
        vec_size=vec_size,
        cfg=cfg,
    )

    # schedule output
    if output.op not in s.outputs:  # has bias
        s[output].compute_inline()
        output = s.outputs[0]

    _, co, oh, ow = s[output].op.axis
    tile_and_bind3d(s, output, co, oh, ow, TC, TH, TW)

    return s


def compute_conv2d_cuda_NCHW_KCRS(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = nn.pad(Input, pad_before, pad_after, name="pad_temp")
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: te.sum(
            temp[nn, rc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w].astype(
                out_dtype
            )
            * Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx],
        ),
        tag="conv2d_nchw",
    )


def schedule_conv2d_cuda_NCHW_KCRS(cfg, s, conv):
    """schedule optimized for batch size = 1"""

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.Target.current()
    if target.kind.name in ["nvptx", "rocm"]:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])

    # fallback support
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            target.kind.name, target.model, "conv2d_nchw.cuda"
        )
        cfg.fallback_with_reference_log(ref_log)
    ##### space definition end #####

    pad_data, kernel = s[conv].op.input_tensors

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, "local")
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope("local")
        OL = conv

    # create cache stage
    AA = s.cache_read(pad_data, "shared", [OL])
    WW = s.cache_read(kernel, "shared", [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg["tile_rc"].apply(s, OL, rc)
    ryo, ryi = cfg["tile_ry"].apply(s, OL, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    N, CO, OH, OW = get_const_tuple(output.shape)
    _, KH, KW, CI = get_const_tuple(kernel.shape)

    if isinstance(N, int):
        cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW)

def compute_conv2d_NCHWc_KCRSk(
    cfg, data, kernel, stride, padding, dilation, out_dtype="float32"
):
    """Convolution operator for 'conv2d_NCHWc_KCRSk'.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] or
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
        6-D with shape [num_filter_chunk, in_channel_chunk, filter_height,
        filter_width, num_filter_block, in_channel_block]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_channel, out_height, out_width, out_channel_block]
    """
    ic_block_factor = 4
    oc_block_factor = 4

    pre_computed = len(kernel.shape) == 5
    if not pre_computed:
        batch, channels, height, width = get_const_tuple(data.shape)
        out_channels, in_channels, kernel_h, kernel_w = get_const_tuple(kernel.shape)

        assert (
            channels % ic_block_factor == 0
        ), "Number of input channels must divide {}".format(ic_block_factor)
        assert (
            out_channels % oc_block_factor == 0
        ), "Number of output channels must divide {}".format(oc_block_factor)

        packed_data = te.compute(
            (batch, channels // ic_block_factor, height, width, ic_block_factor),
            lambda n, c, h, w, vc: data[n, c * ic_block_factor + vc, h, w],
            name="packed_data",
        )
        packed_kernel = te.compute(
            (
                out_channels // oc_block_factor,
                in_channels,
                kernel_h,
                kernel_w,
                oc_block_factor
            ),
            lambda oc_chunk, ic, kh, kw, oc_block: kernel[
                oc_chunk * oc_block_factor + oc_block, ic, kh, kw
            ],
            name="packed_kernel",
        )
    else:
        packed_data = data
        packed_kernel = kernel

    batch, ic_chunk, in_height, in_width, ic_block = get_const_tuple(packed_data.shape)
    oc_chunk, _, kernel_h, kernel_w, oc_block = get_const_tuple(packed_kernel.shape)

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    # pad the input data
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(padding, (kernel_h, kernel_w))
    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down, pad_right, 0]
    pad_data = nn.pad(packed_data, pad_before, pad_after, name="pad_data")

    # compute the output shape
    out_height = (in_height - (kernel_h - 1) * dilation_h - 1 + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - (kernel_w - 1) * dilation_w - 1 + pad_left + pad_right) // stride_w + 1

    oshape = (batch, oc_chunk, out_height, out_width, oc_block)

    icc = te.reduce_axis((0, ic_chunk), name="ic_chunk")
    icb = te.reduce_axis((0, ic_block_factor), name="ic_block")
    kh = te.reduce_axis((0, kernel_h), name="kh")
    kw = te.reduce_axis((0, kernel_w), name="kw")

    conv = te.compute(
        oshape,
        lambda n, occ, oh, ow, ocb: te.sum(
            pad_data[
                n,
                icc,
                oh * stride_h + kh * dilation_h,
                ow * stride_w + kw * dilation_w,
                icb,
            ].astype(out_dtype)
            * packed_kernel[occ, icc * ic_block + icb, kh, kw, ocb].astype(out_dtype),
            axis=[icc, kh, kw, icb],
        ),
    )

    # Type conversion
    output = te.compute(
        oshape, lambda *index: conv(*index).astype(out_dtype), tag="conv2d_NCHWc_KCRSk"
    )

    num_flop = (
        batch
        * oc_chunk
        * oc_block
        * out_height
        * out_width
        * ic_chunk
        * ic_block
        * kernel_h
        * kernel_w
        * 2
    )
    cfg.add_flop(num_flop)

    return output


def schedule_conv2d_NCHWc_KCRSk(cfg, s, output):
    """Schedule conv2d NCHWc template"""

    conv = output.op.input_tensors[0]
    packed_data, packed_kernel = conv.op.input_tensors

    if isinstance(packed_data.op, tvm.te.ComputeOp) and "pad" in packed_data.op.tag:
        pad_data = packed_data
        packed_data = pad_data.op.input_tensors[0]
    else:
        pad_data = packed_data

    # if autotvm.GLOBAL_SCOPE.in_tuning:
    #     # skip this part during tuning to make records accurate
    #     # this part will be pre-computed during NNVM's pre-compute optimization pass
    #     s[packed_data].pragma(s[packed_data].op.axis[0], "debug_skip_region")
    #     s[packed_kernel].pragma(s[packed_kernel].op.axis[0], "debug_skip_region")
    # else:
    #     if isinstance(packed_kernel.op, tvm.te.ComputeOp) and packed_kernel.name == "packed_kernel":
    #         # data and kernel are not pre-computed, schedule layout transform here
    #         schedule_injective_from_existing(s, packed_data)
    #         schedule_injective_from_existing(s, packed_kernel)

    if pad_data != packed_data:
        s[pad_data].compute_inline()

    # create cache stage
    AA = s.cache_read(pad_data, "shared", [conv])
    WW = s.cache_read(packed_kernel, "shared", [conv])

    s[conv].set_scope("local")

    # handle bias
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0].output(0)

    oc_chunk = nn.get_const_int(output.shape[1])
    # tile and bind spatial axes
    n, f, y, x, c = s[output].op.axis
    cfg.define_split("tile_n", n, num_outputs=4)
    cfg.define_split("tile_f", cfg.axis(oc_chunk), num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)

    # this is the scope to attach global config inside this kernel
    kernel_scope, n = s[output].split(n, nparts=1)

    s[output].bind(n, te.thread_axis("blockIdx.z"))
    bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].reorder(bn, bf, by, bx, vn, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)
    s[output].bind(bn, te.thread_axis("blockIdx.z"))
    #s[output].bind(s[output].fuse(bg, bf), te.thread_axis("blockIdx.y"))
    s[output].bind(s[output].fuse(by, bx), te.thread_axis("blockIdx.x"))
    s[output].bind(vn, te.thread_axis("vthread"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    cfg.define_knob("fuse_yx", [0, 1])  # fuse ty,tx or tn,tf
    if cfg["fuse_yx"].val:
        s[output].bind(tn, te.thread_axis("threadIdx.z"))
        s[output].bind(tf, te.thread_axis("threadIdx.y"))
        tyx = s[output].fuse(ty, tx)
        s[output].bind(tyx, te.thread_axis("threadIdx.x"))
        s[conv].compute_at(s[output], tyx)

        # number of threads
        n_tz = cfg["tile_n"].size[2]
        n_ty = cfg["tile_f"].size[2]
        n_tx = cfg["tile_y"].size[2] * cfg["tile_x"].size[2]
    else:
        s[output].bind(tn, te.thread_axis("threadIdx.z"))
        s[output].bind(s[output].fuse(tn, tf), te.thread_axis("threadIdx.z"))
        s[output].bind(ty, te.thread_axis("threadIdx.y"))
        s[output].bind(tx, te.thread_axis("threadIdx.x"))
        s[conv].compute_at(s[output], tx)

        # number of threads
        n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2]
        n_ty = cfg["tile_y"].size[2]
        n_tx = cfg["tile_x"].size[2]

    # tile and bind reduction axes
    n, f, y, x, c = s[conv].op.axis
    rc, ry, rx, rc_block = s[conv].op.reduce_axis
    cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=2)
    cfg.define_split("tile_ry", cfg.axis(ry), num_outputs=2)
    cfg.define_split("tile_rx", cfg.axis(rx), num_outputs=2)
    rco, rci = cfg["tile_rc"].apply(s, conv, rc)
    ryo, ryi = cfg["tile_ry"].apply(s, conv, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, conv, rx)

    s[conv].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x, c, rc_block)
    #_, rc_block = s[conv].split(rc_block, factor=4)
    #s[conv].tensorize(rc_block, _dp4a)

    s[AA].compute_at(s[conv], rxo)
    s[WW].compute_at(s[conv], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        fcd = s[load].op.axis[-1]
        #fcd_outer, fcd = s[load].split(fcd, factor=4)
        s[load].vectorize(fcd)
        #fused = s[load].op.axis[:-1] + [fcd_outer]
        fused = s[load].op.axis[:-1]
        fused = s[load].fuse(*fused)

        fused, tx = s[load].split(fused, factor=n_tx)
        fused, ty = s[load].split(fused, factor=n_ty)
        fused, tz = s[load].split(fused, factor=n_tz)
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # double buffer
    # cfg.define_knob("AA_double_buffer", [0, 1])
    # cfg.define_knob("WW_double_buffer", [0, 1])
    # if cfg["AA_double_buffer"].val:
    #     s[AA].double_buffer()
    # if cfg["WW_double_buffer"].val:
    #     s[WW].double_buffer()

    # unroll
    # cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    # s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    # s[output].pragma(kernel_scope, "unroll_explicit", False)

    return s


@autotvm.template("matmul_vector_accumulator_tune")
def matmul_vector_acc_template(shapeA, shapeB):
    placeholders = compute_matmul_vector_accumulator(shapeA, shapeB)
    s = schedule_matmul_vector_accumulator_autotvm(*placeholders)
    return s, placeholders

@autotvm.template("conv2d_1x1_NCHWc_RSCKk_tune")
def conv2d_1x1_NCHWc_RSCKk_template(input_shape, filter_shape):
    placeholders = compute_conv2d_1x1_NCHWc_RSCKk(input_shape, filter_shape)
    s = schedule_conv2d_1x1_NCHWc_RSCKk(*placeholders)
    return s, placeholders

@autotvm.template("conv2d_1x1_WCHNc_CRSKk_tune")
def conv2d_1x1_WCHNc_CRSKk_template(input_shape, filter_shape):
    placeholders = compute_conv2d_1x1_WCHNc_CRSKk(input_shape, filter_shape)
    s = schedule_conv2d_1x1_WCHNc_CRSKk(*placeholders)
    return s, (placeholders[0], placeholders[1], placeholders[-1])

@autotvm.template("conv2d_mali_NCHW_KCRS_tune")
def conv2d_mali_NCHW_KCRS_template(input_shape, filter_shape):
    cfg = autotvm.get_config()
    data = te.placeholder(input_shape, name="data", dtype="float32")
    filt = te.placeholder(filter_shape, name="filter", dtype="float32")
    data_vec, filter_vec, output, conv = compute_conv2d_mali_NCHW_KCRS(cfg, data, filt, [1,1], [0,0], [0,0], "float32", num_tile=3)
    s = te.create_schedule([x.op for x in [data_vec, filter_vec, output, conv]])
    s = schedule_conv2d_mali_NCHW_KCRS(cfg, s, output, conv, data_vec, filter_vec)
    return s, (data, filt, output)

@autotvm.template("conv2d_cuda_NCHW_KCRS_tune")
def conv2d_cuda_NCHW_KCRS_template(input_shape, filter_shape):
    data = te.placeholder(input_shape, name="data", dtype="float32")
    filt = te.placeholder(filter_shape, name="filter", dtype="float32")
    conv = compute_conv2d_cuda_NCHW_KCRS(data, filt, [1,1], [0,0], [0,0], "float32")
    cfg = autotvm.get_config()
    s = te.create_schedule([x.op for x in [conv]])
    schedule_conv2d_cuda_NCHW_KCRS(cfg, s, conv)
    return s, (data, filt, conv)

@autotvm.template("conv2d_cuda_NCHWc_KCRSk_tune")
def conv2d_cuda_NCHWc_KCRSk_template(input_shape, filter_shape):
    cfg = autotvm.get_config()
    data = te.placeholder(input_shape, name="data", dtype="float32")
    filt = te.placeholder(filter_shape, name="filter", dtype="float32")
    output = compute_conv2d_NCHWc_KCRSk(cfg, data, filt, [1,1], [0,0], [0,0], "float32")
    s = te.create_schedule([x.op for x in [output]])
    s = schedule_conv2d_NCHWc_KCRSk(cfg, s, output)
    return s, (data, filt, output)

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
    elif args.test == "matmul_inner":
        shape = (32, 64, 4)
        placeholders = compute_matmul_inner(shape)
        s = schedule_matmul_inner(*placeholders)
    elif args.test == "matmul_vector_accumulator":
        shapeA, shapeB = (32, 64, 4), (128, 16, 4)
        placeholders = compute_matmul_vector_accumulator(shapeA, shapeB)
        s = schedule_matmul_vector_accumulator(*placeholders)
    elif args.test == "matmul_vector_accumulator_with_local":
        shapeA, shapeB = (32, 64, 4), (128, 16, 4)
        placeholders = compute_matmul_vector_accumulator(shapeA, shapeB)
        s = schedule_matmul_vector_accumulator(*placeholders, local=True)
    elif args.test == "plus_one_rank5":
        shape =(32, 2, 4, 4, 4)
        placeholders = compute5d(shape)
        s = schedule5d(*placeholders)
    elif "tune" in args.test:
        options = {
            "log_filename": "test_tune.log2", #args.test + ".autotvm.log",
            "early_stopping": None,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=1000),
                runner=autotvm.RPCRunner(
                    args.rpc_key,
                    host=args.rpc_tracker_host,
                    port=args.rpc_tracker_port,
                    number=5,
                    timeout=1000,
                ),
            )
        }
        def tune_and_bench(func, template, *args, **kwargs):
            task = autotvm.task.create(template, args=args, target=target, target_host=target_host)
            print(task.config_space)
            #tune_tasks([task], **options)
            with autotvm.apply_history_best(options["log_filename"]):
                with tvm.target.Target(target):
                    return func(*args)
        if args.test == "matmul_vector_accumulator_tune":
            shapeA, shapeB = (32, 64, 4), (128, 16, 4)
            s, placeholders = tune_and_bench(matmul_vector_acc_template, args.test, shapeA, shapeB)
        elif args.test == "conv2d_1x1_NCHWc_RSCKk_tune":
            # mobilenetv1 1x1 conv2d
            input_shape, filter_shape = (1, 128//4, 56, 56, 4), (1, 1, 128, 128//4, 4)
            s, placeholders = tune_and_bench(conv2d_1x1_NCHWc_RSCKk_template, args.test, input_shape, filter_shape)
        elif args.test == "conv2d_1x1_WCHNc_CRSKk_tune":
            input_shape, filter_shape = (56, 128//4, 56, 1, 4), (128, 1, 1, 128//4, 4)
            s, placeholders = tune_and_bench(conv2d_1x1_WCHNc_CRSKk_template, args.test, input_shape, filter_shape)
        elif args.test == "conv2d_mali_NCHW_KCRS_tune":
            # NCHW, KCRS
            input_shape, filter_shape = (1, 128, 56, 56), (128, 128, 1, 1)
            s, placeholders = tune_and_bench(conv2d_mali_NCHW_KCRS_template, args.test, input_shape, filter_shape)
        elif args.test == "conv2d_cuda_NCHW_KCRS_tune":
            # NCHW, KCRS
            input_shape, filter_shape = (1, 128, 56, 56), (128, 128, 1, 1)
            s, placeholders = tune_and_bench(conv2d_cuda_NCHW_KCRS_template, args.test, input_shape, filter_shape)
        elif args.test == "conv2d_cuda_NCHWc_KCRSk_tune":
            # NCHWc, KCRSk
            input_shape, filter_shape = (1, 32, 56, 56, 4), (32, 128, 1, 1, 4)
            s, placeholders = tune_and_bench(conv2d_cuda_NCHWc_KCRSk_template, args.test, input_shape, filter_shape)
        else:
            raise RuntimeError("No test found with name: " + args.test)
    else:
        raise RuntimeError("No test found with name: " + args.test)

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
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    print("time:", "%f ms" % (evaluator(*args_tvm).mean * 1e3))
    if "plus_one" in args.test:
        np_result = args_np[0] + 1.0;
    elif "matmul" in args.test:
        if 'inner' in args.test:
            np_result = np.matmul(args_np[0].reshape(32, 256), args_np[1].reshape(32, 256).transpose(1, 0))
        elif 'accum' in args.test:
            np_result = np.matmul(args_np[0].transpose((1, 0, 2)).reshape(64, 128), args_np[1].reshape(128, 64))
        else:
            np_result = np.matmul(args_np[0].transpose((0, 2, 1)).reshape(128, 64), args_np[1].transpose(1, 0, 2).reshape(64,128))
    elif args.test == "conv2d_1x1_NCHWc_RSCKk_tune":
        vec_length = args_np[1].shape[-1]
        # nchwc -> nchw
        args_np[0] = args_np[0].transpose((0, 1, 4, 2, 3)).reshape(args_np[0].shape[0], args_np[0].shape[1]*args_np[0].shape[-1], args_np[0].shape[2], args_np[0].shape[3])
        # rsckk -> rsck -> kcrs
        args_np[1] = args_np[1].reshape(args_np[1].shape[0], args_np[1].shape[1], args_np[1].shape[2], args_np[1].shape[3]*args_np[1].shape[4]).transpose((3, 2, 0, 1))
        np_result = testing.conv2d_nchw_python(args_np[0], args_np[1], 1, 0)
        # nkhw -> nkhwk
        np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(0, 1, 3, 4, 2)
    elif args.test == "conv2d_1x1_WCHNc_CRSKk_tune":
        vec_length = args_np[1].shape[-1]
        # wchnc -> nchw
        args_np[0] = args_np[0].transpose((3, 1, 4, 2, 0)).reshape(args_np[0].shape[3], args_np[0].shape[1]*args_np[0].shape[-1], args_np[0].shape[2], args_np[0].shape[0])
        # crskk -> crsk -> kcrs
        args_np[1] = args_np[1].reshape(args_np[1].shape[0], args_np[1].shape[1], args_np[1].shape[2], args_np[1].shape[3]*args_np[1].shape[4]).transpose((3, 0, 1, 2))
        np_result = testing.conv2d_nchw_python(args_np[0], args_np[1], 1, 0)
        # nkhw -> nkkhw -> wkhnk
        np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(4, 1, 3, 0, 2)
    elif "NCHW_KCRS" in args.test:
        np_result = testing.conv2d_nchw_python(args_np[0], args_np[1], 1, 0)
    elif args.test == "conv2d_cuda_NCHWc_KCRSk_tune":
        vec_length = args_np[1].shape[-1]
        # nchwc -> nchw
        args_np[0] = args_np[0].transpose((0, 1, 4, 2, 3)).reshape(args_np[0].shape[0], args_np[0].shape[1]*args_np[0].shape[-1], args_np[0].shape[2], args_np[0].shape[3])
        # kcrsk -> kcrs
        args_np[1] = args_np[1].transpose((0, 4, 1, 2, 3)).reshape(args_np[1].shape[0] * args_np[1].shape[4], args_np[1].shape[1], args_np[1].shape[2], args_np[1].shape[3])
        np_result = testing.conv2d_nchw_python(args_np[0], args_np[1], 1, 0)
        # nkhw -> nkhwk
        np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(0, 1, 3, 4, 2)
    np.testing.assert_allclose(args_tvm[-1].asnumpy(), np_result, rtol=1e-3, atol=1e-3)
    print("validation done")


if __name__ == "__main__":
    test_texture()
