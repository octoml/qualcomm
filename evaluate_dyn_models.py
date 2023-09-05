import os
import numpy as np

import tvm
import tvm.testing
from tvm import autotvm
from tvm import relay
from tvm.contrib import utils, ndk
from tvm.runtime.vm import VirtualMachine
from common import convert_to_dtype, advanced_time_evaluator
import argparse

def get_args():
    models = ['onnx_ssd_resnet34', 'onnx_yolo_v3', 'onnx_faster_rcnn']

    parser = argparse.ArgumentParser(
        description="Tune and/or evaluate a curated set of models"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        required=True,
        help="Model to tune and/or evaluate",
        choices=models,
    )
    parser.add_argument(
        "-t",
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "float16_acc32"],
        help="Specify whether the model should be run with single or half precision floating point values",
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
    parser.add_argument(
        "-T",
        "--target",
        type=str,
        default="opencl --device=adreno",
        help="Compilation target",
    )
    parser.add_argument(
        "-H",
        "--target_host",
        type=str,
        default="llvm -mtriple=arm64-linux-android",
        help="Compilation target",
    )
    parser.add_argument(
        "--repeat",
        help="Additional parameter for time evaluator.",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--trials",
        help="Number of trials for AutoTVM tuning.",
        default=333,
        type=int,
    )
    parser.add_argument(
        "-l", "--log", type=str, default=None, help="AutoTVM tuning logfile name"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Benchmark with tuning / without tuning.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debugging.",
    )
    parser.add_argument(
        "--VM",
        action="store_true",
        help="Use VM compiling and benchmarking",
    )

    args = parser.parse_args()
    return args

args = get_args()

def onnx_ssd_resnet34_layers():
    batch_norm = [
        ((1, 3, 1200, 1200), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), 1),
        ((1, 64, 300, 300), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 64, 300, 300), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), 0),
        ((1, 64, 300, 300), (128, 64, 3, 3), (1, 1, 1, 1), (2, 2), 1),
        ((1, 64, 300, 300), (128, 64, 1, 1), (0, 0, 0, 0), (2, 2), 0),
        ((1, 128, 150, 150), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 128, 150, 150), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), 0),
        ((1, 128, 150, 150), (256, 128, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 128, 150, 150), (256, 128, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 256, 150, 150), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 256, 150, 150), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 0),
    ]

    bias_add = [
        ((1, 256, 150, 150), (16, 256, 3, 3), (1, 1, 1, 1), (3, 3), 0),
        ((1, 256, 150, 150), (256, 256, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 256, 150, 150), (512, 256, 3, 3), (1, 1, 1, 1), (2, 2), 1),
        ((1, 512, 75, 75), (24, 512, 3, 3), (1, 1, 1, 1), (3, 3), 0),
        ((1, 512, 75, 75), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 256, 75, 75), (512, 256, 3, 3), (1, 1, 1, 1), (2, 2), 1),
        ((1, 512, 38, 38), (24, 512, 3, 3), (1, 1, 1, 1), (3, 3), 0),
        ((1, 512, 38, 38), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 128, 38, 38), (256, 128, 3, 3), (1, 1, 1, 1), (2, 2), 1),
        ((1, 256, 19, 19), (24, 256, 3, 3), (1, 1, 1, 1), (3, 3), 0),
        ((1, 256, 19, 19), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 128, 19, 19), (256, 128, 3, 3), (0, 0, 0, 0), (2, 2), 1),
        ((1, 256, 9, 9), (16, 256, 3, 3), (1, 1, 1, 1), (3, 3), 0),
        ((1, 256, 9, 9), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 128, 9, 9), (256, 128, 3, 3), (0, 0, 0, 0), (1, 1), 1),
        ((1, 256, 7, 7), (16, 256, 3, 3), (1, 1, 1, 1), (3, 3), 0),
        ((1, 256, 150, 150), (324, 256, 3, 3), (1, 1, 1, 1), (3, 3), 0),
        ((1, 512, 75, 75), (486, 512, 3, 3), (1, 1, 1, 1), (3, 3), 0),
        ((1, 512, 38, 38), (486, 512, 3, 3), (1, 1, 1, 1), (3, 3), 0),
        ((1, 256, 19, 19), (486, 256, 3, 3), (1, 1, 1, 1), (3, 3), 0),
        ((1, 256, 9, 9), (324, 256, 3, 3), (1, 1, 1, 1), (3, 3), 0),
        ((1, 256, 7, 7), (324, 256, 3, 3), (1, 1, 1, 1), (3, 3), 0),
    ]
    nms = [
        ((1, 15130, 81), (1, 80, 15130), 200, 0.5, 0.05)
    ]
    return batch_norm, bias_add, nms
    
def onnx_yolo_v3_layers():
    batch_norm = [
        ((1, 3, 416, 416), (32, 3, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 32, 416, 416), (64, 32, 3, 3), (1, 1, 0, 0), (2, 2), 1),
        ((1, 64, 208, 208), (32, 64, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 32, 208, 208), (64, 32, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 64, 208, 208), (128, 64, 3, 3), (1, 1, 0, 0), (2, 2), 1),
        ((1, 128, 104, 104), (64, 128, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 64, 104, 104), (128, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 128, 104, 104), (256, 128, 3, 3), (1, 1, 0, 0), (2, 2), 1),
        ((1, 256, 52, 52), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 128, 52, 52), (256, 128, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 256, 52, 52), (512, 256, 3, 3), (1, 1, 0, 0), (2, 2), 1),
        ((1, 512, 26, 26), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 256, 26, 26), (512, 256, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 512, 26, 26), (1024, 512, 3, 3), (1, 1, 0, 0), (2, 2), 1),
        ((1, 1024, 13, 13), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 512, 13, 13), (1024, 512, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 512, 13, 13), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 768, 26, 26), (256, 768, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 256, 26, 26), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 384, 52, 52), (128, 384, 1, 1), (0, 0, 0, 0), (1, 1), 1),
    ]

    bias_add = [
        ((1, 256, 52, 52), (255, 256, 3, 3), (0, 0, 0, 0), (1, 1), 0),
        ((1, 512, 26, 26), (255, 512, 3, 3), (0, 0, 0, 0), (1, 1), 0),
        ((1, 1024, 13, 13), (255, 1024, 1, 1), (0, 0, 0, 0), (1, 1), 0),
    ]

    nms = [
        ((1, 10647, 4), (1, 80, 10647), 20, 0.5, 0.6)
    ]
    return batch_norm, bias_add, nms
    
def onnx_faster_rcnn_layers():
    batch_norm = [
        ((1, 3, 1200, 1200), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), 1),
        
    ]
    bias_add = [
        ((1, 3, 800, 800), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), 1),
        ((1, 64, 200, 200), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 64, 200, 200), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 64, 200, 200), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 256, 200, 200), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), 1),
        ((1, 128, 100, 100), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 128, 100, 100), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 256, 200, 200), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), 0),
        ((1, 512, 100, 100), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 512, 100, 100), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), 1),
        ((1, 256, 50, 50), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 256, 50, 50), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 512, 100, 100), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), 0),
        ((1, 1024, 50, 50), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 1024, 50, 50), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), 1),
        ((1, 512, 25, 25), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 512, 25, 25), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 1024, 50, 50), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), 0),
        ((1, 2048, 25, 25), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((1, 2048, 25, 25), (256, 2048, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 1024, 50, 50), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 512, 100, 100), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 256, 200, 200), (256, 256, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 256, 100, 100), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 0),
        ((1, 256, 200, 200), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 0),
        ((1, 256, 25, 25), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 0),
        ((1, 256, 13, 13), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 0),
        ((1, 256, 25, 25), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 256, 50, 50), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 0),
        ((1, 256, 50, 50), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 256, 100, 100), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 256, 200, 200), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 1),
        ((1, 256, 13, 13), (12, 256, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 256, 25, 25), (12, 256, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 256, 50, 50), (12, 256, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 256, 100, 100), (12, 256, 1, 1), (0, 0, 0, 0), (1, 1), 0),
        ((1, 256, 200, 200), (12, 256, 1, 1), (0, 0, 0, 0), (1, 1), 0),
    ]
    nms = [
        ((1, 1000, 4), (1, 1, 1000), 2000, 0.7, 0),
        ((1, 507, 4), (1, 1, 507), 2000, 0.7, 0)
    ]
    return batch_norm, bias_add, nms
    
def generate_model_bn(dtype, input_shape, filter_shape, padding, strides, relu, leaky=False):
    dtype = "float32"
    shape_dict = {
        "input": input_shape,
        "weight": filter_shape,
        "bn_gamma0": (filter_shape[0],),
        "bn_beta0": (filter_shape[0],),
        "bn_mean0": (filter_shape[0],),
        "bn_var0": (filter_shape[0],),
    }
    input = tvm.relay.var("input", shape=input_shape, dtype=dtype)
    weight = tvm.relay.var("weight", shape=filter_shape, dtype=dtype)
    bn_gamma0 = tvm.relay.var("bn_gamma0", relay.TensorType((filter_shape[0],), dtype))
    bn_beta0 = tvm.relay.var("bn_beta0", relay.TensorType((filter_shape[0],), dtype))
    bn_mmean0 = tvm.relay.var("bn_mean0", relay.TensorType((filter_shape[0],), dtype))
    bn_mvar0 = tvm.relay.var("bn_var0", relay.TensorType((filter_shape[0],), dtype))

    channels = filter_shape[0]
    kernel_size = (filter_shape[2], filter_shape[3],)
    
    
    D = relay.nn.conv2d(input, weight, padding=padding, strides=strides, channels=channels, kernel_size=kernel_size)
    D = relay.op.nn.batch_norm(D, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0)
    D2 = D[0]
    if relu:
        D2 = relay.op.nn.relu(data=D2)
        if leaky:
            D2 = relay.op.nn.leaky_relu(data=D2)
    mod = relay.Function([input, weight, bn_gamma0,  bn_beta0, bn_mmean0, bn_mvar0], D2)
        
    params = {
        "weight": tvm.nd.array(np.random.uniform(-128, 127, filter_shape).astype(dtype)),
    }
    module = tvm.IRModule({})
    module["main"] = mod
    module = convert_to_dtype(module["main"], args.dtype)
    dtype = "float32" if args.dtype == "float32" else "float16"
    return module, params, shape_dict, dtype

def generate_model_bias_add(dtype, input_shape, filter_shape, padding, strides, relu):
    dtype = "float32"
    bias_shape = (filter_shape[0],)
    shape_dict = {
        "input": input_shape,
        "weight": filter_shape,
        "bias": bias_shape,
    }
    input = tvm.relay.var("input", shape=input_shape, dtype=dtype)
    weight = tvm.relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)
    channels = filter_shape[0]
    kernel_size = (filter_shape[2], filter_shape[3],)
    D = relay.nn.conv2d(input, weight, padding=padding, strides=strides, channels=channels, kernel_size=kernel_size)
    D = relay.op.nn.bias_add(D, bias)
    if relu:
        D = relay.op.nn.relu(D)
    mod = relay.Function([input, weight, bias], D)
    params = {
        "weight": tvm.nd.array(np.random.uniform(-128, 127, filter_shape).astype(dtype)),
        "bias": tvm.nd.array(np.random.uniform(-128, 127, bias_shape).astype(dtype)),
    }
    module = tvm.IRModule({})
    module["main"] = mod
    module = convert_to_dtype(module["main"], args.dtype)
    dtype = "float32" if args.dtype == "float32" else "float16"
    return module, params, shape_dict, dtype

def generate_model_nms(boxes_shape, scores_shape, max_output_boxes_per_class, iou_threshold, score_threshold):
    shape_dict = {
        "boxes": boxes_shape,
        "scores": scores_shape,
    }
    boxes = relay.var("boxes", relay.ty.TensorType(boxes_shape, "float32"))
    scores = relay.var("scores", relay.ty.TensorType(scores_shape, "float32"))

    out = relay.vision.all_class_non_max_suppression(
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    )

    mod = relay.Function([boxes, scores], out.astuple())
    params = {}
    module = tvm.IRModule({})
    module["main"] = mod
    return module, params, shape_dict

def build_model_ge(mod, params):
    lib_path = "lib.ge.so"
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(
            mod, target_host=args.target_host, target=args.target, params=params
        )
    if "android" in args.rpc_key:
        lib.export_library(lib_path, fcompile=ndk.create_shared)
    else:
        lib.export_library(lib_path)
    return lib, lib_path, graph, params

def build_model_vm(mod, params):
    lib_path = "lib.vm.so"
    if isinstance(mod, tvm.IRModule):
        vm_mod = mod
    else:
        vm_mod = tvm.IRModule()
        vm_mod["main"] = mod
    with tvm.transform.PassContext(opt_level=3):
        vmc = relay.vm.compile(vm_mod, target=args.target, target_host=args.target_host, params=params)
        if "android" in args.rpc_key:
            vmc.mod.export_library(lib_path, fcompile=ndk.create_shared)
        else:
            vmc.mod.export_library(lib_path)

    return vmc, lib_path

def build_model_with_stat(mod, params, stat_file):
    with autotvm.apply_history_best(stat_file):
        if args.VM:
            return build_model_vm(mod, params)
        else:
            return build_model_ge(mod, params)


def run_module(remote, lib_path, input_dict, lib, graph=None, params={}):
    if args.debug:
        from tvm.contrib.debugger import debug_runtime as graph_executor
    else:
        from tvm.contrib import graph_executor
    
    rlib = lib
    if remote:
        print("Using Android OpenCL runtime over RPC")
        if "opencl" in args.target:
            dev = remote.cl(0)
        else:
            dev = remote.cpu(0)
        remote.upload(lib_path)
        rlib = remote.load_module(lib_path)
    else:
        print("Using local runtime")
        dev = tvm.device(args.target, 0)
    
    number = 1
    repeat = args.repeat
    min_repeat_ms = 0
    time_to_work_ms = 1000
    cooldown_interval_ms=1000
    
    if args.VM:
        if args.debug:
            vm = tvm.runtime.profiler_vm.VirtualMachineProfiler(rlib, dev, "naive")
        else:
            vm = VirtualMachine(rlib, dev, "naive")
        data = {}
        for k, v in input_dict.items():
            data[k] = tvm.nd.array(v, dev)
        vm.set_input("main", **data)
        if args.debug:
            res = vm.profile(**data, func_name="main")
            print(res)
            benchmarkResult = None
        else:
            time_f = advanced_time_evaluator(vm, "invoke_stateful", dev, number, repeat, min_repeat_ms, time_to_work_ms, cooldown_interval_ms, mod_func_name="main")
            benchmarkResult = time_f("main")
    else:
        m = graph_executor.create(graph, rlib, dev)
        m.set_input(**params)
        if args.debug:
            m.run()
        time_f = advanced_time_evaluator(m, "run", dev, number, repeat, min_repeat_ms, time_to_work_ms, cooldown_interval_ms)
        benchmarkResult = time_f()
    
    if benchmarkResult:
        cost = benchmarkResult.mean
        cost_ms = cost * 1000
        print(f'{cost_ms}', flush=True)
    else:
        print("VM executor could not be additionally benchmarked with --debug flag. (ZeroDivisionError: float division by zero in 'advanced_time_evaluator'.)")
    

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=333,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=False,
):
    from tvm.autotvm.tuner import XGBTuner
    from tvm.autotvm.tuner import GATuner

    tmp_log_file = log_filename + ".tmp"

    for i, tsk in enumerate(reversed(tasks)):
        print("Task: ", tsk)
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
    # os.remove(tmp_log_file)

def tune(mod, params):
    tasks = autotvm.task.extract_from_program(
        mod, target=args.target, target_host=args.target_host, params=params
    )
    tuning_options = {
        "n_trial": args.trials,
        "log_filename": args.log,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15),
            runner=autotvm.RPCRunner(
                args.rpc_key,
                host=args.rpc_tracker_host,
                port=args.rpc_tracker_port,
                number=50,
                timeout=15,
            ),
        ),
    }
    print("Tuning kernels")
    tune_tasks(tasks, **tuning_options)
    #print("Apply best performing tuning profiles:")
    #with autotvm.apply_history_best(args.log):
    #    bench()

def connect_tracker():
    from tvm import rpc
    print(
        "Tracker attempting connection on {}:{}".format(
            args.rpc_tracker_host, args.rpc_tracker_port
        )
    )
    tracker = rpc.connect_tracker(args.rpc_tracker_host, args.rpc_tracker_port)
    remote = tracker.request(
        args.rpc_key, priority=0
    )
    print("Tracker connected to remote RPC server")
    return remote
    
def tune_model(batch_norm, bias_add, nms):
    for input_shape, filter_shape, workload_padding, strides, relu in batch_norm:
        mod, params, input_shape, _ = generate_model_bn(args.dtype, input_shape, filter_shape, workload_padding, strides, relu)
        tune(mod, params)
    
    for input_shape, filter_shape, workload_padding, strides, relu in bias_add:
        mod, params, input_shape, _ = generate_model_bias_add(args.dtype, input_shape, filter_shape, workload_padding, strides, relu)
        tune(mod, params)
    
    for boxes_shape, scores_shape, max_output_boxes_per_class, iou_threshold, score_threshold in nms:
        mod, params, input_shape = generate_model_nms(boxes_shape, scores_shape, max_output_boxes_per_class, iou_threshold, score_threshold)
        tune(mod, params)
    
def build_and_evaluate(batch_norm, bias_add, nms):
    for input_shape, filter_shape, workload_padding, strides, relu in batch_norm:
        mod, params, shape_dict, dtype = generate_model_bn(args.dtype, input_shape, filter_shape, workload_padding, strides, relu)
        input_dict = {}
        for k, v in shape_dict.items():
            img = np.random.rand(*v).astype(dtype)
            input_dict[k] = img
        remote = connect_tracker()
        if args.VM:
            vmc, lib_path = build_model_with_stat(mod, params, args.log)
            run_module(remote, lib_path, input_dict, vmc)
        else:
            lib, lib_path, graph, params = build_model_with_stat(mod, params, args.log)
            run_module(remote, lib_path, input_dict, lib, graph, params)
        del remote

    for input_shape, filter_shape, workload_padding, strides, relu in bias_add:
        mod, params, shape_dict, dtype = generate_model_bias_add(args.dtype, input_shape, filter_shape, workload_padding, strides, relu)
        input_dict = {}
        for k, v in shape_dict.items():
            img = np.random.rand(*v).astype(dtype)
            input_dict[k] = img
        remote = connect_tracker()
        if args.VM:
            vmc, lib_path = build_model_with_stat(mod, params, args.log)
            run_module(remote, lib_path, input_dict, vmc)
        else:
            lib, lib_path, graph, params = build_model_with_stat(mod, params, args.log)
            run_module(remote, lib_path, input_dict, lib, graph, params)
        del remote

    for boxes_shape, scores_shape, max_output_boxes_per_class, iou_threshold, score_threshold in nms:
        mod, params, shape_dict = generate_model_nms(boxes_shape, scores_shape, max_output_boxes_per_class, iou_threshold, score_threshold)
        input_dict = {}
        for k, v in shape_dict.items():
            img = np.random.rand(*v).astype("float32")
            input_dict[k] = img
        remote = connect_tracker()
        try:
            if args.VM:
                vmc, lib_path = build_model_with_stat(mod, params, args.log)
                run_module(remote, lib_path, input_dict, vmc)
            else:
                lib, lib_path, graph, params = build_model_with_stat(mod, params, args.log)
                run_module(remote, lib_path, input_dict, lib, graph, params)
        except RuntimeError as RE:
            print("Following error occured:", RE)
            continue
        del remote


def run_full():
    if args.rpc_tracker_port != None:
        args.rpc_tracker_port = int(args.rpc_tracker_port)
    if args.model == 'onnx_ssd_resnet34':
        batch_norm, bias_add, nms = onnx_ssd_resnet34_layers()
    elif args.model == 'onnx_yolo_v3':
        batch_norm, bias_add, nms = onnx_yolo_v3_layers()
    elif args.model == 'onnx_faster_rcnn':
        batch_norm, bias_add, nms = onnx_faster_rcnn_layers()
    if args.tune:
        tune_model(batch_norm, bias_add, nms)
    build_and_evaluate(batch_norm, bias_add, nms)
    

if __name__ == "__main__":
    run_full()
