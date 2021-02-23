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
from tvm import autotvm
from tvm.contrib import utils, ndk
from tvm.topi import testing


class ModelImporter(object):
    def available_models(self):
        import inspect
        models = []
        for method in inspect.getmembers(type(self)):
            if "import_" in method[0]:
                models.append(method[0].split("import_")[1])
        return models

    def __call__(self, model, *args, **kwargs):
        import inspect

        for method in inspect.getmembers(type(self)):
            if "import_" + model == method[0]:
                return method[1](self, *args, **kwargs)
        raise ValueError("import_" + model + " not found.")

    def import_resnet50(self, target="llvm", dtype="float32"):
        model, input_shape = gluon_model("resnet50_v1", batch_size=1)
        mod, params = relay.frontend.from_mxnet(model, {"data": input_shape})
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        return (mod, params, input_shape, dtype, target)

    def import_mobilenetv1(self, target="llvm", dtype="float32"):
        model, input_shape = gluon_model("mobilenet1.0", batch_size=1)
        mod, params = relay.frontend.from_mxnet(model, {"data": input_shape})
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        return (mod, params, input_shape, dtype, target)

    def import_vgg16(self, target="llvm", dtype="float32"):
        model, input_shape = gluon_model("vgg16", batch_size=1)
        mod, params = relay.frontend.from_mxnet(model, {"data": input_shape})
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        return (mod, params, input_shape, dtype, target)

    def import_mobilenetv3_ssdlite(self, target="llvm", dtype="float32"):
        import onnx

        graph_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            #+ "/../models/ssd-mobilenetV3-pytorch/mb3-ssd.onnx"
            + "/../models/mobilenetv3_ssdlite_tf1.15export/mobilenetv3_ssdlite.v11.onnx"
        )
        model = onnx.load_model(graph_file)
        input_shape = (1, 3, 300, 300)
        input_names, input_shape = get_input_data_shape_dict(model, input_shape)
        mod, params = relay.frontend.from_onnx(model, input_shape, opset=10, freeze_params=True)
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        from tvm.relay import transform
        mod = transform.DynamicToStatic()(mod)
        return (mod, params, input_shape, dtype, target)

    def import_deeplabv3(self, target="llvm", dtype="float32"):
        import onnx

        graph_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/../models/deeplabv3_mnv2_pascal_train_aug/deeplabv3_mnv2.onnx"
        )
        model = onnx.load_model(graph_file)
        input_shape = {"ImageTensor:0": (1, 224, 224, 3)}
        input_names, shape_dict = get_input_data_shape_dict(
            model, input_shape["ImageTensor:0"]
        )
        mod, params = relay.frontend.from_onnx(model, shape_dict, opset=11, freeze_params=True)
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        from tvm.relay import transform
        mod = transform.DynamicToStatic()(mod)
        return (mod, params, input_shape, dtype, target)

    def import_inceptionv3(self, target="llvm", dtype="float32"):
        import onnx
        model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__)) + "/../models/inceptionv3.onnx")
        if os.path.exists(model_file) == False:
            import keras2onnx
            import tensorflow as tf
            model = tf.keras.applications.InceptionV3(
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                classifier_activation="softmax",
            )
            onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=11)
            onnx.save(onnx_model, model_file)
        model = onnx.load(model_file)
        shape_dict = {'input_1': [1, 299, 299, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target)


    def import_conv2d(self, target="llvm", dtype="float32"):
        input_shape = (1, 32, 112, 112, 4)
        filter_shape = (32, 128, 3, 3, 4)
        A = relay.var("data", shape=input_shape, dtype="float32")
        B = relay.var("weight", shape=filter_shape, dtype="float32")
        C = relay.nn.conv2d(A, B, data_layout="NCHW4c", kernel_layout="OIHW4o")
        func = relay.Function([A, B], C)
        mod, params = relay.testing.init.create_workload(func)
        def validator(inputs):
            vec_length = input_shape[-1]
            # nchwc -> nchw
            data = inputs[0].transpose((0, 1, 4, 2, 3)).reshape(inputs[0].shape[0], inputs[0].shape[1]*inputs[0].shape[-1], inputs[0].shape[2], inputs[0].shape[3])
            # kcrsk -> kcrs
            w_np = params["weight"].asnumpy()
            kernel = w_np.transpose((0, 4, 1, 2, 3)).reshape(w_np.shape[0] * w_np.shape[4], w_np.shape[1], w_np.shape[2], w_np.shape[3])
            np_result = testing.conv2d_nchw_python(data, kernel, 1, 0)
            # nkhw -> nkhwk
            np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(0, 1, 3, 4, 2)
            return [np_result,]
        return (mod, params, {"data": input_shape}, dtype, target, validator)


    def import_conv2d_conv2d(self, target="llvm", dtype="float32"):
        # input_shape = (1, 128, 112, 112)
        # filter_shape = (128, 128, 3, 3)
        input_shape = (1, 32, 112, 112, 4)
        filter_shape = (32, 128, 3, 3, 4)
        A = relay.var("data", shape=input_shape, dtype="float32")
        B1 = relay.var("weight1", shape=filter_shape, dtype="float32")
        B2 = relay.var("weight2", shape=filter_shape, dtype="float32")
        C = relay.nn.conv2d(A, B1, data_layout="NCHW4c", kernel_layout="OIHW4o")
        D = relay.nn.conv2d(C, B2, data_layout="NCHW4c", kernel_layout="OIHW4o")
        mod = relay.Function([A, B1, B2], D)

        params = {
            "weight1": tvm.nd.array(np.random.uniform(-1, 1, filter_shape).astype("float32")),
            "weight2": tvm.nd.array(np.random.uniform(-1, 1, filter_shape).astype("float32")),
        }
        def validator(inputs):
            vec_length = input_shape[-1]
            # nchwc -> nchw
            data = inputs[0].transpose((0, 1, 4, 2, 3)).reshape(inputs[0].shape[0], inputs[0].shape[1]*inputs[0].shape[-1], inputs[0].shape[2], inputs[0].shape[3])
            # kcrsk -> kcrs
            w1_np = params["weight1"].asnumpy()
            kernel1 = w1_np.transpose((0, 4, 1, 2, 3)).reshape(w1_np.shape[0] * w1_np.shape[4], w1_np.shape[1], w1_np.shape[2], w1_np.shape[3])
            w2_np = params["weight2"].asnumpy()
            kernel2 = w2_np.transpose((0, 4, 1, 2, 3)).reshape(w2_np.shape[0] * w2_np.shape[4], w2_np.shape[1], w2_np.shape[2], w2_np.shape[3])
            conv2d = testing.conv2d_nchw_python(data, kernel1, 1, 0)
            np_result = testing.conv2d_nchw_python(conv2d, kernel2, 1, 0)
            # nkhw -> nkhwk
            np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(0, 1, 3, 4, 2)
            return [np_result,]
        return (mod, params, {"data": input_shape}, dtype, target, validator)

def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Tune and/or evaluate a curated set of models"
    )
    models = ModelImporter().available_models()

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
        "--type",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Specify whether the model should be run with single or half precision floating point values",
    )
    parser.add_argument(
        "-l", "--log", type=str, default=None, help="AutoTVM tuning logfile name"
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
        default="opencl --device=mali",
        help="Compilation target",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Whether or not to run autotuning"
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Use graph runtime debugger to output per layer perf. data and other statistics",
    )

    args = parser.parse_args()
    if args.log == None:
        args.log = "logs/" + args.model + "." + args.type + ".autotvm.log"
    if args.rpc_tracker_port != None:
        args.rpc_tracker_port = int(args.rpc_tracker_port)
    args.tuning_options = {
        "log_filename": args.log,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15),
            runner=autotvm.RPCRunner(
                args.rpc_key,
                host=args.rpc_tracker_host,
                port=args.rpc_tracker_port,
                number=3,
                timeout=15,
                #min_repeat_ms=150,
                #cooldown_interval=150
            ),
        ),
    }
    return args


args = get_args()


def main():
    executor = Executor(use_tracker="android")
    executor.schedule(args.model, target=args.target, dtype=args.type)
    if args.tune:
        executor.tune_pending_benchmarks()
    else:
        executor.tune_pending_benchmarks(apply_previous_tune=True)
    executor.run_pending_benchmarks()


def downcast_fp16(func, module):
    from tvm.relay.expr_functor import ExprMutator
    from tvm.relay.expr import Call, Var, Constant, TupleGetItem
    from tvm.relay import transform as _transform
    from tvm.relay import cast
    from tvm.ir import IRModule
    from tvm.relay import function as _function

    """Downcast to fp16 mutator
    Parameters
    ---------
    graph: Function
        The original graph.

    Retruns
    -------
    The graph after dowmcasting to half-precision floating-point.
    """
    filter_list = ["vision.get_valid_counts", "vision.non_max_suppression"]

    class DowncastMutator(ExprMutator):
        """Downcast to fp16 mutator"""

        def visit_call(self, call):
            dtype = "float32" if call.op.name in filter_list else "float16"
            new_fn = self.visit(call.op)
            # Collect the original dtypes
            type_list = []
            if call.op.name in filter_list:
                # For NMS
                for arg in call.args:
                    if isinstance(arg, TupleGetItem) and isinstance(
                        arg.tuple_value, Call
                    ):
                        tuple_types = arg.tuple_value.checked_type.fields
                        type_list.append(tuple_types[arg.index].dtype)
                if call.op.name == "vision.get_valid_counts":
                    tuple_types = call.checked_type.fields
                    for cur_type in tuple_types:
                        type_list.append(cur_type.dtype)

            args = [self.visit(arg) for arg in call.args]
            new_args = list()
            arg_idx = 0
            for arg in args:
                if isinstance(arg, (Var, Constant)):
                    new_args.append(cast(arg, dtype=dtype))
                else:
                    if call.op.name in filter_list:
                        if (
                            isinstance(arg, TupleGetItem)
                            and type_list[arg_idx] == "int32"
                        ):
                            new_args.append(arg)
                        else:
                            new_args.append(cast(arg, dtype=dtype))
                    else:
                        new_args.append(arg)
                arg_idx += 1
            if (
                call.op.name in filter_list
                and call.op.name != "vision.get_valid_counts"
            ):
                return cast(Call(new_fn, new_args, call.attrs), dtype="float16")
            return Call(new_fn, new_args, call.attrs)

    class UpcastMutator(ExprMutator):
        """upcast output back to fp32 mutator"""

        def visit_call(self, call):
            return cast(call, dtype="float32")

    def infer_type(node, mod=None):
        """A method to infer the type of an intermediate node in the relay graph."""
        if isinstance(mod, IRModule):
            mod["main"] = _function.Function(tvm.relay.analysis.free_vars(node), node)
            mod = _transform.InferType()(mod)
            entry = mod["main"]
            ret = entry.body
        else:
            new_mod = IRModule.from_expr(node)
            if mod is not None:
                new_mod.update(mod)
                new_mod = _transform.InferType()(new_mod)
                entry = new_mod["main"]
                ret = entry if isinstance(node, _function.Function) else entry.body

        return ret

    func = infer_type(func, module)
    downcast_pass = DowncastMutator()
    func = downcast_pass.visit(func)
    upcast_pass = UpcastMutator()
    func = upcast_pass.visit(func)
    func = infer_type(func, module)
    new_mod = IRModule.from_expr(func)
    # new_mod.update(module)
    return new_mod


def get_input_data_shape_dict(graph_def, input_shape):
    if isinstance(input_shape, list):
        input_names = {}
        shape_dict = {}
        for i in range(len(input_shape)):
            input_names[i] = graph_def.graph.input[i].name
            shape_dict[input_names[i]] = input_shape[i]
    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_shape}

    return input_names, shape_dict


def gluon_model(name, batch_size=None):
    import mxnet.gluon as gluon

    model = gluon.model_zoo.vision.get_model(name, pretrained=True)
    if "resnet50_v1" or "mobilenet1.0" in name:
        data_shape = (batch_size, 3, 224, 224)
    elif "inception" in name:
        data_shape = (batch_size, 3, 299, 299)
    else:
        raise ValueError("Input shape unknown for gluon model: " + name)

    return model, data_shape


class Executor(object):
    def __init__(self, use_tracker=False):
        self.benchmarks = []
        self.tuning_jobs = []
        self.tracker = None
        self.remote = None
        self.host_target = "llvm"
        self.use_tracker = use_tracker
        if use_tracker == "android":
            self.host_target = "llvm -mtriple=arm64-linux-android"
        elif use_tracker != False:

            class BackendNotImplementedForRPCBenchmarking(Exception):
                pass

            raise BackendNotImplementedForRPCBenchmarking

    def schedule(self, model, *args, **kwargs):
        importer = ModelImporter()
        self._schedule_jobs(*importer(model, *args, **kwargs))

    def run_pending_benchmarks(self):
        for bench in self.benchmarks:
            bench()

    def tune_pending_benchmarks(
        self, apply_previous_tune=False, opt=args.tuning_options
    ):
        for tune in self.tuning_jobs:
            tune(apply_previous_tune, options=args.tuning_options)

    def _connect_tracker(self):
        from tvm import rpc

        print(
            "Tracker attempting connection on {}:{}".format(
                args.rpc_tracker_host, args.rpc_tracker_port
            )
        )
        self.tracker = rpc.connect_tracker(args.rpc_tracker_host, args.rpc_tracker_port)
        self.remote = self.tracker.request(
            args.rpc_key, priority=0, session_timeout=600
        )
        print("Tracker connected to remote RPC server")

    def _disconnect_tracker(self):
        self.remote = None
        self.tracker = None

    def _benchmark(
        self,
        tvm_mod,
        params,
        input_shape,
        target="llvm",
        target_host="llvm",
        dtype="float32",
        validator=None
    ):
        if args.debug:
            from tvm.contrib.debugger import debug_runtime as graph_runtime
        else:
            from tvm.contrib import graph_runtime

        if self.use_tracker and self.remote == None:
            self._connect_tracker()

        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(
                tvm_mod, target_host=target_host, target=target, params=params
            )

        if self.remote:
            print("Using Android OpenCL runtime over RPC")
            temp = utils.tempdir()
            dso_binary = "dev_lib_cl.so"
            dso_binary_path = temp.relpath(dso_binary)
            if "opencl" in target:
                ctx = self.remote.cl(0)
            else:
                ctx = self.remote.cpu(0)
            lib.export_library(dso_binary_path, ndk.create_shared)
            self.remote.upload(dso_binary_path)
            print("Uploading binary...")
            rlib = self.remote.load_module(dso_binary)
            m = graph_runtime.create(graph, rlib, ctx)
        else:
            print("Using local runtime")
            ctx = tvm.context(target, 0)
            m = graph_runtime.create(graph, lib, ctx)

        m.set_input(**params)
        inputs = []
        if isinstance(input_shape, dict):
            for key in input_shape:
                inputs.append(np.random.normal(size=input_shape[key]).astype(dtype))
                m.set_input(key, inputs[-1])
        else:
            inputs.append(np.random.normal(size=input_shape).astype(dtype))
            m.set_input("data", inputs[-1])

        print("Evaluating...")
        time_f = m.module.time_evaluator("run", ctx, number=10)
        cost = time_f().mean
        print("%g secs/iteration\n" % cost)
        if validator:
            ref_outputs = validator(inputs)
            for i, ref_output in enumerate(ref_outputs):
                output = m.get_output(i)
                np.testing.assert_allclose(output.asnumpy(), ref_output, rtol=1e-3, atol=1e-3)
            print("Validation done")


    def _schedule_jobs(self, mod, params, input_shape, dtype, target, validator=None):
        def bench():
            self._benchmark(
                mod,
                params,
                input_shape,
                target=target,
                target_host=self.host_target,
                dtype=dtype,
                validator=validator
            )

        benchmark_index = len(self.benchmarks)
        self.benchmarks.append(bench)

        def tune(apply_previous_tune=False, options=args.tuning_options):
            print("Extracting tasks")
            tasks = autotvm.task.extract_from_program(
                mod, target=target, target_host=self.host_target, params=params
            )
            if apply_previous_tune == False:
                print("Tuning kernels")
                Executor.tune_tasks(tasks, **options)

            def tuned_benchmark():
                print("Apply best performing tuning profiles:")
                with autotvm.apply_history_best(options["log_filename"]):
                    self._benchmark(
                        mod,
                        params,
                        input_shape,
                        target=target,
                        target_host=self.host_target,
                        validator=validator
                    )

            self.benchmarks.pop(benchmark_index)
            self.benchmarks.append(tuned_benchmark)

        self.tuning_jobs.append(tune)

    @staticmethod
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

if __name__ == "__main__":
    main()
