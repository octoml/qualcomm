import os
import numpy as np

import tvm
import tvm.testing
from tvm import autotvm
from tvm import relay
from tvm.contrib import utils, ndk

class TestConv2dResNet50():
    def __init__(self):
        self.target_host = "llvm -mtriple=arm64-linux-android"
        self.target = "opencl --device=adreno"
        self.rpc_key = "android"
        self.rpc_tracker_host = "0.0.0.0"
        self.rpc_tracker_port = 9190

        self.dtype = "float32"
        self.stat_file = f"resnet_{self.dtype}.autotvm.log"

        self.layers_data_bn = [
            ((1, 3, 1200, 1200), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), 1),
            ((1, 64, 300, 300), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1),
            ((1, 64, 300, 300), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), 0),
            ((1, 64, 300, 300), (128, 64, 3, 3), (1, 1, 1, 1), (2, 2), 1),
            ((1, 64, 300, 300), (128, 64, 1, 1), (0, 0, 0, 0), (2, 2), 0),
            ((1, 128, 150, 150), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), 1),
            ((1, 128, 150, 150), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), 0),
            ((1, 128, 150, 150), (256, 128, 3, 3), (1, 1, 1, 1), (1, 1), 1),
            ((1, 128, 150, 150), (256, 128, 1, 1), (0, 0, 0, 0), (1, 1), 0),
            # ((1, 256, 150, 150), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 1),
            # ((1, 256, 150, 150), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), 0),
        ]

        self.layers_data_bias_add = [
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

        self.nms_data = [
            ((1, 15130, 81), (1, 80, 15130), 200, 0.5, 0.05)
        ]
    

    def generate_model_bn(self, dtype, input_shape, filter_shape, padding, strides, relu):
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
        mod = relay.Function([input, weight, bn_gamma0,  bn_beta0, bn_mmean0, bn_mvar0], D2)
            
        params = {
            "weight": tvm.nd.array(np.random.uniform(-128, 127, filter_shape).astype(dtype)),
        }
        return mod, params

    def generate_model_bias_add(self, dtype, input_shape, filter_shape, padding, strides, relu):
        bias_shape = (filter_shape[0],)
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
        return mod, params

    def generate_model_nms(self, boxes_shape, scores_shape, max_output_boxes_per_class, iou_threshold, score_threshold):
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
        return mod, params

    def tune_tasks(
        self,
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
        #if os.path.exists(tmp_log_file) and use_transfer_learning == False:
        #    os.remove(tmp_log_file)

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

    def tune_model(self, mod, params):
        tasks = autotvm.task.extract_from_program(
            mod, target=self.target, target_host=self.target_host, params=params
        )
        tuning_options = {
            "log_filename": self.stat_file,
            "early_stopping": None,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15),
                runner=autotvm.RPCRunner(
                    self.rpc_key,
                    host=self.rpc_tracker_host,
                    port=self.rpc_tracker_port,
                    number=50,
                    timeout=15,
                ),
            ),
        }
        print("Tuning kernels")
        self.tune_tasks(tasks, **tuning_options)
        #print("Apply best performing tuning profiles:")
        #with autotvm.apply_history_best(self.stat_file):
        #    bench()

    def connect_tracker(self):
        from tvm import rpc

        print(
            "Tracker attempting connection on {}:{}".format(
                self.rpc_tracker_host, self.rpc_tracker_port
            )
        )
        self.tracker = rpc.connect_tracker(self.rpc_tracker_host, self.rpc_tracker_port)
        self.remote = self.tracker.request(
            self.rpc_key, priority=0
        )
        print("Tracker connected to remote RPC server")

    def build_model(self, mod, params):
        print("mod:\n", mod)

        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(
                mod, target_host=self.target_host, target=self.target, params=params
            )
        return graph, lib, params

    def build_model_with_stat(self, mod, params, stat_file):
        with autotvm.apply_history_best(stat_file):
            return self.build_model(mod, params)

    def create_module(self, graph, lib, debug=False):
        if debug:
            from tvm.contrib.debugger import debug_runtime as graph_executor
        else:
            from tvm.contrib import graph_executor
        print("Using Android OpenCL runtime over RPC")
        temp = utils.tempdir()
        dso_binary = "dev_lib_cl.so"
        dso_binary_path = temp.relpath(dso_binary)
        if "opencl" in self.target:
            dev = self.remote.cl(0)
        else:
            dev = self.remote.cpu(0)
        lib.export_library(dso_binary_path, ndk.create_shared)
        self.remote.upload(dso_binary_path)
        print("Uploading binary...")
        rlib = self.remote.load_module(dso_binary)
        m = graph_executor.create(graph, rlib, dev)
        return rlib, m, dev

    def run_module(self, module, params, input_shape, dtype, dev):
        module.set_input(**params)
        inp = np.random.normal(size=input_shape).astype(dtype)
        module.set_input("input", inp)
        print("Evaluating...", flush=True)
        number = 1
        repeat = 50
        module.run(number=number, repeat=repeat)
    
    def run_module_nms(self, module, dtype):
        print("Evaluating...", flush=True)
        number = 1
        repeat = 50
        module.run(number=number, repeat=repeat)


    def kill(self):
        del self.remote
        del self.tracker

    def test_tune(self):
        # for input_shape, filter_shape, workload_padding, strides, relu in self.layers_data_bn:
        #     mod, params = self.generate_model_bn(self.dtype, input_shape, filter_shape, workload_padding, strides, relu)
        #     self.tune_model(mod, params)
        
        # for input_shape, filter_shape, workload_padding, strides, relu in self.layers_data_bias_add:
        #     mod, params = self.generate_model_bias_add(self.dtype, input_shape, filter_shape, workload_padding, strides, relu)
        #     self.tune_model(mod, params)
        
        for boxes_shape, scores_shape, max_output_boxes_per_class, iou_threshold, score_threshold in self.nms_data:
            mod, params = self.generate_model_nms(boxes_shape, scores_shape, max_output_boxes_per_class, iou_threshold, score_threshold)
            self.tune_model(mod, params)
        
    def test_run(self):
        for input_shape, filter_shape, workload_padding, strides, relu in self.layers_data_bn:
            mod, params = self.generate_model_bn(self.dtype, input_shape, filter_shape, workload_padding, strides, relu)
            self.connect_tracker()
            graph, lib, params = self.build_model_with_stat(mod, params, self.stat_file)
            rlib, module, dev = self.create_module(graph, lib, debug=True)
            self.run_module(module, params, input_shape, self.dtype, dev)
            del dev
            del module
            del rlib
            self.kill()
        
        for input_shape, filter_shape, workload_padding, strides, relu in self.layers_data_bias_add:
            mod, params = self.generate_model_bias_add(self.dtype, input_shape, filter_shape, workload_padding, strides, relu)
            self.connect_tracker()
            graph, lib, params = self.build_model_with_stat(mod, params, self.stat_file)
            rlib, module, dev = self.create_module(graph, lib, debug=True)
            self.run_module(module, params, input_shape, self.dtype, dev)
            del dev
            del module
            del rlib
            self.kill()
        
        for boxes_shape, scores_shape, max_output_boxes_per_class, iou_threshold, score_threshold in self.nms_data:
            mod, params = self.generate_model_nms(boxes_shape, scores_shape, max_output_boxes_per_class, iou_threshold, score_threshold)
            self.connect_tracker()
            graph, lib, params = self.build_model_with_stat(mod, params, self.stat_file)
            rlib, module, dev = self.create_module(graph, lib, debug=True)
            self.run_module_nms(module, self.dtype)
            del dev
            del module
            del rlib
            self.kill()
    
    def run_full(self):
        # self.test_tune()
        self.test_run()


if __name__ == "__main__":
    TestConv2dResNet50().run_full()
