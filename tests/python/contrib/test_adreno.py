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
import mxnet.gluon as gluon
import tvm
from tvm import te
from tvm import relay
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from compare_with_tf import *
import onnx
import onnxruntime

import tvm.relay.testing.tf as tf_importer
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner

#from tvm.contrib.debugger import debug_runtime
from tvm.contrib import graph_runtime
from tvm.contrib import util, ndk

device_key = "android"
tracker_host = os.environ["TVM_TRACKER_HOST"]
tracker_port = int(os.environ["TVM_TRACKER_PORT"])

tuning_options = {
    'log_filename': "autotvm_tuning.log",
    'early_stopping': None,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func=ndk.create_shared),
        runner=autotvm.RPCRunner(device_key, host=tracker_host, port=tracker_port, number=5, timeout=100),
    ),
}

def get_network(name, batch_size=None, is_gluon_model=True):
    if is_gluon_model:
        model = gluon.model_zoo.vision.get_model(name, pretrained=True)
    if "resnet50_v1" or "mobilenet1.0" in name:
        data_shape = (batch_size, 3, 224, 224)
    elif "inception" in name:
        data_shape = (batch_size, 3, 299, 299)
    else:
        raise ValueError("Unsupported network: " + name)

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
    def connect_tracker(self):
        from tvm import rpc
        print("Tracker attempting connection on {}:{}".format(tracker_host, tracker_port))
        self.tracker = rpc.connect_tracker(tracker_host, tracker_port)
        self.remote = self.tracker.request(device_key, priority=0,session_timeout=60)
        print("Tracker connected to remote RPC server")

    def disconnect_tracker(self):
        self.remote = None
        self.tracker = None

    def run_pending_benchmarks(self):
        for bench in self.benchmarks:
            bench()

    def tune_pending_benchmarks(self):
        for tune in self.tuning_jobs:
            tune()

    def benchmark(self,tvm_mod, params, input_shape, target='llvm', target_host="llvm"):
        if self.use_tracker and self.remote == None:
            self.connect_tracker()

        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(tvm_mod, target_host=target_host, target=target, params=params)
            #lib.save("./host.ll")
            #lib.imported_modules[0].save("./device.cl")

        if self.remote:
            print('Using Android OpenCL runtime over RPC')
            temp = util.tempdir()
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
            print('Using local runtime')
            ctx = tvm.context(target, 0)
            m = graph_runtime.create(graph, lib, ctx)
        if isinstance(input_shape, dict):
            key = list(input_shape)[0]
            input_shape = input_shape[key]
        else:
            key = 'data'
        m.set_input(key, np.random.normal(size=input_shape).astype('float32'))
        m.set_input(**params)
        print("Evaluating...")
        time_f = m.module.time_evaluator("run", ctx, number=10)
        cost = time_f().mean
        #prof_res = np.array(time_f().results) * 1000  # milliseconds
        #print("%-20s %-19s (%s)" % (network, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))
        print('%g secs/iteration\n' % cost)
        if self.remote:
            self.disconnect_tracker()


    def test_resnet50_ingestion(self, target="llvm"):
        gluon_model, input_shape = get_network("resnet50_v1", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        self.benchmark(mod, params, input_shape, target=target, target_host=self.host_target)

    def test_inceptionv3_ingestion(self, target="llvm"):
        gluon_model, input_shape = get_network("inceptionv3", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        self.benchmark(mod, params, input_shape, target=target, target_host=self.host_target)

    def test_mobilenetv1_ingestion(self, target="llvm"):
        gluon_model, input_shape = get_network("mobilenet1.0", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        def bench():
            self.benchmark(mod, params, input_shape, target=target, target_host=self.host_target)
        benchmark_index = len(self.benchmarks)
        self.benchmarks.append(bench)
        def tune():
            print("Extracting tasks")
            tasks = autotvm.task.extract_from_program(mod["main"], target=target, target_host=self.host_target, params=params)
            print("Tuning kernels")


            #self._tune_kernels(tasks, **tuning_options)
            tune_tasks(tasks, **tuning_options)

            print ("Apply best performing tuning profiles:")
            for i,task in enumerate(tasks):
                dispatch_context = autotvm.apply_history_best(tuning_options["log_filename"])
                best_config = dispatch_context.query(task.target, task.workload)
                print("task", i, best_config)
            def tuned_benchmark():
                with autotvm.apply_history_best(tuning_options["log_filename"]):
                    benchmark(mod, params, input_shape, target=target, target_host=self.host_target)
            self.benchmarks.pop(benchmark_index)
            self.benchmarks.append(tuned_benchmark)
        self.tuning_jobs.append(tune)

    def test_vgg16_ingestion(self, target="llvm"):
        gluon_model, input_shape = get_network("vgg16", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        self.benchmark(mod, params, input_shape, target=target, target_host=self.host_target)

    def test_vgg16bn_ingestion(self, target="llvm"):
        gluon_model, input_shape = get_network("vgg16_bn", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        self.benchmark(mod, params, input_shape, target=target, target_host=self.host_target)

    def test_mobilenetv3_ssdlite_ingestion(self, check = False):
        # TF pretrained model ssd_mobilenet_v3_small_coco
        # Link: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
        # Direct Tensorflow approach
        graph_def = tf_importer.get_workload(os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/models/mobilenetv3_ssdlite_tf1.15export/frozen_inference_graph.pb"))
        #tf.train.write_graph(graph_def, "./",name="mnv3-ssdlite-tf1.15export.pbtxt")
        graph_def = tf_importer.ProcessGraphDefParam(graph_def)
        if check:
            mod, params = relay.frontend.from_tensorflow(graph_def)
        else:
            mod, params = relay.frontend.from_tensorflow(graph_def, shape={"image_tensor": (1,320,320,3)})


    def test_deeplabv3_ingestion(self):
        graph_def = tf_importer.get_workload(os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/models/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb"))
        #tf.train.write_graph(graph_def, "./",name="deeplabv3_mobilenetv2.pbtxt")
        graph_def = tf_importer.ProcessGraphDefParam(graph_def)
        #mod, params = relay.frontend.from_tensorflow(graph_def)
        mod, params = relay.frontend.from_tensorflow(graph_def, shape={"ImageTensor": (1,320,320,3)})

    def test_inceptionv3_tf_ingestion(self, target="llvm"):
        graph_def = tf_importer.get_workload(os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/models/inception_v3_2016_08_28_frozen_opt.pb"))
        #tf.train.write_graph(graph_def, "./",name="inceptionv3.pbtxt")
        graph_def = tf_importer.ProcessGraphDefParam(graph_def)
        input_shape = {"input": (1,299,299,3)}
        #mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape)
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape, layout='NCHW')
        self.benchmark(mod, params, input_shape, target=target, target_host=self.host_target)

    def test_mobilenetv1_tf_ingestion(self, target="llvm"):
        graph_def = tf_importer.get_workload(os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/models/ssd_mobilenet_v1_coco.pb"))
        #tf.train.write_graph(graph_def, "./",name="ssd-mobilenetv1-coco.pbtxt")
        graph_def = tf_importer.ProcessGraphDefParam(graph_def)
        input_shape = {"image_tensor": (1,224,224,3)}
        #mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape)
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape, layout='NCHW')
        self.benchmark(mod, params, input_shape, target=target, target_host=self.host_target)

    def bench_conv2d_keras(self):
        # keras_model = tf.keras.Sequential()
        # keras_model.add(tf.keras.layers.InputLayer(input_shape=[224, 224, 3]))
        # keras_model.add(tf.keras.layers.Conv2D(16, 3, strides=(2,2), padding="SAME"))
        # print(keras_model.summary())
        #return keras_model, {'input_1': [1, 3, 224, 224]}

        model, shape_dict = keras_model, {'input_1': [1, 3, 224, 224]}
        mod, params = relay.frontend.from_keras(model, shape_dict)
        target = 'opencl'
        #target = "llvm -mcpu=skylake-avx512"
        # Tell TVM which device to run on
        ctx = tvm.opencl()

        # Compile the operators in the graph.
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, target=target, params=params)

        # Create a graph runtime and run our model
        #from tvm.contrib import graph_runtime
        from tvm.contrib.debugger import debug_runtime as graph_runtime
        # Dummy numpy data that we'll use for benchmarking
        test_data = np.random.normal(size=shape_dict['input_1']).astype('float32')
        # Create a runtime from our compile library and function graph.
        m = graph_runtime.create(graph, lib, ctx)
        # Set input and parameters
        m.set_input('input_1', test_data)
        m.set_input(**params)
        # Run the model
        m.run()
        tvm_output = m.get_output(0)

    def bench_conv2d_tf(self, layout = "NHWC", target="opencl", filter_size=3, tune=None, tuning_opt={}):
        if filter_size == 3:
            if layout == "NHWC":
                input = np.random.uniform(0, 10, size=(1,224,224,3)).astype(np.float32)
            else:
                input = np.random.uniform(0, 10, size=(1,3,224,224)).astype(np.float32)
            filter = np.random.uniform(size=(3,3,3,16)).astype(np.float32)
        elif filter_size == 1:
            if layout == "NHWC":
                input = np.random.uniform(0, 10, size=(1,20,20,240)).astype(np.float32)
            else:
                input = np.random.uniform(0, 10, size=(1,240,20,20)).astype(np.float32)
            filter = np.random.uniform(size=(1,1,240,40)).astype(np.float32)
        data = tf.placeholder(np.float32, input.shape, name="data")
        filter_p = tf.placeholder(np.float32, filter.shape, name="filter_p")
        conv = tf.nn.conv2d(data, filter_p, 1, padding="SAME", data_format=layout, name="Conv2D")

        with tf.Session() as sess:
            final_graph_def = tf_testing.AddShapesToGraphDef(sess, ["Conv2D"])
            mod, params = relay.frontend.from_tensorflow(final_graph_def,
                                                         layout=layout,
                                                         shape={"data" : input.shape})
            if tune is None:
                self.benchmark(mod, params, input.shape, target=target)
                return

            tasks = autotvm.task.extract_from_program(mod["main"],
                                                      target=target,
                                                      target_host=self.host_target,
                                                      params=params,
                                                      ops=(relay.op.get(op) for op in tune))
            self._tune_kernels(tasks, **tuning_opt)

            for task in tasks:
                dispatch_context = autotvm.apply_history_best("autotvm_tuning.log")
                best_config = dispatch_context.query(task.target, task.workload)
                print("\nBest config:")
                print(best_config)
            with autotvm.apply_history_best('autotvm_tuning.log'):
                self.benchmark(mod, params, input.shape, target=target)


    # Function for running a tuning jobs across tunable tasks
    def _tune_kernels(self, tasks,
                     measure_option,
                     early_stopping=None,
                     log_filename='tuning.log'):

        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

            # Create an XGBoost tuner
            tuner_obj = XGBTuner(task, loss_type='rank', feature_type='knob')

            # Try out 128 different schedules and pick the best (normally you'd try many more)
            n_trial = 2048
            tuner_obj.tune(n_trial=n_trial,
                           early_stopping=early_stopping,
                           measure_option=measure_option,
                           callbacks=[
                               autotvm.callback.progress_bar(n_trial, prefix=prefix),
                               autotvm.callback.log_to_file(log_filename)])

def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=20,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


# RN50, InceptionV3, MobilenetV1, VGG16, Mobilenetv3-ssd, DeepLabv3-mobilenetv2
if __name__ == "__main__":
    #test_runner = Executor()
    test_runner = Executor(use_tracker="android")

    # Successful
    #test_runner.test_resnet50_ingestion(target="opencl --device=mali") # 0.373638 secs/iteration
    #test_runner.test_mobilenetv1_ingestion(target="opencl --device=mali") # 0.0861629 secs/iteration
    #test_runner.test_mobilenetv1_ingestion(target="llvm")
    #test_runner.test_inceptionv3_tf_ingestion(target="opencl --device=mali") # 0.887882 secs/iteration
    #test_runner.test_resnet50_ingestion(target="opencl --device=mali")
    #test_runner.test_inceptionv3_tf_ingestion(target="llvm")

    # Unsuccessful
    #test_runner.test_inceptionv3_ingestion(target="opencl --device=mali") # Broken pipe during RPC TVMArray.copyfrom/to
    #test_runner.test_vgg16_ingestion(target="opencl --device=mali") # OOM error mrpc:RPCProces: Throwing OutOfMemoryError "Failed to allocate a 411041848 byte allocation with 4969365 free bytes and 251MB until OOM, target footprint 9938733, growth limit 268435456" (VmSize 6622184 kB)
    #test_runner.test_mobilenetv3_ssdlite_ingestion() # Ingestion error: null argument to op.where
    #test_runner.test_deeplabv3_ingestion() # Ingestion error: op.subtract takes two args not three
    #test_runner.test_mobilenetv1_tf_ingestion(target="opencl --device=mali") #  Ingestion error: File "/Users/csullivan/Projects/incubator-tvm/src/relay/transforms/fold_scale_axis.cc", line 246 # TVMError: FoldScaleAxis only accept dataflow-form

    # Untested



    # Tuning
    test_runner.test_mobilenetv1_ingestion(target="opencl --device=mali")
    test_runner.tune_pending_benchmarks()
    #test_runner.run_pending_benchmarks()
