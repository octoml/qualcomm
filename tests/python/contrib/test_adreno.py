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

def benchmark(tvm_mod, params, input_shape, target='llvm', target_host="llvm", remote=None):
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(tvm_mod, target_host=target_host, target=target, params=params)
        #lib.save("./host.ll")
        #lib.imported_modules[0].save("./device.cl")

    if remote:
        print('Using Android OpenCL runtime over RPC')
        temp = util.tempdir()
        dso_binary = "dev_lib_cl.so"
        dso_binary_path = temp.relpath(dso_binary)
        if "opencl" in target:
            ctx = remote.cl(0)
        else:
            ctx = remote.cpu(0)
        lib.export_library(dso_binary_path, ndk.create_shared)
        remote.upload(dso_binary_path)
        print("Uploading binary...")
        rlib = remote.load_module(dso_binary)
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

class Executor(object):
    def __init__(self, use_tracker=False):
        self.tracker = None
        self.remote = None
        self.host_target = "llvm"
        if use_tracker == "android":
            from tvm import rpc
            key = "android"
            tracker_host = os.environ["TVM_TRACKER_HOST"]
            tracker_port = int(os.environ["TVM_TRACKER_PORT"])
            print("Tracker attempting connection on {}:{}".format(tracker_host, tracker_port))
            self.tracker = rpc.connect_tracker(tracker_host, tracker_port)
            self.remote = self.tracker.request(key, priority=0,session_timeout=60)
            print("Tracker connected to remote RPC server")
            self.host_target = "llvm -mtriple=arm64-linux-android"
        elif use_tracker != False:
            class BackendNotImplementedForRPCBenchmarking(Exception):
                pass
            raise BackendNotImplementedForRPCBenchmarking

    def test_resnet50_ingestion(self, target="llvm"):
        gluon_model, input_shape = get_network("resnet50_v1", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        benchmark(mod, params, input_shape, target=target, target_host=self.host_target, remote=self.remote)

    def test_inceptionv3_ingestion(self, target="llvm"):
        gluon_model, input_shape = get_network("inceptionv3", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        benchmark(mod, params, input_shape, target=target, target_host=self.host_target, remote=self.remote)

    def test_mobilenetv1_ingestion(self, target="llvm"):
        gluon_model, input_shape = get_network("mobilenet1.0", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        benchmark(mod, params, input_shape, target=target, target_host=self.host_target, remote=self.remote)

    def test_vgg16_ingestion(self, target="llvm"):
        gluon_model, input_shape = get_network("vgg16", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        benchmark(mod, params, input_shape, target=target, target_host=self.host_target, remote=self.remote)

    def test_vgg16bn_ingestion(self, target="llvm"):
        gluon_model, input_shape = get_network("vgg16_bn", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        benchmark(mod, params, input_shape, target=target, target_host=self.host_target, remote=self.remote)

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
        benchmark(mod, params, input_shape, target=target, target_host=self.host_target, remote=self.remote)


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
                benchmark(mod, params, input.shape, target=target)
                return

            tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                                  params=params,
                                                  ops=(relay.op.get(op) for op in tune))
            self._tune_kernels(tasks, **tuning_opt)

            for task in tasks:
                dispatch_context = autotvm.apply_history_best("autotvm_tuning.log")
                best_config = dispatch_context.query(task.target, task.workload)
                print("\nBest config:")
                print(best_config)
            with autotvm.apply_history_best('autotvm_tuning.log'):
                benchmark(mod, params, input.shape, target=target)


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


# RN50, InceptionV3, MobilenetV1, VGG16, Mobilenetv3-ssd, DeepLabv3-mobilenetv2
if __name__ == "__main__":
    test_runner = Executor(use_tracker="android")
    #test_runner = Executor()
    test_runner.test_resnet50_ingestion(target="opencl --device=mali")
    #test_runner.test_inceptionv3_ingestion(target="opencl --device=mali")
    #test_runner.test_inceptionv3_tf_ingestion(target="opencl --device=mali")
    #test_runner.test_mobilenetv1_ingestion(target="opencl --device=mali")
    #test_runner.test_mobilenetv1_ingestion(target="opencl")
    #test_runner.test_mobilenetv1_ingestion(target="llvm")
    #test_runner.test_vgg16_ingestion(target="opencl --device=mali")

    #test_inceptionv3_ingestion()
    #test_mobilenetv1_ingestion()
    #test_vgg16_ingestion()
    #test_mobilenetv3_ssdlite_ingestion()
    #test_deeplabv3_ingestion()

    #bench_conv2d_keras()
    tuning_option = {
        'log_filename': "autotvm_tuning.log",
        'early_stopping': None,

        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(number=3, repeat=1,
                                       min_repeat_ms=10),
        ),
    }

    #bench_conv2d_tf(layout="NCHW", target="opencl --device=mali", filter_size=1)
    #bench_conv2d_tf(layout="NCHW", target="opencl --device=mali", filter_size=1, tune=["nn.conv2d"], tuning_opt=tuning_option)

    #bench_conv2d_tf(layout="NHWC", filter_size=1)
    #bench_conv2d_tf(layout="NHWC", filter_size=1, tune=["nn.conv2d"], tuning_opt=tuning_option)
    #bench_conv2d_tf(layout="NHWC", target="opencl", filter_size=1, tune=["nn.conv2d"], tuning_opt=tuning_option)
