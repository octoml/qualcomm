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
from tvm.autotvm.tuner import GATuner

#from tvm.contrib.debugger import debug_runtime as graph_runtime
from tvm.contrib import graph_runtime
from tvm.contrib import util, ndk

try:
    from tensorflow import lite as interpreter_wrapper
except ImportError:
    from tensorflow.contrib import lite as interpreter_wrapper

import onnx

def get_output_nodes_from_graph_def(graph_def):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    inputs = set()
    for op in graph.get_operations():
        for inp in op.inputs:
            inputs.add(inp.name.split(":")[0])
    outputs = []
    for op in graph.get_operations():
        if op.name not in inputs:
            outputs.append(op.name)
    return outputs

def get_input_data_shape_dict(graph_def, input_data):
    if isinstance(input_data, list):
        input_names = {}
        shape_dict = {}
        for i, _ in enumerate(input_data):
            input_names[i] = graph_def.graph.input[i].name
            shape_dict[input_names[i]] = input_data[i].shape
    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_data.shape}

    return input_names, shape_dict

def build_tvm_graph_from_tflite(tflite_model_buf, input_data, input_node, num_output=1, target='llvm',
                                out_names=None, mode='graph_runtime',layout='NCHW'):
    """ Generic function to compile on relay and execute on tvm """
    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except ImportError:
        raise ImportError("The tflite package must be installed")

    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)

    shape_dict = {}
    dtype_dict = {}
    for i, e in enumerate(input_node):
        shape_dict[e] = input_data[i].shape
        dtype_dict[e] = input_data[i].dtype.name

    mod, params = relay.frontend.from_tflite(tflite_model,
                                             shape_dict=shape_dict,
                                             dtype_dict=dtype_dict)

    return mod, params

def run_tvm_graph(tflite_model_buf, input_data, input_node, num_output=1, target='llvm',
                  out_names=None, mode='graph_runtime'):
    """ Generic function to compile on relay and execute on tvm """
    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except ImportError:
        raise ImportError("The tflite package must be installed")

    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)

    shape_dict = {}
    dtype_dict = {}
    for i, e in enumerate(input_node):
        shape_dict[e] = input_data[i].shape
        dtype_dict[e] = input_data[i].dtype.name

    mod, params = relay.frontend.from_tflite(tflite_model,
                                             shape_dict=shape_dict,
                                             dtype_dict=dtype_dict)

    if mode in ['debug', 'vm']:
        ex = relay.create_executor(mode, mod=mod, ctx=tvm.cpu(), target="llvm")
        inputs = []
        for param in mod['main'].params:
            found = False
            for i, n in enumerate(input_node):
                if n == param.name_hint:
                    found = True
                    inputs.append(tvm.nd.array(input_data[i]))
                    break
            # Interpreter doesn't bind constants, so still need to find in params
            if not found:
                inputs.append(tvm.nd.array(params[param.name_hint]))
        result = ex.evaluate()(*inputs)
        return vmobj_to_list(result)
    else:
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(mod, target, params=params)

        ctx = tvm.context(target, 0)
        from tvm.contrib import graph_runtime
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        for i, e in enumerate(input_node):
            m.set_input(e, tvm.nd.array(input_data[i].astype(input_data[i].dtype)))

        m.set_input(**params)
        # execute
        m.run()
        # get outputs
        assert out_names is None or num_output == len(out_names), "out_names: {} num_output: {}".format(
            out_names, num_output)
        tvm_output_list = []
        for i in range(0, num_output):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list

def run_tflite_graph(tflite_model_buf, input_data):
    """ Generic function to execute TFLite """
    input_data = convert_to_list(input_data)

    interpreter = interpreter_wrapper.Interpreter(model_content=tflite_model_buf)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(input_details)):
        interpreter.resize_tensor_input(input_details[i]['index'], input_data[i].shape)
    interpreter.allocate_tensors()

    # set input
    assert len(input_data) == len(input_details)
    for i in range(len(input_details)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])

    # Run
    interpreter.invoke()

    # get output
    tflite_output = list()
    for i in range(len(output_details)):
        tflite_output.append(interpreter.get_tensor(output_details[i]['index']))

    return tflite_output

def compare_tflite_with_tvm(in_data, in_name, input_tensors,
                            output_tensors, init_global_variables=False,
                            out_names=None, quantized=False, input_range=None, mode='graph_runtime'):
    """Generic function to generate and compare TFLite and TVM output"""
    in_data = convert_to_list(in_data)
    in_name = convert_to_list(in_name)
    out_names = convert_to_list(out_names)
    in_node = [0] * len(in_name)
    for i in range(len(in_name)):
        in_node[i] = in_name[i].split(':')[0] if ":" in in_name[i] else in_name[i]

    with tf.Session() as sess:
        if init_global_variables:
            sess.run(variables.global_variables_initializer())
        # convert to tflite model
        converter = tf.lite.TFLiteConverter.from_session(
            sess, input_tensors, output_tensors)

        if quantized:
            converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
            input_arrays = converter.get_input_arrays()
            input_stats = {}
            # calculate the mean and quantization scale for every input tensor,
            # with respect to its fp32 input range, defined in fake_quant.
            # s = 255/(fmax-fmin);  m = -fmin*s (the zero point)
            for i in input_arrays:
                try:
                    quant_scale = 255 / (input_range[i][1] - input_range[i][0])
                except ZeroDivisionError:
                    raise ZeroDivisionError('Min and max of the input range for tensor ' + i + ' can\'t be equal')
                mean = - input_range[i][0] * quant_scale
                input_stats[i] = (mean, quant_scale)
            converter.quantized_input_stats = input_stats

        tflite_model_buffer = converter.convert()
        tflite_output = run_tflite_graph(tflite_model_buffer, in_data)

        for device in ["llvm"]:
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                continue

            tvm_output = run_tvm_graph(tflite_model_buffer, in_data, in_node, target=device,
                                       num_output=len(out_names), out_names=out_names, mode=mode)

            # WARNING: the results could well be random values clipped to 0 or 255 because of badly tuned output
            # range for the specific operator. While adding test ensure that we aren't getting only clipped values
            # in output tensors that still pass the assertion. For reference see _test_elemwise_qnn_out_range()
            if quantized:
                for i in range(len(tflite_output)):
                    # allow absolute tolerance of 1 in the quantized results
                    tvm.testing.assert_allclose(tflite_output[i], tvm_output[i], atol=1, rtol=1e-5)
            else:
                for i in range(len(tflite_output)):
                    tvm.testing.assert_allclose(tflite_output[i], tvm_output[i], atol=1e-5, rtol=1e-5)




def convert_to_fp16(path_to_model, input_name, output_names, target_type='fp16', conversion_blacklist=[]):
    from tensorflow.core.framework import types_pb2, graph_pb2, attr_value_pb2
    from tensorflow.tools.graph_transforms import TransformGraph
    from google.protobuf import text_format

    # Const should be float32 in object detection api during nms (see here: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/non-max-suppression-v4.html)
    conversion_blacklist.extend(["Postprocessor/BatchMultiClassNonMaxSuppression/MultiClassNonMaxSuppression/non_max_suppression/iou_threshold",
                                 "Postprocessor/BatchMultiClassNonMaxSuppression/MultiClassNonMaxSuppression/non_max_suppression/score_threshold"])

    def rewrite_batch_norm_node_v2(node, graph_def, target_type='fp16'):
        """
        Rewrite FusedBatchNorm with FusedBatchNormV2 for reserve_space_1 and reserve_space_2 in FusedBatchNorm require float32 for
        gradient calculation (See here: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/fused-batch-norm)
        """
        if target_type == 'fp16':
            dtype = types_pb2.DT_HALF
        elif target_type == 'fp64':
            dtype = types_pb2.DT_DOUBLE
        else:
            dtype = types_pb2.DT_FLOAT
        new_node = graph_def.node.add()
        new_node.op = "FusedBatchNormV2"
        new_node.name = node.name
        new_node.input.extend(node.input)
        new_node.attr["U"].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
        for attr in list(node.attr.keys()):
            if attr == "T":
                node.attr[attr].type = dtype
            new_node.attr[attr].CopyFrom(node.attr[attr])
        print("rewrite fused_batch_norm done!")

    def convert_graph_to_fp16(model_path, input_name=None, output_names=None, target_type='fp16'):
        if target_type == 'fp16':
            dtype = types_pb2.DT_HALF
        elif target_type == 'fp64':
            dtype = types_pb2.DT_DOUBLE
        else:
            dtype = types_pb2.DT_FLOAT
        source_graph_def = tf_importer.get_workload(model_path)
        target_graph_def = graph_pb2.GraphDef()
        target_graph_def.versions.CopyFrom(source_graph_def.versions)
        for node in source_graph_def.node:
            # fused batch norm node
            # if node.op == "FusedBatchNorm":
            #     rewrite_batch_norm_node_v2(node, target_graph_def, target_type=target_type)
            #     continue
            # replicate node
            new_node = target_graph_def.node.add()
            new_node.op = node.op
            new_node.name = node.name
            new_node.input.extend(node.input)
            attrs = list(node.attr.keys())
            # keep batch norm params node
            # if ("BatchNorm" in node.name) or ('batch_normalization' in node.name):
            #     for attr in attrs:
            #         new_node.attr[attr].CopyFrom(node.attr[attr])
            #     continue
            # replace dtype in node attr with target dtype
            for attr in attrs:
                # keep special node in fp32
                if node.name in conversion_blacklist:
                    new_node.attr[attr].CopyFrom(node.attr[attr])
                    continue
                if node.attr[attr].type == types_pb2.DT_FLOAT:
                    # modify node dtype
                    new_node.attr[attr].type = dtype
                    continue
                if attr == "value":
                    tensor = node.attr[attr].tensor
                    if tensor.dtype == types_pb2.DT_FLOAT:
                        # if float_val exists
                        if tensor.float_val:
                            float_val = tf.make_ndarray(node.attr[attr].tensor)
                            new_node.attr[attr].tensor.CopyFrom(tf.make_tensor_proto(float_val, dtype=dtype))
                            continue
                        # if tensor content exists
                        if tensor.tensor_content:
                            tensor_shape = [x.size for x in tensor.tensor_shape.dim]
                            tensor_weights = tf.make_ndarray(tensor)
                            # reshape tensor
                            tensor_weights = np.reshape(tensor_weights, tensor_shape)
                            tensor_proto = tf.make_tensor_proto(tensor_weights, dtype=dtype)
                            new_node.attr[attr].tensor.CopyFrom(tensor_proto)
                            continue
                new_node.attr[attr].CopyFrom(node.attr[attr])
        # transform graph
        if output_names:
            if not input_name:
                input_name = []
            transforms = ["strip_unused_nodes"]
            target_graph_def = TransformGraph(target_graph_def, input_name, output_names, transforms)
        return target_graph_def


    return convert_graph_to_fp16(path_to_model, input_name=input_name, output_names=output_names, target_type=target_type)



device_key = "android"
tracker_host = os.environ["TVM_TRACKER_HOST"]
tracker_port = int(os.environ["TVM_TRACKER_PORT"])

tuning_options = {
    'log_filename': "autotvm_tuning.log",
    'early_stopping': None,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=10),
        runner=autotvm.RPCRunner(device_key, host=tracker_host, port=tracker_port, number=5, timeout=10),
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

    def tune_pending_benchmarks(self, apply_previous_tune=False, opt=tuning_options):
        for tune in self.tuning_jobs:
            tune(apply_previous_tune, options=tuning_options)

    def benchmark(self,tvm_mod, params, input_shape, target='llvm', target_host="llvm", dtype='float32'):
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
        m.set_input(key, np.random.normal(size=input_shape).astype(dtype))
        m.set_input(**params)
        print("Evaluating...")
        time_f = m.module.time_evaluator("run", ctx, number=10)
        cost = time_f().mean
        #prof_res = np.array(time_f().results) * 1000  # milliseconds
        #print("%-20s %-19s (%s)" % (network, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))
        print('%g secs/iteration\n' % cost)

    def schedule_jobs(self, mod, params, input_shape, dtype, target):
        def bench():
            self.benchmark(mod, params, input_shape, target=target, target_host=self.host_target, dtype=dtype)
        benchmark_index = len(self.benchmarks)
        self.benchmarks.append(bench)
        def tune(apply_previous_tune=False, options=tuning_options):
            print("Extracting tasks")
            tasks = autotvm.task.extract_from_program(mod["main"], target=target, target_host=self.host_target, params=params)
            if apply_previous_tune == False:
                print("Tuning kernels")
                tune_tasks(tasks, **options)

            print ("Apply best performing tuning profiles:")
            for i,task in enumerate(tasks):
                dispatch_context = autotvm.apply_history_best(options["log_filename"])
                best_config = dispatch_context.query(task.target, task.workload)
                print("task", i, best_config)
            def tuned_benchmark():
                with autotvm.apply_history_best(options["log_filename"]):
                    self.benchmark(mod, params, input_shape, target=target, target_host=self.host_target)
            self.benchmarks.pop(benchmark_index)
            self.benchmarks.append(tuned_benchmark)
        self.tuning_jobs.append(tune)

    def test_resnet50_ingestion(self, target="llvm", dtype='float32'):
        gluon_model, input_shape = get_network("resnet50_v1", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        self.schedule_jobs(mod, params, input_shape, dtype, target)

    def test_resnet50_keras_ingestion(self, target="llvm", dtype='float32'):
        import keras
        keras.backend.set_floatx(dtype)
        model = keras.applications.resnet50.ResNet50()
        input_shape = {model.layers[0].name: (1, 3, 224, 224)}
        mod, params = relay.frontend.from_keras(model, input_shape)
        self.schedule_jobs(mod, params, input_shape, dtype, target)

    def test_inceptionv3_ingestion(self, target="llvm"):
        gluon_model, input_shape = get_network("inceptionv3", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        self.schedule_jobs(mod, params, input_shape, dtype, target)

    def test_mobilenetv1_ingestion(self, target="llvm", dtype='float32'):
        gluon_model, input_shape = get_network("mobilenet1.0", batch_size=1)
        if dtype != 'float32':
            gluon_model.cast(dtype)
            gluon_model.hybridize()
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape}, dtype=dtype)
        self.schedule_jobs(mod, params, input_shape, dtype, target)

    def test_vgg16_keras_ingestion(self, target="llvm", dtype='float32'):
        import keras
        keras.backend.set_floatx(dtype)
        model = keras.applications.vgg16.VGG16()
        input_shape = {model.layers[0].name: (1, 3, 224, 224)}
        mod, params = relay.frontend.from_keras(model, input_shape)
        self.schedule_jobs(mod, params, input_shape, dtype, target)

    def test_vgg16_ingestion(self, target="llvm", dtype='float32'):
        gluon_model, input_shape = get_network("vgg16", batch_size=1)
        mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
        self.schedule_jobs(mod, params, input_shape, dtype, target)

    def test_vgg16bn_ingestion(self, target="llvm", dtype='float32'):
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
        self.schedule_jobs(mod, params, input_shape, dtype, target)


    def test_deeplabv3_ingestion(self, target="llvm", dtype='float32'):
        graph_def = tf_importer.get_workload(os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/models/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb"))
        #tf.train.write_graph(graph_def, "./",name="deeplabv3_mobilenetv2.pbtxt")
        graph_def = tf_importer.ProcessGraphDefParam(graph_def)
        #mod, params = relay.frontend.from_tensorflow(graph_def)
        mod, params = relay.frontend.from_tensorflow(graph_def, shape={"ImageTensor": (1,320,320,3)})
        self.schedule_jobs(mod, params, input_shape, dtype, target)

    def test_deeplabv3_tflite_ingestion(self, target="llvm", dtype='float32'):
        model_name = "deeplabv3_mnv2_pascal_train_aug"
        graph_def_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/models/")
        graph_def_file = os.path.abspath(graph_def_path  + "/" +  model_name + "/frozen_inference_graph.pb")
        input_arrays = ["ImageTensor"]
        output_arrays = ["SemanticPredictions"]
        input_shape = {"ImageTensor": (1,224,224,3)}
        converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays, input_shapes=input_shape)
        converter.inference_type = tf.float32
        converter.inference_input_type = tf.uint8
        converter.quantized_input_stats = {input_arrays[0] : (128, 127)}
        #converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
        tflite_model = converter.convert()
        tflite_model_file = os.path.abspath(graph_def_path + "/" + model_name + "/" + model_name + "_converted_" + dtype + ".tflite")
        open(tflite_model_file, "wb").write(tflite_model)
        with open(tflite_model_file, "rb") as f:
            tflite_model_buf = f.read()
        data = np.random.uniform(size=input_shape["ImageTensor"]).astype(dtype)
        mod, params = build_tvm_graph_from_tflite(tflite_model_buf, data, 'ImageTensor')


    def test_inceptionv3_tf_ingestion(self, target="llvm", dtype='float32'):
        if dtype == 'float32':
            graph_def = tf_importer.get_workload(os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/models/inception_v3_2016_08_28_frozen_opt.pb"))
            #tf.train.write_graph(graph_def, "./",name="inceptionv3.pbtxt")
        elif dtype == 'float16':
            graph_def = convert_to_fp16(os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/models/inception_v3_2016_08_28_frozen_opt.pb"), input_name='input',output_names=['InceptionV3/Predictions/Reshape_1'])
            #tf.train.write_graph(graph_def, "./",name="inceptionv3_fp16.pbtxt")
        else:
            raise "Only fp32/16 are supported"
        graph_def = tf_importer.ProcessGraphDefParam(graph_def)
        input_shape = {"input": (1,299,299,3)}
        #mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape)
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape, layout='NCHW')
        self.schedule_jobs(mod, params, input_shape, dtype, target)

    def test_mobilenetv1_tf_ingestion(self, target="llvm"):
        graph_def = tf_importer.get_workload(os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/models/ssd_mobilenet_v1_coco.pb"))
        #tf.train.write_graph(graph_def, "./",name="ssd-mobilenetv1-coco.pbtxt")
        graph_def = tf_importer.ProcessGraphDefParam(graph_def)
        input_shape = {"image_tensor": (1,224,224,3)}
        #mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape)
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape, layout='NCHW')
        self.schedule_jobs(mod, params, input_shape, dtype, target)


    def test_mobilenetv1_tflite_ingestion(self, target="llvm", dtype='float32'):
        model_name = 'mobilenet_v1_1.0_224'
        http = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/" + model_name + ".tgz"
        if dtype == 'float32':
            tflite_model_file = tf_testing.get_workload_official(http, model_name + ".tflite")
        elif dtype == 'float16':
            graph_def_path = os.path.abspath("/Users/csullivan/Downloads/" + model_name)
            graph_def_file = os.path.abspath(graph_def_path  + "/" + model_name + "_frozen.pb")
            input_arrays = ["input"]
            output_arrays = ["MobilenetV1/Predictions/Softmax"]
            converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
            converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
            tflite_model = converter.convert()
            tflite_model_file = os.path.abspath(graph_def_path + "/" + model_name + "_converted_" + dtype + ".tflite")
            open(tflite_model_file, "wb").write(tflite_model)

        with open(tflite_model_file, "rb") as f:
            tflite_model_buf = f.read()
        data = np.random.uniform(size=(1, 224, 224, 3)).astype(dtype)
        mod, params = build_tvm_graph_from_tflite(tflite_model_buf, data, 'input')
        desired_layouts = {
              'nn.conv2d': ['NCHW', 'default'],
              'nn.depthwise_conv2d': ['NCHW', 'default'],
        }
        input_shape = {'input':data.shape}
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        self.schedule_jobs(mod, params, input_shape, dtype, target)
        #self.benchmark(mod, params, input_shape, target=target, target_host=self.host_target) # NHWC error

        # graph_def = tf_importer.ProcessGraphDefParam(graph_def)
        # input_shape = {"image_tensor": (1,224,224,3)}
        # #mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape)
        # mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape, layout='NCHW')
        # self.benchmark(mod, params, input_shape, target=target, target_host=self.host_target)

    def test_mobilenetv1_onnx_ingestion(self, target="llvm"):
        graph_file = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/models/ssd_mnv1.op10.onnx")
        model = onnx.load_model(graph_file)
        data = np.random.uniform(size=(1, 3, 224, 224)).astype('float32')
        input_names, shape_dict = get_input_data_shape_dict(model, data)
        mod, params = relay.frontend.from_onnx(model, shape_dict, opset=11)
        self.schedule_jobs(mod, params, input_shape, dtype, target)

def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1024,
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
    #test_runner.test_inceptionv3_tf_ingestion(target="opencl --device=mali", dtype='float16') # 0.637642 secs/iteration
    #test_runner.test_resnet50_ingestion(target="opencl --device=mali")
    #test_runner.test_resnet50_ingestion(target="opencl --device=mali", dtype='float16')
    #test_runner.test_inceptionv3_tf_ingestion(target="llvm")

    # Unsuccessful
    #test_runner.test_inceptionv3_ingestion(target="opencl --device=mali") # Broken pipe during RPC TVMArray.copyfrom/to
    #test_runner.test_vgg16_ingestion(target="opencl --device=mali") # OOM error mrpc:RPCProces: Throwing OutOfMemoryError "Failed to allocate a 411041848 byte allocation with 4969365 free bytes and 251MB until OOM, target footprint 9938733, growth limit 268435456" (VmSize 6622184 kB)
    #test_runner.test_mobilenetv3_ssdlite_ingestion() # Ingestion error: null argument to op.where
    #test_runner.test_deeplabv3_ingestion() # Ingestion error: op.subtract takes two args not three
    #test_runner.test_mobilenetv1_tf_ingestion(target="opencl --device=mali") #  Ingestion error: File "/Users/csullivan/Projects/incubator-tvm/src/relay/transforms/fold_scale_axis.cc", line 246 # TVMError: FoldScaleAxis only accept dataflow-form
    #test_runner.test_mobilenetv1_tflite_ingestion(target="opencl --device=mali")
    #test_runner.test_mobilenetv1_onnx_ingestion(target="opencl --device=mali")

    # Untested

    # Tuning
    #test_runner.test_inceptionv3_tf_ingestion(target="opencl --device=mali")
    #test_runner.test_mobilenetv1_ingestion(target="opencl --device=mali",dtype='float16')
    #test_runner.test_mobilenetv1_ingestion(target="opencl --device=mali",dtype='float32')
    #test_runner.test_mobilenetv1_ingestion(target="opencl --device=mali")



    #test_runner.tune_pending_benchmarks(apply_previous_tune=True, opt=tuning_options)
    #test_runner.run_pending_benchmarks()

    # tflite

    #test_runner.test_mobilenetv1_tflite_ingestion(target="opencl --device=mali", dtype='float32')
    # opt = tuning_options
    # opt['log_filename'] = 'mobilenetv1.fp32.autotvm.log'
    # test_runner.tune_pending_benchmarks(apply_previous_tune=True, opt=opt)
    #test_runner.run_pending_benchmarks()

    #opt['log_filename'] = 'mobilenetv1.fp32.autotvm_tuning.fp16.poco865.log'
    #test_runner.tune_pending_benchmarks(apply_previous_tune=True, opt=opt)

    ## tune tflite fp16
    # test_runner.test_mobilenetv1_tflite_ingestion(target="opencl --device=mali", dtype='float16')
    # opt = tuning_options
    # opt['log_filename'] = 'mobilenetv1_tflite_fp16_autotvm_tuning.log'
    # test_runner.tune_pending_benchmarks(opt=opt)
    # test_runner.run_pending_benchmarks()

    ## mobilenetv1 tflite fp32 tuned
    # 0.0359528 secs/iteration
    # test_runner.test_mobilenetv1_tflite_ingestion(target="opencl --device=mali", dtype='float32')
    # opt = tuning_options
    # opt['log_filename'] = 'mobilenetv1_tflite_fp32_autotvm_tuning.log'
    # test_runner.tune_pending_benchmarks(opt=opt)
    # test_runner.run_pending_benchmarks()


    #test_runner.test_mobilenetv1_tflite_ingestion(target="opencl --device=mali", dtype='float16')
    #test_runner.run_pending_benchmarks()

    # mobilenetv1 tflite fp32 (tune from gluon import)
    # 0.0389187 secs/iteration
    # test_runner.test_mobilenetv1_tflite_ingestion(target="opencl --device=mali", dtype='float32')
    # opt = tuning_options
    # opt['log_filename'] = 'mobilenetv1.fp32.autotvm.log' # from gluon tuning, some missing
    # test_runner.tune_pending_benchmarks(apply_previous_tune=True, opt=opt)
    # test_runner.run_pending_benchmarks()

    # mobilenetv1 tflite fp16 (tune from gluon import)
    # 0.0352906 secs/iteration
    # test_runner.test_mobilenetv1_tflite_ingestion(target="opencl --device=mali", dtype='float16')
    # opt = tuning_options
    # opt['log_filename'] = 'mobilenetv1_tflite_fp16_autotvm_tuning.log'
    # test_runner.tune_pending_benchmarks(apply_previous_tune=True, opt=opt)
    # test_runner.run_pending_benchmarks()


    # mobilenetv1 gluon fp32
    # 0.0346617 secs/iteration
    # test_runner.test_mobilenetv1_ingestion(target="opencl --device=mali",dtype='float32')
    # opt = tuning_options
    # opt['log_filename'] = 'mobilenetv1_gluon_fp32_autotvm_tuning.log'
    # test_runner.tune_pending_benchmarks(apply_previous_tune=True, opt=opt)
    # test_runner.run_pending_benchmarks()

    # mobilenetv1 gluon fp16
    # 0.0300264 secs/iteration
    # test_runner.test_mobilenetv1_ingestion(target="opencl --device=mali",dtype='float16')
    # opt = tuning_options
    # opt['log_filename'] = 'mobilenetv1_gluon_fp16_autotvm_tuning.log'
    # test_runner.tune_pending_benchmarks(apply_previous_tune=True, opt=opt)
    # test_runner.run_pending_benchmarks()

    # # resnet50 gluon fp32
    # 0.372 secs/iteration
    # test_runner.test_resnet50_ingestion(target="opencl --device=mali", dtype='float32')
    # test_runner.run_pending_benchmarks()

    # resnet50 gluon fp16
    # 0.372 secs/iteration
    # test_runner.test_resnet50_ingestion(target="opencl --device=mali", dtype='float16')
    # test_runner.run_pending_benchmarks()

    # test_runner.test_mobilenetv1_ingestion(target="opencl --device=mali", dtype='float16')
    # opt = tuning_options
    # opt['log_filename'] = 'tmp_finish.log'
    # test_runner.tune_pending_benchmarks(opt=opt)
    # test_runner.run_pending_benchmarks()


    #test_runner.test_inceptionv3_ingestion(target="opencl --device=mali") # Broken pipe during RPC TVMArray.copyfrom/to
    #test_runner.run_pending_benchmarks()

    # resnet50 keras/tf fp32
    # 0.365892 secs/iteration
    # test_runner.test_resnet50_keras_ingestion(target="opencl --device=mali")
    # test_runner.run_pending_benchmarks()

    test_runner.test_deeplabv3_tflite_ingestion(target="opencl --device=mali")
    test_runner.run_pending_benchmarks()
