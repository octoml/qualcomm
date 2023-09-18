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
from tvm.relay import testing
from tvm import autotvm
from tvm.contrib import utils, ndk
from tvm.topi import testing
from common import convert_to_dtype, advanced_time_evaluator


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


    def get_onnx_from_tf1(self, model_url, filename, input_names, output_names, shape_override = None):
        tf_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/models/{}.pb".format(filename)
        )

        from tvm.contrib import download
        download.download(model_url, tf_model_file)
        # converted using command line:
        # python -m tf2onnx.convert --graphdef mace_resnet-v2-50.pb --output mace_resnet-v2-50.onnx --inputs input:0[1,224,224,3] --outputs resnet_v2_50/predictions/Reshape_1:0
        onnx_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/models/{}.onnx".format(filename))
        if os.path.exists(onnx_model_file) == False:
            import tf2onnx
            import tensorflow as tf
            try:
                tf_compat_v1 = tf.compat.v1
            except ImportError:
                tf_compat_v1 = tf
            # Tensorflow utility functions
            import tvm.relay.testing.tf as tf_testing

            with tf_compat_v1.gfile.GFile(tf_model_file, "rb") as f:
                graph_def = tf_compat_v1.GraphDef()
                graph_def.ParseFromString(f.read())
                #graph = tf.import_graph_def(graph_def, name="")
                # Call the utility to import the graph definition into default graph.
                graph_def = tf_testing.ProcessGraphDefParam(graph_def)

                model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(graph_def,
                    name=filename, input_names=input_names, output_names=output_names,
                    shape_override = shape_override,
                    output_path=onnx_model_file)

        return onnx_model_file


    def get_graphdef_from_tf1(self, model_url, filename):
        graph_def = None
        tf_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/models/{}.pb".format(filename)
        )

        from tvm.contrib import download
        download.download(model_url, tf_model_file)
        # converted using command line:
        # python -m tf2onnx.convert --graphdef mace_resnet-v2-50.pb --output mace_resnet-v2-50.onnx --inputs input:0[1,224,224,3] --outputs resnet_v2_50/predictions/Reshape_1:0
        onnx_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/../models/{}.onnx".format(filename))
        import tensorflow as tf
        try:
            tf_compat_v1 = tf.compat.v1
        except ImportError:
            tf_compat_v1 = tf
        # Tensorflow utility functions
        import tvm.relay.testing.tf as tf_testing

        with tf_compat_v1.gfile.GFile(tf_model_file, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        return graph_def

    def import_mace_mobilenetv1_nhwc(self, target="llvm", dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/mobilenet-v1/mobilenet-v1-1.0.pb"
        filename = "mace_mobilenet-v1-1.0"
        graph_def = self.get_graphdef_from_tf1(model_url, filename)
        shape_dict = {"input": (1, 224, 224, 3)}
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict,
                                        outputs=["MobilenetV1/Predictions/Reshape_1"])

        # downcast to float16
        mod = convert_to_dtype(mod["main"], dtype)
        dtype = "float32" if dtype == "float32" else "float16"

        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, "NHWC", preproc="keras_mobilenetv1"))

    def import_mace_mobilenetv1_nchw(self, target="llvm", dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/mobilenet-v1/mobilenet-v1-1.0.pb"
        filename = "mace_mobilenet-v1-1.0"
        input_names = ["input:0"]
        output_names = ["MobilenetV1/Predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'input:0': [1, 224, 224, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

        # downcast to float16
        mod = convert_to_dtype(mod["main"], dtype)
        dtype = "float32" if dtype == "float32" else "float16"

        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, "NHWC", preproc="keras_mobilenetv1"))

    def import_conv2d_deeplabv3(self, target="llvm", dtype="float32"):
        dtype_init="float32"
        input_shape = (1, 513, 513, 3)
        filter_shape = (3, 3, 3, 32)
        bias_shape = (1, 1, 1, 32)
        A = relay.var("data", shape=input_shape, dtype=dtype_init)
        B = relay.var("weight", shape=filter_shape, dtype=dtype_init)
        bias = relay.var("bias", shape=bias_shape, dtype=dtype_init)

        #C = relay.nn.relu(A)
        conv = relay.nn.conv2d(A, B, data_layout="NHWC", kernel_layout="HWIO",
                            padding=[1,1,1,1],strides=[2,2],
                            out_dtype=dtype_init, channels=32, kernel_size=(3,3))
        D = relay.op.add(conv, bias)
        D = relay.op.nn.relu(D)

        mod = relay.Function([A, B, bias], D)
        np.random.seed(0)
        initializer = relay.testing.init.Xavier()
        filter_data = np.zeros(filter_shape).astype(dtype_init)
        bias_data = np.zeros(bias_shape).astype(dtype_init)
        initializer("weight", filter_data)
        initializer("bias", bias_data)
        params = {
            "weight": tvm.nd.array(filter_data),
            "bias" : tvm.nd.array(bias_data),
        }

        # downcast to float16
        mod = convert_to_dtype(mod, dtype)
        dtype = "float32" if dtype == "float32" else "float16"

        return (mod, params, {"data": input_shape}, dtype, target)


    def import_mace_resnet50_v2(self, target="llvm", dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/resnet-v2-50/resnet-v2-50.pb"
        filename = "mace_resnet-v2-50"
        input_names = ["input:0"]
        shape_override = {"input:0": [1, 299, 299, 3]}
        output_names = ["resnet_v2_50/predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names, shape_override)
        import onnx
        model = onnx.load(onnx_model_file)
        mod, params = relay.frontend.from_onnx(model, shape_override, freeze_params=True)

        # downcast to float16
        mod = convert_to_dtype(mod["main"], dtype)
        dtype = "float32" if dtype == "float32" else "float16"

        return (mod, params, shape_override, dtype, target, \
                ImageNetValidator(shape_override, "NHWC", preproc="keras"))


    def import_ac_resnet50_tf(self, target="llvm", dtype="float32"):
        model_url = "https://download.01.org/opencv/public_models/012020/resnet-50-tf/resnet_v1-50.pb"
        filename = "resnet_v1-50"
        input_names = ["map/TensorArrayStack/TensorArrayGatherV3:0"]
        shape_override = {"map/TensorArrayStack/TensorArrayGatherV3:0": [1, 224, 224, 3]}
        output_names = ["softmax_tensor:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names, shape_override)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'map/TensorArrayStack/TensorArrayGatherV3:0': [1, 224, 224, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

        mod = relay.quantize.prerequisite_optimize(mod, params)

        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, "NHWC", preproc="keras_mobilenetv1"))


    def import_mace_inceptionv3(self, target="llvm", dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/inception-v3/inception-v3.pb"
        filename = "mace_inception-v3"
        input_names = ["input:0"]
        output_names = ["InceptionV3/Predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'input:0': [1, 299, 299, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

        # downcast to float16
        mod = convert_to_dtype(mod["main"], dtype)
        dtype = "float32" if dtype == "float32" else "float16"

        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, "NHWC", preproc="keras"))

    def import_mxnet_vgg16(self, target="llvm", dtype="float32"):
        model, input_shape = gluon_model("vgg16", batch_size=1)
        shape_dict = {"data": input_shape}
        mod, params = relay.frontend.from_mxnet(model, shape_dict)

        # downcast to float16
        mod = convert_to_dtype(mod["main"], dtype)
        dtype = "float32" if dtype == "float32" else "float16"

        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, preproc="mxnet"))

    def import_mace_deeplabv3(self, target="llvm", dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/deeplab-v3-plus/deeplab-v3-plus-mobilenet-v2.pb"
        filename = "mace_deeplab-v3-plus-mobilenet-v2"
        graph_def = self.get_graphdef_from_tf1(model_url, filename)
        shape_dict = {"sub_7": (1, 513, 513, 3)}
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict,
                                        outputs=["ResizeBilinear_2"])

        # hack for insufficient pattern support in FlattenAtrousConv
        # if it is called after convert to fp16 with mixed precision, it will not be able
        # to catch cast.
        # TODO(amalyshe) We need to extend FlattenAtrousConv but for now we are calling it
        # explicitly
        mod = tvm.relay.transform.FlattenAtrousConv()(mod)
        # downcast to float16
        mod = convert_to_dtype(mod["main"], dtype)
        dtype = "float32" if dtype == "float32" else "float16"

        return (mod, params, shape_dict, dtype, target, Deeplabv3Validator(shape_dict, dtype))


    def import_mace_yolov3(self, target="llvm", dtype="float32"):
        model_url = "http://cnbj1.fds.api.xiaomi.com/mace/miai-models/yolo-v3/yolo-v3.pb"
        filename = "mace_yolo-v3"
        graph_def = self.get_graphdef_from_tf1(model_url, filename)
        shape_dict = {"input_1": (1, 416, 416, 3)}
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict,
                                        outputs=["conv2d_59/BiasAdd","conv2d_67/BiasAdd","conv2d_75/BiasAdd"])

        # downcast to float16
        mod = convert_to_dtype(mod["main"], dtype)
        dtype = "float32" if dtype == "float32" else "float16"

        return (mod, params, shape_dict, dtype, target, Yolov3Validator(shape_dict))


    def import_onnx_ssd_resnet34(self, target="llvm", dtype="float32"):
        archive_url = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd/model/ssd-12.tar.gz"
        filename = "ssd-12.tar.gz"
        from tvm.contrib import download
        import onnx
        import tarfile
        download.download(archive_url, filename)
        archive = tarfile.open(filename)
        directory = "ssd_resnet34"
        archive.extractall(directory)
        archive.close()
        directory = os.path.join(directory, "ssd-12")
        model_file = os.path.join(directory, "ssd-12.onnx")
        onnx_model = onnx.load(model_file)
        shape_dict = {"image": (1, 3, 1200, 1200)}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
        test_files_dir = os.path.join(directory, "test_data_set_0")

        # downcast to float16
        mod = convert_to_dtype(mod["main"], dtype)
        dtype = "float32" if dtype == "float32" else "float16"

        #return (mod, params, shape_dict, dtype, target, ONNXTestSamplesValidator(test_files_dir, input_names=list(shape_dict.keys())))
        return (mod, params, shape_dict, dtype, target, SSDResnetValidator())


    def import_onnx_yolo_v3(self, target="llvm", dtype="float32"):
        archive_url = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov3/model/yolov3-12.tar.gz"
        filename = "yolov3-12.tar.gz"
        from tvm.contrib import download
        import onnx
        import tarfile
        download.download(archive_url, filename)
        archive = tarfile.open(filename)
        directory = "onnx_yolov3"
        archive.extractall(directory)
        archive.close()
        directory = os.path.join(directory, "yolov3-12")
        model_file = os.path.join(directory, "yolov3-12.onnx")
        onnx_model = onnx.load(model_file)
        shape_dict = {
            "input_1": (1, 3, 416, 416),
            "image_shape": (1, 2),
        }
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
        test_files_dir = os.path.join(directory, "test_data_set_0")

        # downcast to float16
        mod = convert_to_dtype(mod["main"], dtype)
        dtype = "float32" if dtype == "float32" else "float16"
        print("=" * 10)
        print(mod)
        print("=" * 10)

        return (mod, params, shape_dict, dtype, target, ONNXYolov3Validator())
        #return (mod, params, shape_dict, dtype, target, ONNXTestSamplesValidator(test_files_dir, input_names=list(shape_dict.keys())))


    def import_onnx_faster_rcnn(self, target="llvm", dtype="float32"):
        min_shape = 800.0
        def _get_shape():
            from PIL import Image
            from tvm.contrib import download
            # Download test image
            image_url = "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
            image_fn = "dog.png"
            image_url = "https://raw.githubusercontent.com/onnx/models/main/vision/object_detection_segmentation/faster-rcnn/dependencies/demo.jpg"
            image_fn = "demo.png"
            download.download(image_url, image_fn)

            # Prepare test image for inference
            #import ipdb; ipdb.set_trace()
            image = Image.open(image_fn)
            print(image.size)
            ratio = min_shape / min(image.size[0], image.size[1])
            #return (3, int(ratio * image.size[1]), int(ratio * image.size[0])) # [c, h, w]
            return (3, int(min_shape), int(min_shape)) # [c, h, w]
        archive_url = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx"
        filename = "FasterRCNN-12"
        from tvm.contrib import download
        import onnx
        download.download(archive_url, filename)
        onnx_model = onnx.load(filename)
        shape_dict = {
            "image": _get_shape(),
        }
        mod_file = f"onnx_faster_rcnn_mod_{dtype}.json"
        params_file = f"onnx_faster_rcnn_params_{dtype}.json"
        if not os.path.exists(mod_file):
            mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

            # downcast to float16
            mod = convert_to_dtype(mod["main"], dtype)
            with open(mod_file, "w") as file:
                file.write(tvm.ir.save_json(mod))

            with open(params_file, "wb") as file:
                file.write(relay.save_param_dict(params))
        else:
            with open(mod_file, "r") as file:
                mod = tvm.ir.load_json(file.read())

            with open(params_file, "rb") as file:
                params = relay.load_param_dict(file.read())
        dtype = "float32" if dtype == "float32" else "float16"
        print("=" * 10)
        print(mod)
        print("=" * 10)

        return (mod, params, shape_dict, dtype, target, FasterRCNNValidator(min_shape))


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
        default="float16",
        choices=["float32", "float16", "float16_acc32"],
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
        action="store_true",
        help="Use graph runtime debugger to output per layer perf. data and other statistics",
    )
    parser.add_argument(
        "--VM",
        action="store_true",
        help="Use VM compiling and benchmarking",
    )

    args = parser.parse_args()
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
                number=50,
                timeout=15,
                #min_repeat_ms=150,
                #cooldown_interval=150
            ),
        ),
    }
    return args


args = get_args()


def main():
    if "opencl" in args.target:
        executor = Executor(use_tracker="android")
    else:
        executor = Executor()
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
    if "resnet50_v1" in name or "mobilenet1.0" in name or "resnet50_v2" in name or "vgg16" in name:
        model = gluon.model_zoo.vision.get_model(name, pretrained=True)
        data_shape = (batch_size, 3, 224, 224)
    elif "inceptionv3" in name:
        model = gluon.model_zoo.vision.inception_v3(pretrained=True)
        data_shape = (batch_size, 3, 299, 299)
    else:
        raise ValueError("Input shape unknown for gluon model: " + name)
    return model, data_shape


def gluoncv_model(name, batch_size=None):
    from gluoncv import model_zoo
    if "yolo3" in name:
        model = model_zoo.get_model(name, pretrained=True)
        data_shape = (batch_size, 3, 416, 416)
    return model, data_shape

class Validator(object):
    def __init__(self, inputs):
        if isinstance(inputs, dict):
            self.inputs = inputs
        else:
            assert len(inputs) == 1
            self.inputs = {"data" : inputs[0]}
    def GetReference(self):
        return []
    def Validate(self):
        return None
    def GetInputDictionary(self):
        return self.inputs

class ImageNetValidator(Validator):
    def __init__(self, shape_dict, layout="NCHW", preproc=None):
        assert layout in ("NCHW", "NHWC"), "Requested layout is not currently supported: " + layout
        assert len(shape_dict) == 1
        from PIL import Image
        from tvm.contrib import download
        from os.path import join, isfile
        from matplotlib import pyplot as plt

        name = list(shape_dict.keys())[0]

        # Download ImageNet categories
        categ_url = "https://github.com/uwsampl/web-data/raw/main/vta/models/"
        categ_fn = "synset.txt"
        download.download(join(categ_url, categ_fn), categ_fn)
        self.synset = eval(open(categ_fn).read())

        # Download test image
        image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
        image_fn = "cat.png"
        download.download(image_url, image_fn)

        # Prepare test image for inference
        #import ipdb; ipdb.set_trace()
        image = Image.open(image_fn)
        if layout == "NHWC":
            image = image.resize(shape_dict[name][1:-1])
        elif layout == "NCHW":
            image = image.resize(shape_dict[name][2:])

        #image = self.preprocess(np.array(image))
        if "mxnet" in preproc:
            image = np.array(image) - np.array([123.0, 117.0, 104.0])
            image /= np.array([58.395, 57.12, 57.375])
            image = image.transpose((2, 0, 1))
            image = image[np.newaxis, :]
        elif "keras" in preproc:
            image = np.array(image)[np.newaxis, :].astype("float32")
            from tensorflow.keras.applications.inception_v3 import preprocess_input
            image = preprocess_input(image)
        elif "keras_mobilenetv1" in preproc:
            image = np.array(image)[np.newaxis, :].astype("float32")
            from tensorflow.keras.applications.mobilenet import preprocess_input
            image = preprocess_input(image)

        self.inputs = {name : image}

    def Validate(self, m, ref_outputs=[]):
        if isinstance(m, tvm.runtime.vm.VirtualMachine) or isinstance(m, tvm.runtime.profiler_vm.VirtualMachineProfiler):
            tvm_output = m.get_outputs()[0]
        else:
            tvm_output = m.get_output(0)
        #import ipdb; ipdb.set_trace()
        top_categories = np.argsort(tvm_output.asnumpy()[0])
        # Report top-5 classification results
        print("\nTop5 predictions: \n")
        top5 = np.flip(top_categories, axis=0)[:5]
        # print("\t#1:", self.synset[top_categories[-1]])
        # print("\t#2:", self.synset[top_categories[-2]])
        # print("\t#3:", self.synset[top_categories[-3]])
        # print("\t#4:", self.synset[top_categories[-4]])
        # print("\t#5:", self.synset[top_categories[-5]])
        print("\t#1:", self.synset[top5[1-1]])
        print("\t#2:", self.synset[top5[2-1]])
        print("\t#3:", self.synset[top5[3-1]])
        print("\t#4:", self.synset[top5[4-1]])
        print("\t#5:", self.synset[top5[5-1]])
        print("\t", top5)
        ImageNetClassifier = False
        for k in top_categories[-5:]:
            if "cat" in self.synset[k]:
                ImageNetClassifier = True
        assert ImageNetClassifier, "Failed ImageNet classifier validation check"


class VOCValidator(Validator):
    # this function is from yolo3.utils.letterbox_image
    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        from PIL import Image
        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    def preprocess(self, img):
        model_image_size = (416, 416)
        boxed_image = self.letterbox_image(img, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)
        return image_data

    def __init__(self, shape_dict, layout="NCHW", preproc=None):
        assert layout in ("NCHW", "NHWC"), "Requested layout is not currently supported: " + layout
        assert len(shape_dict) == 1
        from PIL import Image
        from tvm.contrib import download
        from os.path import join, isfile
        from matplotlib import pyplot as plt

        name = list(shape_dict.keys())[0]

        # Download test image
        image_url = "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
        image_fn = "dog.png"
        download.download(image_url, image_fn)

        # Prepare test image for inference
        #import ipdb; ipdb.set_trace()
        image = Image.open(image_fn)
        image_data = self.preprocess(image)

        self.inputs = {name : image_data}

    def Validate(self, m, ref_outputs=[]):
        # class_IDs, scores, bounding_boxs
        classid = m.get_output(0)
        scores = m.get_output(1)
        bounding_boxs = m.get_output(2)
        for a in classid:
            print(a)

class Deeplabv3Validator(Validator):
    def __init__(self, input_shape, dtype):
        from os.path import join
        from tvm.contrib import download
        assert isinstance(input_shape, dict)
        assert dtype in ["float16", "float32"]
        np.random.seed(1)
        self.dtype = dtype
        self.inputs = {}
        for key in input_shape:
            self.inputs[key] = np.random.normal(size=input_shape[key]).astype("float32")

        categ_url = "https://github.com/Deelvin/qualcomm/raw/avoronov/rebase_master_v2/"
        categ_fn = "deeplabv3_reference_output_{}".format(dtype)
        download.download(join(categ_url, categ_fn), categ_fn)
        # genered by target="llvm -keys=cpu" at np.random.seed(1)
        self.ref_outputs = eval(open(categ_fn).read())

    def GetReference(self):
        return self.ref_outputs

    def Validate(self, m, ref_outputs=[]):
        if self.dtype == "float16":
            rtol=1e-1
            atol=1e-1
        if self.dtype == "float32":
            rtol=1e-3
            atol=1e-3
        if isinstance(m, tvm.runtime.vm.VirtualMachine) or isinstance(m, tvm.runtime.profiler_vm.VirtualMachineProfiler):
            outputs = m.get_outputs()
            for i in range(len(outputs)):
                tvm_output = outputs[i]
                np.testing.assert_allclose(tvm_output.asnumpy(), ref_outputs[i], rtol=rtol, atol=atol)
            print("Deeplabv3Validator pass:", "rtol", rtol, "atol",atol)
        else:
            for i in range(m.get_num_outputs()):
                tvm_output = m.get_output(i)
                np.testing.assert_allclose(tvm_output.asnumpy(), ref_outputs[i], rtol=rtol, atol=atol)
            print("Deeplabv3Validator pass:", "rtol", rtol, "atol",atol)

class Yolov3Validator(Validator):
    class BoundBox:
        def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
            self.xmin = xmin
            self.ymin = ymin
            self.xmax = xmax
            self.ymax = ymax
            self.objness = objness
            self.classes = classes
            self.label = -1
            self.score = -1

        def get_label(self):
            if self.label == -1:
                self.label = np.argmax(self.classes)

            return self.label

        def get_score(self):
            if self.score == -1:
                self.score = self.classes[self.get_label()]

            return self.score

    def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5
        boxes = []

        def _sigmoid(x):
            return 1. / (1. + np.exp(-x))

        netout[..., :2]  = _sigmoid(netout[..., :2])
        netout[..., 4:]  = _sigmoid(netout[..., 4:])
        netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for i in range(grid_h*grid_w):
            row = i / grid_w
            col = i % grid_w
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[int(row)][int(col)][b][4]
                if(objectness.all() <= obj_thresh): continue
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]
                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
                # last elements are class probabilities
                classes = netout[int(row)][col][b][5:]
                box = Yolov3Validator.BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
                boxes.append(box)
        return boxes

    def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
        new_w, new_h = net_w, net_h
        for i in range(len(boxes)):
            x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
            y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    def bbox_iou(box1, box2):
        def _interval_overlap(interval_a, interval_b):
            x1, x2 = interval_a
            x3, x4 = interval_b
            if x3 < x1:
                if x4 < x1:
                    return 0
                else:
                    return min(x2,x4) - x1
            else:
                if x2 < x3:
                    return 0
                else:
                    return min(x2,x4) - x3
        intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        union = w1*h1 + w2*h2 - intersect
        return float(intersect) / union

    def do_nms(boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return
        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])
            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                if boxes[index_i].classes[c] == 0: continue
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if Yolov3Validator.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0

    # load and prepare an image
    @staticmethod
    def load_image_pixels(filename, shape):
        try:
            from keras.preprocessing.image import load_img
            from keras.preprocessing.image import img_to_array
        except:
            from tensorflow.keras.utils import load_img
            from tensorflow.keras.utils import img_to_array
        # load the image to get its shape
        image = load_img(filename)
        width, height = image.size
        # load the image with the required size
        image = load_img(filename, target_size=shape)
        # convert to numpy array
        image = img_to_array(image)
        # scale pixel values to [0, 1]
        image = image.astype('float32')
        image /= 255.0
        # add a dimension so that we have one sample
        image = np.expand_dims(image, 0)
        return image, width, height

    # get all of the results above a threshold
    @staticmethod
    def get_boxes(boxes, labels, thresh):
        v_boxes, v_labels, v_scores = list(), list(), list()
        # enumerate all boxes
        for box in boxes:
            # enumerate all possible labels
            for i in range(len(labels)):
                # check if the threshold for this label is high enough
                if box.classes[i] > thresh:
                    v_boxes.append(box)
                    v_labels.append(labels[i])
                    v_scores.append(box.classes[i]*100)
                    # don't break, many labels may trigger for one box
        return v_boxes, v_labels, v_scores

    # draw all results
    @staticmethod
    def draw_boxes(filename, v_boxes, v_labels, v_scores):
        from matplotlib import pyplot
        from matplotlib.patches import Rectangle
        # load the image
        from PIL import Image
        if ".png" not in filename:
            name, extension = filename.rsplit('.', 1)
            im1 = Image.open(filename)
            filename = "{}.png".format(name)
            im1.save(filename)

        data = pyplot.imread(filename)
        # plot the image
        pyplot.imshow(data)
        # get the context for drawing boxes
        ax = pyplot.gca()
        # plot each box
        for i in range(len(v_boxes)):
            box = v_boxes[i]
            # get coordinates
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='white')
            # draw the box
            ax.add_patch(rect)
            # draw text and score in top left corner
            label = "%s (%.3f)" % (v_labels[i], v_scores[i])
            pyplot.text(x1, y1, label, color='white')
        # show the plot
        pyplot.show()

    def __init__(self, input_shape, dtype="float32"):
        from tvm.contrib import download
        from os.path import join
        n, h, w, c = list(input_shape.values())[0]
        self.input_w, self.input_h = h, w

        # Download Coco names
        names_url = "https://github.com/pjreddie/darknet/raw/master/data/"
        names_fn = "coco.names"
        download.download(join(names_url, names_fn), names_fn, overwrite=True)
        self.labels = [line.rstrip() for line in open(names_fn).readlines()]

        # Download test image
        image_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"
        self.image_fn = "dog.jpg"
        download.download(image_url, self.image_fn)

        # # load and prepare image
        image, image_w, image_h = Yolov3Validator.load_image_pixels(self.image_fn, (self.input_w, self.input_h))
        self.image_w = image_w
        self.image_h = image_h
        self.image = image
        self.inputs = { list(input_shape.keys())[0]: image }

class ONNXTestSamplesValidator(Validator):
    def __init__(self, test_data_dir, input_names, dtype="float32"):
        import onnx
        import glob
        from onnx import numpy_helper

        self.test_data_dir = test_data_dir
        inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
        self.inputs = {}
        for i in range(inputs_num):
            input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
            tensor = onnx.TensorProto()
            with open(input_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            inp = numpy_helper.to_array(tensor)
            self.inputs[input_names[i]] = inp

    def Validate(self, m, ref_outputs=[], show=False):
        import onnx
        import glob
        from onnx import numpy_helper
        # output
        if isinstance(m, tvm.runtime.vm.VirtualMachine) or isinstance(m, tvm.runtime.profiler_vm.VirtualMachineProfiler):
            outputs = []
            tmp = m.get_outputs()
            for i in range(len(tmp)):
                tvm_output = tmp[i]
                outputs.append(tvm_output.asnumpy())
        else:
            num_outputs = m.get_num_outputs()
            outputs = []
            for i in range(num_outputs):
                tvm_output = m.get_output(i)
                outputs.append(tvm_output.asnumpy())
        refs = []
        inputs_num = len(glob.glob(os.path.join(self.test_data_dir, 'output_*.pb')))
        self.inputs = {}
        for i in range(inputs_num):
            input_file = os.path.join(self.test_data_dir, 'output_{}.pb'.format(i))
            tensor = onnx.TensorProto()
            with open(input_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            refs.append(numpy_helper.to_array(tensor))
        #labels = []
        #scores_list = []
        #boxes_list = []
        #from tvm.contrib import download
        #classes_url = "https://raw.githubusercontent.com/qqwweee/keras-yolo3/master/model_data/coco_classes.txt"
        #classes_fn = "coco_classes.txt"
        #download.download(classes_url, classes_fn)
        #classes = [line.rstrip('\n') for line in open(classes_fn)]
        #for idx_ in outputs[2]:
        #    class_idx = idx_[1]
        #    score = outputs[1][tuple(idx_)]
        #    idx_1 = (idx_[0], idx_[2])
        #    box = outputs[0][idx_1]
        #    labels.append(classes[class_idx])
        #    scores_list.append(score)
        #    boxes_list.append(box)
        #    print("bigger: label: {}, score: {}, box: {}".format(classes[class_idx], score, box))

        #print(outputs[0].shape)
        #print(outputs[1].shape)
        #print(outputs[2].shape)
        for i in range(len(outputs)):
            np.testing.assert_allclose(outputs[i], refs[i], rtol=1e-2, atol=1e-2)


class FasterRCNNValidator(Validator):
    def preprocess(self, image):
        from PIL import Image
        # Resize
        ratio = self.min_shape / min(image.size[0], image.size[1])
        #image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)
        image = image.resize((int(self.min_shape), int(self.min_shape)), Image.BILINEAR)

        # Convert to BGR
        image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

        # HWC -> CHW
        image = np.transpose(image, [2, 0, 1])

        # Normalize
        mean_vec = np.array([102.9801, 115.9465, 122.7717])
        for i in range(image.shape[0]):
            image[i, :, :] = image[i, :, :] - mean_vec[i]

        # Pad to be divisible of 32
        import math
        padded_h = int(math.ceil(image.shape[1] / 32) * 32)
        padded_w = int(math.ceil(image.shape[2] / 32) * 32)

        padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
        padded_image[:, :image.shape[1], :image.shape[2]] = image
        image = padded_image

        return image

    def __init__(self, min_shape, preproc=None):
        from PIL import Image
        from tvm.contrib import download
        from os.path import join, isfile
        from matplotlib import pyplot as plt
        self.min_shape = min_shape

        # Download test image
        image_url = "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
        image_fn = "dog.png"
        download.download(image_url, image_fn)

        # Prepare test image for inference
        #import ipdb; ipdb.set_trace()
        self.image = Image.open(image_fn)
        image_data = self.preprocess(self.image)

        self.inputs = {"image" : image_data}

    def Validate(self, m, ref_outputs=[]):
        from tvm.contrib import download
        classes_url = "https://raw.githubusercontent.com/onnx/models/main/vision/object_detection_segmentation/faster-rcnn/dependencies/coco_classes.txt"
        classes_fn = "coco_classes_faster.txt"
        download.download(classes_url, classes_fn)
        classes = [line.rstrip('\n') for line in open(classes_fn)]

        # class_IDs, scores, bounding_boxs
        if isinstance(m, tvm.runtime.vm.VirtualMachine) or isinstance(m, tvm.runtime.profiler_vm.VirtualMachineProfiler):
            tvm_output = m.get_outputs()
            boxes = tvm_output[0].asnumpy()
            labels = tvm_output[1].asnumpy()
            scores = tvm_output[2].asnumpy()
        else:
            boxes = m.get_output(0).asnumpy()
            labels = m.get_output(1).asnumpy()
            scores = m.get_output(2).asnumpy()
        score_threshold = 0.7
        assert boxes.shape[0] == labels.shape[0] and labels.shape[0] == scores.shape[0]
        #for box, label, score in zip(boxes, labels, scores):
        for i in range(len(boxes)):
            if scores[i] > score_threshold:
                print("label: {}, score: {}, box: {}".format(classes[labels[i]], scores[i], boxes[i]))
                #assert classes[labels[i]] == 'dog' or classes[labels[i]] == 'bicycle'


class SSDResnetValidator(Validator):
    def preprocess(self, image):
        from PIL import Image
        img = image.resize((1200, 1200), Image.BILINEAR)
        img_data = np.array(img)
        img_data = np.transpose(img_data, [2, 0, 1])
        img_data = np.expand_dims(img_data, 0)
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[1]):
            norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        return norm_img_data

    def __init__(self, preproc=None):
        from PIL import Image
        from tvm.contrib import download
        from os.path import join, isfile
        from matplotlib import pyplot as plt

        # Download test image
        image_url = "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
        image_fn = "dog.png"
        download.download(image_url, image_fn)

        # Prepare test image for inference
        #import ipdb; ipdb.set_trace()
        self.image = Image.open(image_fn)
        image_data = self.preprocess(self.image)

        self.inputs = {"image" : image_data}

    def Validate(self, m, ref_outputs=[]):
        from tvm.contrib import download
        classes_url = "https://raw.githubusercontent.com/onnx/models/main/vision/object_detection_segmentation/faster-rcnn/dependencies/coco_classes.txt"
        classes_fn = "coco_classes_resnet.txt"
        download.download(classes_url, classes_fn)
        classes = [line.rstrip('\n') for line in open(classes_fn)]

        # class_IDs, scores, bounding_boxs
        if isinstance(m, tvm.runtime.vm.VirtualMachine) or isinstance(m, tvm.runtime.profiler_vm.VirtualMachineProfiler):
            tvm_output = m.get_outputs()
            boxes = tvm_output[0].asnumpy()
            labels = tvm_output[1].asnumpy()
            scores = tvm_output[2].asnumpy()
        else:
            boxes = m.get_output(0).asnumpy()
            labels = m.get_output(1).asnumpy()
            scores = m.get_output(2).asnumpy()
        score_threshold = 0.7
        for i in range(len(boxes)):
            if scores[0][i] > score_threshold:
                print("label: {}, score: {}, box: {}".format(classes[labels[0][i]], scores[0][i], boxes[0][i]))
                assert classes[labels[0][i]] == 'dog' or classes[labels[0][i]] == 'bicycle'


class ONNXYolov3Validator(Validator):
    def preprocess(self, img):
        #from PIL import Image
        def _letterbox_image(image, size):
            from PIL import Image
            '''resize image with unchanged aspect ratio using padding'''
            iw, ih = image.size
            w, h = size
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
            return new_image

        model_image_size = (416, 416)
        boxed_image = _letterbox_image(img, tuple(reversed(model_image_size)))
        #boxed_image = img.resize(model_image_size, Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)
        return image_data

    def __init__(self, preproc=None):
        from PIL import Image
        from tvm.contrib import download
        from os.path import join, isfile
        from matplotlib import pyplot as plt

        # Download test image
        image_url = "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
        image_fn = "dog.png"
        download.download(image_url, image_fn)

        # Prepare test image for inference
        #import ipdb; ipdb.set_trace()
        self.image = Image.open(image_fn)
        image_data = self.preprocess(self.image)
        image_size = np.array([self.image.size[1], self.image.size[0]], dtype="float32").reshape(1, 2)

        self.inputs = {
            "input_1" : image_data,
            "image_shape" : image_size,
        }

    def Validate(self, m, ref_outputs=[]):
        from tvm.contrib import download
        classes_url = "https://raw.githubusercontent.com/qqwweee/keras-yolo3/master/model_data/coco_classes.txt"
        #classes_url = "https://raw.githubusercontent.com/onnx/models/main/vision/object_detection_segmentation/faster-rcnn/dependencies/coco_classes.txt"
        classes_fn = "coco_classes.txt"
        download.download(classes_url, classes_fn)
        classes = [line.rstrip('\n') for line in open(classes_fn)]

        # class_IDs, scores, bounding_boxs
        if isinstance(m, tvm.runtime.vm.VirtualMachine) or isinstance(m, tvm.runtime.profiler_vm.VirtualMachineProfiler):
            tvm_output = m.get_outputs()
            boxes = tvm_output[0].asnumpy()
            scores = tvm_output[1].asnumpy()
            indices = tvm_output[2].asnumpy()
        else:
            boxes = m.get_output(0).asnumpy()
            scores = m.get_output(1).asnumpy()
            indices = m.get_output(2).asnumpy()
        score_threshold = 0.7
        print(boxes.shape)
        print(scores.shape)
        print(indices.shape)
        for idx_ in indices:
            class_idx = idx_[1]
            score = scores[tuple(idx_)]
            idx_1 = (idx_[0], idx_[2])
            box = boxes[idx_1]
            possible_objs = ['dog', 'bicycle', 'truck']
            if score > score_threshold and score <= 100:
                print("label: {}, score: {}, box: {}".format(classes[class_idx], score, box))
                assert classes[class_idx] in possible_objs


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
            args.rpc_key, priority=0, session_timeout=6000
        )
        print("Tracker connected to remote RPC server")

    def _disconnect_tracker(self):
        self.remote = None
        self.tracker = None

    def check_distribution(self, y, tolerance=0.05, show_plot=False):
        import warnings
        from sklearn.linear_model import LinearRegression

        num_samples = len(y)
        x = np.array(list(range(num_samples))).reshape((-1, 1))

        model = LinearRegression(fit_intercept=True, copy_X=True)
        model.fit(x, y)
        print("intercept (b0):", model.intercept_)
        print("slope (b1):", model.coef_)
        print("coefficient of determination:", model.score(x, y))
        if (model.score(x, y) >= tolerance):
            warnings.warn("The coefficient of determination is higher than the acceptable coefficient of determination, use cooling of the device.", UserWarning)
            show_plot = True

        if show_plot:
            import matplotlib.pyplot as plt
            plt.plot(y)
            plt.plot(model.predict(x))
            plt.show()

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
            from tvm.contrib.debugger import debug_runtime as graph_executor
        else:
            from tvm.contrib import graph_executor

        if self.use_tracker and self.remote == None:
            self._connect_tracker()

        with relay.build_config(opt_level=3):
            # lib2 = relay.build(tvm_mod, target=target, target_host=target_host, params=params)
            # lib2.export_library("_model.so", ndk.create_shared)
            graph, lib, params = relay.build(
                tvm_mod, target_host=target_host, target=target, params=params
            )
            # print("JSON:\n", graph)

        if self.remote:
            print("Using Android OpenCL runtime over RPC")
            temp = utils.tempdir()
            dso_binary = "dev_lib_cl.so"
            dso_binary_path = temp.relpath(dso_binary)
            if "opencl" in target:
                ctx = self.remote.cl(0)
            else:
                ctx = self.remote.cpu(0)
            lib.export_library(dso_binary_path, fcompile=ndk.create_shared)
            remote_path = "/data/local/tmp/" + dso_binary
            self.remote.upload(dso_binary_path)
            print("Uploading binary...")
            rlib = self.remote.load_module(dso_binary)
            m = graph_executor.create(graph, rlib, ctx)
        else:
            print("Using local runtime")
            ctx = tvm.device(target, 0)
            m = graph_executor.create(graph, lib, ctx)

        m.set_input(**params)
        inputs = []
        if isinstance(validator, Validator):
            inputs = validator.GetInputDictionary()
            for key, data in inputs.items():
                m.set_input(key, data)
        elif isinstance(input_shape, dict):
            for key in input_shape:
                inputs.append(np.random.normal(size=input_shape[key]).astype(dtype))
                m.set_input(key, inputs[-1])
        else:
            inputs.append(np.random.normal(size=input_shape).astype(dtype))
            m.set_input("data", inputs[-1])

        print("Evaluating...", flush=True)
        number = 1
        repeat = 100
        min_repeat_ms = 0
        time_to_work_ms = 1000
        cooldown_interval_ms=1000
        if args.debug:
            m.run()
            time_f = advanced_time_evaluator(m, "run", ctx, number, repeat, min_repeat_ms, time_to_work_ms, cooldown_interval_ms)
        else:
            time_f = advanced_time_evaluator(m, "run", ctx, number, repeat, min_repeat_ms, time_to_work_ms, cooldown_interval_ms)

        benchmarkResult = time_f()
        cost = benchmarkResult.mean
        print("%g secs/iteration\n" % cost)
        results = benchmarkResult.results
        self.check_distribution(results)
        print(benchmarkResult)

        if validator:
            if isinstance(validator, Validator):
                ref_outputs = validator.GetReference()
                validator.Validate(m, ref_outputs)
            else:
                ref_outputs = validator(inputs)
                for i, ref_output in enumerate(ref_outputs):
                    tvm_output = m.get_output(i)
                    output = tvm_output.asnumpy()
                    np.testing.assert_allclose(output, ref_output, rtol=1e-3, atol=1e-3)
            print("Validation done")


    def _benchmark_vm(
        self,
        tvm_mod,
        params,
        input_shape,
        target="llvm",
        target_host="llvm",
        dtype="float32",
        validator=None
    ):
        from tvm.runtime.vm import VirtualMachine
        from tvm.runtime import profiler_vm

        if self.use_tracker and self.remote == None:
            self._connect_tracker()

        if isinstance(tvm_mod, tvm.IRModule):
            mod = tvm_mod
        else:
            mod = tvm.IRModule()
            mod["main"] = tvm_mod

        with tvm.transform.PassContext(opt_level=3):
            vmc = relay.vm.compile(mod, target_host=target_host, target=target, params=params)

        if self.remote:
            print("Using Android OpenCL runtime over RPC")
            temp = utils.tempdir()
            dso_binary = "dev_lib_cl.so"
            dso_binary_path = temp.relpath(dso_binary)
            if "opencl" in target:
                ctx = self.remote.cl(0)
            else:
                ctx = self.remote.cpu(0)
            vmc.mod.export_library(dso_binary_path, fcompile=ndk.create_shared)
            self.remote.upload(dso_binary_path)
            print("Uploading binary...")
            rlib = self.remote.load_module(dso_binary)
        else:
            print("Using local runtime")
            ctx = tvm.device(target, 0)
            rlib = vmc

        vm = VirtualMachine(rlib, ctx, "naive")

        inputs = []
        if isinstance(validator, Validator):
            inputs = validator.GetInputDictionary()
            data = {}
            for k, v in inputs.items():
                data[k] = tvm.nd.array(v, ctx)
            vm.set_input("main", **data)
        elif isinstance(input_shape, dict):
            data = {}
            for key in input_shape:
                data[key] = tvm.nd.array(np.random.normal(size=input_shape[key]).astype("float32"), ctx)
            vm.set_input("main", **data)
        else:
            data = tvm.nd.array(np.random.normal(size=input_shape).astype("float32"), ctx)
            vm.set_input("main", data)

        print("Evaluating...", flush=True)

        number = 1
        repeat = 100
        min_repeat_ms = 0
        time_to_work_ms = 1000
        cooldown_interval_ms=1000
        time_f = advanced_time_evaluator(vm, "invoke_stateful", ctx, number, repeat, min_repeat_ms, time_to_work_ms, cooldown_interval_ms, mod_func_name="main")

        benchmarkResult = time_f("main")
        cost = benchmarkResult.mean
        print("%g secs/iteration\n" % cost)
        print(benchmarkResult)

        if validator:
            if isinstance(validator, Validator):
                ref_outputs = validator.GetReference()
                validator.Validate(vm, ref_outputs)
            else:
                ref_outputs = validator(inputs)
                for i, ref_output in enumerate(ref_outputs):
                    tvm_output = vm.get_outputs(i)
                    output = tvm_output.asnumpy()
                    np.testing.assert_allclose(output, ref_output, rtol=1e-3, atol=1e-3)
            print("Validation done")

        if args.debug:
            vm = tvm.runtime.profiler_vm.VirtualMachineProfiler(rlib, ctx, "naive")
            res = vm.profile(**data, func_name="main")
            print(res)


    def _schedule_jobs(self, mod, params, input_shape, dtype, target, validator=None):
        if args.VM:
            def bench():
                self._benchmark_vm(
                    mod,
                    params,
                    input_shape,
                    target=target,
                    target_host=self.host_target,
                    dtype=dtype,
                    validator=validator
                )
        else:
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
            if apply_previous_tune == False:
                print("Extracting tasks")
                tasks = autotvm.task.extract_from_program(
                    mod, target=target, target_host=self.host_target, params=params
                )
                print("Tuning kernels")
                Executor.tune_tasks(tasks, **options)

            def tuned_benchmark():
                print("Apply best performing tuning profiles:")

                if (options["log_filename"]):
                    with autotvm.apply_history_best(options["log_filename"]):
                        bench()
                else:
                    bench()

            self.benchmarks.pop(benchmark_index)
            self.benchmarks.append(tuned_benchmark)

        self.tuning_jobs.append(tune)

    @staticmethod
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

if __name__ == "__main__":
    main()
