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
        shape_dict = {"data": input_shape}
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        # layout transformation
        if "adreno" in target:
            layout_config = relay.transform.LayoutConfig(skip_layers=[0])
            desired_layouts = {"nn.conv2d": ["NCHW4c", "OIHW4o"]}
            with layout_config:
                seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
                with tvm.transform.PassContext(opt_level=3):
                    mod = seq(mod)
        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, preproc="mxnet"))

    def import_resnet50_v2(self, target="llvm", dtype="float32"):
        model, input_shape = gluon_model("resnet50_v2", batch_size=1)
        shape_dict = {"data": input_shape}
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        # layout transformation
        if "adreno" in target:
            layout_config = relay.transform.LayoutConfig(skip_layers=[0])
            desired_layouts = {"nn.conv2d": ["NCHW4c", "OIHW4o"]}
            with layout_config:
                seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
                with tvm.transform.PassContext(opt_level=3):
                    mod = seq(mod)
        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, preproc="mxnet"))

    def import_mobilenetv1(self, target="llvm", dtype="float32"):
        model, input_shape = gluon_model("mobilenet1.0", batch_size=1)
        shape_dict = {"data": input_shape}
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        # layout transformation
        if "adreno" in target:
            layout_config = relay.transform.LayoutConfig(skip_layers=[0])
            desired_layouts = {"nn.conv2d": ["NCHW4c", "OIHW4o"]}
            with layout_config:
                seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
                with tvm.transform.PassContext(opt_level=3):
                    mod = seq(mod)
        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)

        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, preproc="mxnet"))

    def import_vgg16(self, target="llvm", dtype="float32"):
        model, input_shape = gluon_model("vgg16", batch_size=1)
        shape_dict = {"data": input_shape}
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        # layout transformation
        if "adreno" in target:
            layout_config = relay.transform.LayoutConfig(skip_layers=[0])
            desired_layouts = {"nn.conv2d": ["NCHW4c", "OIHW4o"]}
            with layout_config:
                seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
                with tvm.transform.PassContext(opt_level=3):
                    mod = seq(mod)
            mod = relay.quantize.prerequisite_optimize(mod, params)
        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, preproc="mxnet"))

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
        mod, params = relay.frontend.from_onnx(model, input_shape, opset=11, freeze_params=True)
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        from tvm.relay import transform
        mod = transform.DynamicToStatic()(mod)
        return (mod, params, input_shape, dtype, target)

    def import_deeplabv3(self, target="llvm", dtype="float32"):
        import tensorflow as tf
        try:
            tf_compat_v1 = tf.compat.v1
        except ImportError:
            tf_compat_v1 = tf
        # Tensorflow utility functions
        import tvm.relay.testing.tf as tf_testing

        model_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/../models/mace_deeplabv3/deeplab-v3-plus-mobilenet-v2.pb"
        )

        with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            #graph = tf.import_graph_def(graph_def, name="")
            # Call the utility to import the graph definition into default graph.
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        input_shape = {"sub_7": (1, 513, 513, 3)}
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape)

        from tvm.relay import transform
        #mod = transform.DynamicToStatic()(mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
            mod = relay.quantize.prerequisite_optimize(mod, params)

        # layout transformation
        if "adreno" in target:
            layout_config = relay.transform.LayoutConfig(skip_layers=[0, 54])
            desired_layouts = {"nn.conv2d": ["NCHW4c", "OIHW4o"]}
            with layout_config:
                seq = tvm.transform.Sequential([
                    relay.transform.SimplifyExpr(),
                    relay.transform.ConvertLayout(desired_layouts)
                    ])
                with tvm.transform.PassContext(opt_level=3):
                    mod = seq(mod)
            mod = relay.quantize.prerequisite_optimize(mod, params)
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

        if "adreno" in target:
            # layout transformation
            layout_config = relay.transform.LayoutConfig(skip_layers=[0])
            desired_layouts = {"nn.conv2d": ["NCHW4c", "OIHW4o"]}
            with layout_config:
                seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
                with tvm.transform.PassContext(opt_level=3):
                    mod = seq(mod)
            mod = relay.quantize.prerequisite_optimize(mod, params)
        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)


        #return (mod, params, shape_dict, dtype, target)
        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, "NHWC", preproc="keras"))


    def import_yolov3(self, target="llvm", dtype="float32"):
        model_url = "http://cnbj1.fds.api.xiaomi.com/mace/miai-models/yolo-v3/yolo-v3.pb"
        model_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/../models/mace_yolov3/yolo-v3.pb"
        )

        from tvm.contrib import download
        download.download(model_url, model_path)

        import tensorflow as tf
        try:
            tf_compat_v1 = tf.compat.v1
        except ImportError:
            tf_compat_v1 = tf
        # Tensorflow utility functions
        import tvm.relay.testing.tf as tf_testing

        with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            #graph = tf.import_graph_def(graph_def, name="")
            # Call the utility to import the graph definition into default graph.
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        input_shape = {"input_1": (1, 416, 416, 3)}
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape,
                                        outputs=["conv2d_59/BiasAdd","conv2d_67/BiasAdd","conv2d_75/BiasAdd"])

        from tvm.relay import transform
        #mod = transform.DynamicToStatic()(mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
            mod = relay.quantize.prerequisite_optimize(mod, params)
        
        # layout transformation
        if "adreno" in target:
            #layout_config = relay.transform.LayoutConfig(skip_layers=[0,58,66,74])
            layout_config = relay.transform.LayoutConfig(skip_layers=[0,58,66,74])
            desired_layouts = {"nn.conv2d": ["NCHW4c", "OIHW4o"]}
            with layout_config:
                seq = tvm.transform.Sequential([
                    relay.transform.SimplifyExpr(),
                    relay.transform.ConvertLayout(desired_layouts)
                    ])
                with tvm.transform.PassContext(opt_level=3):
                    mod = seq(mod)
            mod = relay.quantize.prerequisite_optimize(mod, params)
        #print(mod)
        return (mod, params, input_shape, dtype, target)

    def import_yolov3_mxnet(self, target="llvm", dtype="float32"):
        model, input_shape = gluoncv_model("yolo3_darknet53_voc", batch_size=1)
        shape_dict = {"data": input_shape}
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        # layout transformation
        if "adreno" in target:
            skip_layers=[0,58,66,74]
            layout_config = relay.transform.LayoutConfig(skip_layers=skip_layers)
            desired_layouts = {"nn.conv2d": ["NCHW4c", "OIHW4o"]}
            with layout_config:
                seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
                with tvm.transform.PassContext(opt_level=3):
                    mod = seq(mod)
            mod = relay.quantize.prerequisite_optimize(mod, params)
        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        print(mod)
        return (mod, params, shape_dict, dtype, target, VOCValidator(shape_dict, preproc="gluoncv"))

    def import_depthwise_conv2d(self, target="llvm", dtype="float32"):
        input_shape = (1, 16, 112, 112, 4)
        filter_shape = (16, 1, 3, 3, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.var("weight", shape=filter_shape, dtype=dtype)
        C = relay.nn.conv2d(A, B, strides=(2,2), padding=(1,1,1,1), groups=64, data_layout="NCHW4c", kernel_layout="OIHW4o", out_dtype=dtype)
        mod = relay.Function([A, B], C)
        #mod, params = relay.testing.init.create_workload(func)
        np.random.seed(0)
        initializer = relay.testing.init.Xavier()
        filter_data = np.zeros(filter_shape).astype(dtype)
        initializer("weight", filter_data)
        params = {
            "weight": tvm.nd.array(filter_data),
        }

        def validator(inputs):
            vec_length = input_shape[-1]
            # nchwc -> nchw
            data = inputs[0].transpose((0, 1, 4, 2, 3)).reshape(inputs[0].shape[0], inputs[0].shape[1]*inputs[0].shape[-1], inputs[0].shape[2], inputs[0].shape[3])
            data = data.astype("float32")
            # kcrsk -> kcrs
            w_np = params["weight"].asnumpy()
            kernel = w_np.transpose((0, 4, 1, 2, 3)).reshape(w_np.shape[0] * w_np.shape[4], w_np.shape[1], w_np.shape[2], w_np.shape[3])
            np_result = testing.conv2d_nchw_python(data, kernel, stride=(2,2), padding=(1,1,1,1), groups=64)
            # nkhw -> nkhwk
            np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(0, 1, 3, 4, 2)
            return [np_result,]

        return (mod, params, {"data": input_shape}, dtype, target, validator)

    def import_conv2d_nchw(self, target="llvm", dtype="float32"):
        input_shape = (1, 128, 112, 112)
        filter_shape = (128, 128, 3, 3)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.var("weight", shape=filter_shape, dtype=dtype)
        #C = relay.nn.relu(A)
        C = relay.nn.conv2d(A, B, data_layout="NCHW", kernel_layout="OIHW", out_dtype=dtype, channels=128, kernel_size=(3,3))
        mod = relay.Function([A, B], C)
        # mod, params = relay.testing.init.create_workload(func)
        np.random.seed(0)
        initializer = relay.testing.init.Xavier()
        filter_data = np.zeros(filter_shape).astype(dtype)
        initializer("weight", filter_data)
        params = {
            "weight": tvm.nd.array(filter_data),
        }

        # def validator(inputs):
        #     vec_length = input_shape[-1]
        #     # nchwc -> nchw
        #     data = inputs[0]
        #     # convert reference to float32 for use in testing api which only supports float32 activations
        #     data = data.astype("float32")
        #     # kcrsk -> kcrs
        #     w_np = params["weight"].asnumpy()
        #     kernel = w_np
        #     np_result = testing.conv2d_nchw_python(data, kernel, 1, 0)
        #     return [np_result,]

        return (mod, params, {"data": input_shape}, dtype, target)

    def import_conv2d(self, target="llvm", dtype="float32"):
        input_shape = (1, 32, 112, 112, 4)
        filter_shape = (32, 128, 3, 3, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.var("weight", shape=filter_shape, dtype=dtype)
        C = relay.nn.conv2d(A, B, data_layout="NCHW4c", kernel_layout="OIHW4o", out_dtype=dtype)
        mod = relay.Function([A, B], C)
        #mod, params = relay.testing.init.create_workload(func)
        np.random.seed(0)
        initializer = relay.testing.init.Xavier()
        filter_data = np.zeros(filter_shape).astype(dtype)
        initializer("weight", filter_data)
        params = {
            "weight": tvm.nd.array(filter_data),
        }

        def validator(inputs):
            vec_length = input_shape[-1]
            # nchwc -> nchw
            data = inputs[0].transpose((0, 1, 4, 2, 3)).reshape(inputs[0].shape[0], inputs[0].shape[1]*inputs[0].shape[-1], inputs[0].shape[2], inputs[0].shape[3])
            # convert reference to float32 for use in testing api which only supports float32 activations
            data = data.astype("float32")
            # kcrsk -> kcrs
            w_np = params["weight"].asnumpy()
            kernel = w_np.transpose((0, 4, 1, 2, 3)).reshape(w_np.shape[0] * w_np.shape[4], w_np.shape[1], w_np.shape[2], w_np.shape[3])
            np_result = testing.conv2d_nchw_python(data, kernel, 1, 0)
            # nkhw -> nkhwk
            np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(0, 1, 3, 4, 2)
            return [np_result,]

        return (mod, params, {"data": input_shape}, dtype, target, validator)

    def import_conv2d_3x3(self, target="llvm", dtype="float32"):
        input_shape, filter_shape = (1, 128, 7, 7, 4), (128, 512, 3, 3, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.var("weight", shape=filter_shape, dtype=dtype)
        C = relay.nn.conv2d(A, B, data_layout="NCHW4c", kernel_layout="OIHW4o", padding=[1,1,1,1], out_dtype=dtype)
        mod = relay.Function([A, B], C)
        #mod, params = relay.testing.init.create_workload(func)
        np.random.seed(0)
        initializer = relay.testing.init.Xavier()
        filter_data = np.zeros(filter_shape).astype(dtype)
        initializer("weight", filter_data)
        params = {
            "weight": tvm.nd.array(filter_data),
        }

        def validator(inputs):
            vec_length = input_shape[-1]
            # nchwc -> nchw
            data = inputs[0].transpose((0, 1, 4, 2, 3)).reshape(inputs[0].shape[0], inputs[0].shape[1]*inputs[0].shape[-1], inputs[0].shape[2], inputs[0].shape[3])
            data = data.astype("float32")
            # kcrsk -> kcrs
            w_np = params["weight"].asnumpy()
            kernel = w_np.transpose((0, 4, 1, 2, 3)).reshape(w_np.shape[0] * w_np.shape[4], w_np.shape[1], w_np.shape[2], w_np.shape[3])
            np_result = testing.conv2d_nchw_python(data, kernel, padding=[1,1,1,1], stride=[1,1])
            # nkhw -> nkhwk
            np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(0, 1, 3, 4, 2)
            return [np_result,]
        return (mod, params, {"data": input_shape}, dtype, target, validator)
        #return (mod, params, {"data": input_shape}, dtype, target)



    def import_conv2d_conv2d(self, target="llvm", dtype="float32"):
        # input_shape = (1, 128, 112, 112)
        # filter_shape = (128, 128, 3, 3)
        input_shape = (1, 32, 112, 112, 4)
        filter_shape = (32, 128, 3, 3, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B1 = relay.var("weight1", shape=filter_shape, dtype=dtype)
        B2 = relay.var("weight2", shape=filter_shape, dtype=dtype)
        C = relay.nn.conv2d(A, B1, data_layout="NCHW4c", kernel_layout="OIHW4o")
        D = relay.nn.conv2d(C, B2, data_layout="NCHW4c", kernel_layout="OIHW4o")
        mod = relay.Function([A, B1, B2], D)

        params = {
            "weight1": tvm.nd.array(np.random.uniform(-1, 1, filter_shape).astype(dtype)),
            "weight2": tvm.nd.array(np.random.uniform(-1, 1, filter_shape).astype(dtype)),
        }
        def validator(inputs):
            vec_length = input_shape[-1]
            # nchwc -> nchw
            data = inputs[0].transpose((0, 1, 4, 2, 3)).reshape(inputs[0].shape[0], inputs[0].shape[1]*inputs[0].shape[-1], inputs[0].shape[2], inputs[0].shape[3])
            data = data.astype("float32")
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

    def import_conv2d_mem_reuse(self, target="llvm", dtype="float32"):
        input_shape = (1, 32, 112, 112, 4)
        filter_shape = (32, 128, 3, 3, 4)
        W1 = relay.var("weight1", shape=filter_shape, dtype=dtype)
        W2 = relay.var("weight2", shape=filter_shape, dtype=dtype)
        W3 = relay.var("weight3", shape=filter_shape, dtype=dtype)
        W4 = relay.var("weight4", shape=filter_shape, dtype=dtype)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.nn.conv2d(A, W1, data_layout="NCHW4c", kernel_layout="OIHW4o")
        C = relay.nn.conv2d(B, W2, data_layout="NCHW4c", kernel_layout="OIHW4o")
        D = relay.nn.conv2d(C, W3, data_layout="NCHW4c", kernel_layout="OIHW4o")
        E = relay.nn.conv2d(D, W4, data_layout="NCHW4c", kernel_layout="OIHW4o")
        mod = relay.Function([A, W1, W2, W3, W4], E)

        params = {
            "weight1": tvm.nd.array(np.random.uniform(0.5, 1.5, filter_shape).astype(dtype)),
            "weight2": tvm.nd.array(np.random.uniform(0.5, 1.5, filter_shape).astype(dtype)),
            "weight3": tvm.nd.array(np.random.uniform(0.5, 1.5, filter_shape).astype(dtype)),
            "weight4": tvm.nd.array(np.random.uniform(0.5, 1.5, filter_shape).astype(dtype)),
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
            w3_np = params["weight3"].asnumpy()
            kernel3 = w3_np.transpose((0, 4, 1, 2, 3)).reshape(w3_np.shape[0] * w3_np.shape[4], w3_np.shape[1], w3_np.shape[2], w3_np.shape[3])
            w4_np = params["weight4"].asnumpy()
            kernel4 = w4_np.transpose((0, 4, 1, 2, 3)).reshape(w4_np.shape[0] * w4_np.shape[4], w4_np.shape[1], w4_np.shape[2], w4_np.shape[3])
            conv2d = testing.conv2d_nchw_python(data, kernel1, 1, 0)
            conv2d = testing.conv2d_nchw_python(conv2d, kernel2, 1, 0)
            conv2d = testing.conv2d_nchw_python(conv2d, kernel3, 1, 0)
            conv2d = testing.conv2d_nchw_python(conv2d, kernel4, 1, 0)
            np_result = conv2d
            # nkhw -> nkhwk
            np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(0, 1, 3, 4, 2)
            return [np_result,]

        class classify(Validator):
            def __init__(self):
                self.inputs = {"data" : np.random.uniform(0.5, 1.5, input_shape).astype(dtype)}
            def GetReference(self):
                inputs = []
                for key, data in self.inputs.items():
                    inputs.append(data)
                return validator(inputs)
            def Validate(self, m, ref_outputs):
                for i, ref_output in enumerate(ref_outputs):
                    tvm_output = m.get_output(i)
                    output = tvm_output.asnumpy()
                    np.testing.assert_allclose(output, ref_output, rtol=1e-3, atol=1e-3)




        return (mod, params, {"data": input_shape}, dtype, target, classify())

    def import_conv2d_mem_reuse1(self, target="llvm", dtype="float32"):
        #input_shape = (1, 1, 112, 112, 4)
        input_shape = (1, 4, 112, 112)
        filter_shape = (1, 4, 2, 2, 4)
        #A = relay.var("data", shape=input_shape, dtype=dtype)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B1 = relay.var("weight1", shape=filter_shape, dtype=dtype)
        B2 = relay.var("weight2", shape=filter_shape, dtype=dtype)
        B3 = relay.var("weight3", shape=filter_shape, dtype=dtype)
        B4 = relay.var("weight4", shape=filter_shape, dtype=dtype)
        Ap = relay.layout_transform(A, src_layout="NCHW", dst_layout="NCHW4c")
        C0 = relay.nn.conv2d(Ap, B1, data_layout="NCHW4c", kernel_layout="OIHW4o")
        C = relay.nn.relu(C0)
        D0 = relay.nn.conv2d(C, B2,  data_layout="NCHW4c", kernel_layout="OIHW4o")
        D = relay.nn.relu(D0)
        E0 = relay.nn.conv2d(D, B3,  data_layout="NCHW4c", kernel_layout="OIHW4o")
        E = relay.nn.relu(E0)
        F0 = relay.nn.conv2d(E, B4,  data_layout="NCHW4c", kernel_layout="OIHW4o")
        F = relay.nn.relu(F0)
        mod = relay.Function([A, B1, B2, B3, B4], F)

        params = {
            "weight1": tvm.nd.array(np.random.uniform(-1, 1, filter_shape).astype(dtype)),
            "weight2": tvm.nd.array(np.random.uniform(-1, 1, filter_shape).astype(dtype)),
            "weight3": tvm.nd.array(np.random.uniform(-1, 1, filter_shape).astype(dtype)),
            "weight4": tvm.nd.array(np.random.uniform(-1, 1, filter_shape).astype(dtype)),
        }
        def validator(inputs):
            vec_length = 4
            # nchwc -> nchw
            data = inputs[0]
            # kcrsk -> kcrs
            w1_np = params["weight1"].asnumpy()
            kernel1 = w1_np.transpose((0, 4, 1, 2, 3)).reshape(w1_np.shape[0] * w1_np.shape[4], w1_np.shape[1], w1_np.shape[2], w1_np.shape[3])
            w2_np = params["weight2"].asnumpy()
            kernel2 = w2_np.transpose((0, 4, 1, 2, 3)).reshape(w2_np.shape[0] * w2_np.shape[4], w2_np.shape[1], w2_np.shape[2], w2_np.shape[3])
            w3_np = params["weight3"].asnumpy()
            kernel3 = w3_np.transpose((0, 4, 1, 2, 3)).reshape(w3_np.shape[0] * w3_np.shape[4], w3_np.shape[1], w3_np.shape[2], w3_np.shape[3])
            w4_np = params["weight3"].asnumpy()
            kernel4 = w4_np.transpose((0, 4, 1, 2, 3)).reshape(w4_np.shape[0] * w4_np.shape[4], w4_np.shape[1], w4_np.shape[2], w4_np.shape[3])
            conv2d = testing.conv2d_nchw_python(data, kernel1, 1, 0)
            conv2d = np.maximum(conv2d, 0)
            conv2d = testing.conv2d_nchw_python(conv2d, kernel2, 1, 0)
            conv2d = np.maximum(conv2d, 0)
            conv2d = testing.conv2d_nchw_python(conv2d, kernel3, 1, 0)
            conv2d = np.maximum(conv2d, 0)
            conv2d = testing.conv2d_nchw_python(conv2d, kernel4, 1, 0)
            conv2d = np.maximum(conv2d, 0)
            np_result = conv2d
            # nkhw -> nkhwk
            np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(0, 1, 3, 4, 2)
            return [np_result,]
        return (mod, params, {"data": input_shape}, dtype, target, validator)

    def import_conv2d_x4(self, target="llvm", dtype="float32"):
        input_shape = (1, 3, 224, 224)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        w1_shape = (32, 3, 3, 3)
        w1 = relay.var("weight1", shape=w1_shape, dtype=dtype)
        w2_shape = (32, 1, 3, 3)
        w2 = relay.var("weight2", shape=w2_shape, dtype=dtype)
        w3_shape = (64, 32, 1, 1)
        w3 = relay.var("weight3", shape=w3_shape, dtype=dtype)
        w4_shape = (64, 1, 3, 3)
        w4 = relay.var("weight4", shape=w4_shape, dtype=dtype)

        B = relay.nn.conv2d(A, w1, strides=[2, 2], padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3], data_layout="NCHW") # general
        C = relay.nn.conv2d(B, w2, padding=[1, 1, 1, 1], groups=32, channels=32, kernel_size=[3, 3]) # depthwise
        D = relay.nn.conv2d(C, w3, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]) # 1x1
        E = relay.nn.conv2d(D, w4, strides=[2, 2], padding=[1, 1, 1, 1], groups=64, channels=64, kernel_size=[3, 3]) # depthwise strided

        func = relay.Function([A, w1, w2, w3, w4], E)
        mod = tvm.IRModule.from_expr(func)

        params = {
            "weight1": tvm.nd.array(np.random.uniform(-1, 1, w1_shape).astype(dtype)),
            "weight2": tvm.nd.array(np.random.uniform(-1, 1, w2_shape).astype(dtype)),
            "weight3": tvm.nd.array(np.random.uniform(-1, 1, w3_shape).astype(dtype)),
            "weight4": tvm.nd.array(np.random.uniform(-1, 1, w4_shape).astype(dtype)),
        }

        # layout transformation
        layout_config = relay.transform.LayoutConfig(skip_layers=[0, 3])
        desired_layouts = {"nn.conv2d": ["NCHW4c", "OIHW4o"]}
        with layout_config:
            seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
        # downcast to float16
        # if dtype == "float16":
        #     mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        return (mod, params, {"data": input_shape}, dtype, target)

    # fused_nn_conv2d_add_nn_relu_65
    # fn (%p0: Tensor[(1, 32, 17, 17, 4), float16], %p1: Tensor[(48, 128, 7, 1, 4), float16], %p2: Tensor[(1, 48, 1, 1, 4), float16], Primitive=1) -> Tensor[(1, 48, 17, 17, 4), float16] {
    #   %0 = nn.conv2d(%p0, %p1, padding=[3, 0, 3, 0], kernel_size=[7, 1], data_layout="NCHW4c", kernel_layout="OIHW4o") /* ty=Tensor[(1, 48, 17, 17, 4), float16] */;
    #   %1 = add(%0, %p2) /* ty=Tensor[(1, 48, 17, 17, 4), float16] */;
    #   nn.relu(%1) /* ty=Tensor[(1, 48, 17, 17, 4), float16] */
    # }
    def import_conv2d_a6x_compiler_hang(self, target="llvm", dtype="float32"):
        input_shape, filter_shape, bias_shape = (1, 128, 17, 17), (48, 128, 7, 1, 4), (1, 48, 1, 1, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.var("weight", shape=filter_shape, dtype=dtype)
        C = relay.var("bias", shape=bias_shape, dtype=dtype)

        a = relay.layout_transform(A, src_layout="NCHW", dst_layout="NCHW4c")
        D = relay.nn.conv2d(a, B, data_layout="NCHW4c", kernel_layout="OIHW4o", padding=[3, 0, 3, 0], kernel_size=[7, 1],  out_dtype=dtype)
        E = C + D
        F = relay.nn.relu(E)
        mod = relay.Function([A, B, C], F)
        #mod, params = relay.testing.init.create_workload(func)
        np.random.seed(0)
        initializer = relay.testing.init.Xavier()
        filter_data = np.zeros(filter_shape).astype(dtype)
        bias_data = np.ones(bias_shape).astype(dtype)
        initializer("weight", filter_data)
        params = {
            "weight": tvm.nd.array(filter_data),
            "bias": tvm.nd.array(bias_data),
        }

        def validator(inputs):
            vec_length = 4
            data = inputs[0].astype("float32")
            # kcrsk -> kcrs
            w_np = params["weight"].asnumpy()
            kernel = w_np.transpose((0, 4, 1, 2, 3)).reshape(w_np.shape[0] * w_np.shape[4], w_np.shape[1], w_np.shape[2], w_np.shape[3])
            np_result = testing.conv2d_nchw_python(data, kernel, stride=(1,1), padding=(3,0,3,0))
            bias = params["bias"].asnumpy()
            bias = bias.transpose((0, 1, 4, 2, 3)).reshape(bias.shape[0], bias.shape[1] * bias.shape[-1], bias.shape[2], bias.shape[3])
            np_result += bias
            np_result = np.maximum(np_result, 0)
            # nkhw -> nkhwk
            np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(0, 1, 3, 4, 2)
            return [np_result,]
        return (mod, params, {"data": input_shape}, dtype, target, validator)
        #return (mod, params, {"data": input_shape}, dtype, target)

    def import_conv2d_a6x_compiler_hang_kernel_single_op(self, target="llvm", dtype="float32"):
        input_shape, filter_shape, bias_shape = (1, 32, 17, 17, 4), (48, 128, 7, 1, 4), (1, 48, 1, 1, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.var("weight", shape=filter_shape, dtype=dtype)
        C = relay.var("bias", shape=bias_shape, dtype=dtype)

        a = relay.nn.relu(A)
        D = relay.nn.conv2d(a, B, data_layout="NCHW4c", kernel_layout="OIHW4o", padding=[3, 0, 3, 0], kernel_size=[7, 1],  out_dtype=dtype)
        E = C + D
        F = relay.nn.relu(E)
        mod = relay.Function([A, B, C], F)
        #mod, params = relay.testing.init.create_workload(func)
        np.random.seed(0)
        initializer = relay.testing.init.Xavier()
        filter_data = np.zeros(filter_shape).astype(dtype)
        bias_data = np.ones(bias_shape).astype(dtype)
        initializer("weight", filter_data)
        params = {
            "weight": tvm.nd.array(filter_data),
            "bias": tvm.nd.array(bias_data),
        }

        def validator(inputs):
            vec_length = 4
            data = np.maximum(inputs[0], 0)
            data = data.transpose((0, 1, 4, 2, 3)).reshape(data.shape[0], data.shape[1]*data.shape[-1], data.shape[2], data.shape[3])
            data = data.astype("float32")
            # kcrsk -> kcrs
            w_np = params["weight"].asnumpy()
            kernel = w_np.transpose((0, 4, 1, 2, 3)).reshape(w_np.shape[0] * w_np.shape[4], w_np.shape[1], w_np.shape[2], w_np.shape[3])
            np_result = testing.conv2d_nchw_python(data, kernel, stride=(1,1), padding=(3,0,3,0))
            bias = params["bias"].asnumpy()
            bias = bias.transpose((0, 1, 4, 2, 3)).reshape(bias.shape[0], bias.shape[1] * bias.shape[-1], bias.shape[2], bias.shape[3])
            np_result += bias
            np_result = np.maximum(np_result, 0)
            # nkhw -> nkhwk
            np_result = np_result.reshape(np_result.shape[0], np_result.shape[1]//vec_length, vec_length, np_result.shape[2], np_result.shape[3]).transpose(0, 1, 3, 4, 2)
            return [np_result,]
        return (mod, params, {"data": input_shape}, dtype, target, validator)
        #return (mod, params, {"data": input_shape}, dtype, target)

    def import_global_pooling(self, target="llvm", dtype="float32"):
        input_shape = (1, 512, 8, 8, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.nn.global_avg_pool2d(A, layout="NCHW4c")
        mod = relay.Function([A], B)

        def validator(inputs):
            np_result = np.apply_over_axes(np.mean, inputs[0], [2, 3])
            return [np_result,]
        return (mod, {}, {"data": input_shape}, dtype, target, validator)
        #return (mod, params, {"data": input_shape}, dtype, target)

    def import_max_pooling(self, target="llvm", dtype="float32"):
        input_shape = (1, 512, 8, 8, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.nn.max_pool2d(A, pool_size=(8,8), layout="NCHW4c")
        mod = relay.Function([A], B)

        def validator(inputs):
            #np_result = np.apply_over_axes(np.max, inputs[0], [2, 3])
            import skimage.measure
            np_result = skimage.measure.block_reduce(inputs[0], (1, 1, 8, 8, 1), np.max)
            return [np_result,]

            return [np_result,]
        return (mod, {}, {"data": input_shape}, dtype, target, validator)
        #return (mod, params, {"data": input_shape}, dtype, target)

    def import_max_pooling2(self, target="llvm", dtype="float32"):
        input_shape = (1, 16, 224, 224, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.nn.max_pool2d(A, pool_size=(2, 2), strides=(2,2), layout="NCHW4c")
        mod = relay.Function([A], B)

        def validator(inputs):
            import skimage.measure
            np_result = skimage.measure.block_reduce(inputs[0], (1, 1, 2, 2, 1), np.max)
            return [np_result,]
        return (mod, {}, {"data": input_shape}, dtype, target, validator)
        #return (mod, params, {"data": input_shape}, dtype, target)

    def import_avg_pooling(self, target="llvm", dtype="float32"):
        input_shape = (1, 512, 8, 8, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.nn.avg_pool2d(A, pool_size=(8,8), layout="NCHW4c")
        mod = relay.Function([A], B)

        def validator(inputs):
            np_result = np.apply_over_axes(np.mean, inputs[0], [2, 3])
            return [np_result,]
        return (mod, {}, {"data": input_shape}, dtype, target, validator)
        #return (mod, params, {"data": input_shape}, dtype, target)

    def import_concat(self, target="llvm", dtype="float32"):
        input_shape1 = (1, 80, 8, 8)
        input_shape2 = (1, 432, 8, 8)
        A = relay.var("data1", shape=input_shape1, dtype=dtype)
        B = relay.var("data2", shape=input_shape2, dtype=dtype)
        Ap = relay.layout_transform(A, src_layout="NCHW", dst_layout="NCHW4c")
        Bp = relay.layout_transform(B, src_layout="NCHW", dst_layout="NCHW4c")
        Ap = relay.nn.max_pool2d(Ap, pool_size=(1,1), layout="NCHW4c")
        Bp = relay.nn.max_pool2d(Bp, pool_size=(1,1), layout="NCHW4c")
        Cp = relay.concatenate([Ap, Bp], axis=1)
        Cp = relay.nn.max_pool2d(Cp, pool_size=(1,1), layout="NCHW4c")
        C = relay.layout_transform(Cp, src_layout="NCHW4c", dst_layout="NCHW")
        mod = relay.Function([A, B], C)

        def validator(inputs):
            print([i.shape for i in inputs])
            np_result = np.concatenate(inputs, axis=1)
            return [np_result,]
        return (mod, {}, {"data1": input_shape1, "data2": input_shape2}, dtype, target, validator)
        #return (mod, params, {"data": input_shape}, dtype, target)

    def import_layout_transform_expand(self, target="llvm", dtype="float32"):
        input_shape = (1, 128, 8, 8)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.layout_transform(A, src_layout="NCHW", dst_layout="NCHW4c")
        C = relay.nn.avg_pool2d(B, pool_size=(8,8), layout="NCHW4c")
        mod = relay.Function([A], C)

        def validator(inputs):
            vec_length = 4
            np_result = inputs[0].reshape(inputs[0].shape[0], inputs[0].shape[1]//vec_length, vec_length, inputs[0].shape[2], inputs[0].shape[3]).transpose(0, 1, 3, 4, 2)
            np_result = np.apply_over_axes(np.mean, np_result, [2, 3])
            return [np_result,]
        return (mod, {}, {"data": input_shape}, dtype, target, validator)

    def import_layout_transform_contract(self, target="llvm", dtype="float32"):
        input_shape = (1, 32, 8, 8, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.nn.avg_pool2d(A, pool_size=(8,8), layout="NCHW4c")
        C = relay.layout_transform(B, src_layout="NCHW4c", dst_layout="NCHW")

        mod = relay.Function([A], C)

        def validator(inputs):
            vec_length = 4
            np_result = np.apply_over_axes(np.mean, inputs[0], [2, 3])
            np_result = np_result.transpose((0, 1, 4, 2, 3)).reshape(np_result.shape[0], np_result.shape[1]*np_result.shape[-1], np_result.shape[2], np_result.shape[3])
            return [np_result,]
        return (mod, {}, {"data": input_shape}, dtype, target, validator)


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
        action="store_true",
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
            builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15, n_parallel=2),
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
    import mxnet.gluon as gluon

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

        self.inputs = {name : image}

    def Validate(self, m, ref_outputs=[]):
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
            # print("Relay model to compile:\n")
            # print(tvm_mod)
            graph, lib, params = relay.build(
                tvm_mod, target_host=target_host, target=target, params=params
            )
            #lib2 = relay.build(tvm_mod, target=target, target_host=target_host, params=params)
            #lib2.export_library("_model.so", ndk.create_shared)

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
            lib.export_library(dso_binary_path, ndk.create_shared)
            remote_path = "/data/local/tmp/" + dso_binary
            self.remote.upload(dso_binary_path)
            print("Uploading binary...")
            rlib = self.remote.load_module(dso_binary)
            m = graph_runtime.create(graph, rlib, ctx)
        else:
            print("Using local runtime")
            ctx = tvm.device(target, 0)
            m = graph_runtime.create(graph, lib, ctx)

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

        #num_iter = 1
        #print("change number of iter before benchmarking")
        num_iter = 100
        if args.debug:
            m.run()
            time_f = m.module.time_evaluator("run", ctx, number=num_iter)
        else:
            time_f = m.module.time_evaluator("run", ctx, number=num_iter*10)
        cost = time_f().mean
        print("%g secs/iteration\n" % cost)

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
