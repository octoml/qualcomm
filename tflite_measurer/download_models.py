import os


MODELS_DIR = "models"
# If the url has a path to local file then download step will be skipped and the model should be just converted to TFLite format
MODELS_INFO = {
    "mobilenet": {
        "url": "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/mobilenet-v1/mobilenet-v1-1.0.pb",
        "input_shapes": {'input': [1, 224, 224, 3]},
        "output_names": ["MobilenetV1/Predictions/Reshape_1"],
    },
    "resnet50": {
        "url": "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/resnet-v2-50/resnet-v2-50.pb",
        "input_shapes": {"input": [1, 299, 299, 3]},
        "output_names": ["resnet_v2_50/predictions/Reshape_1"],
    },
    "inception": {
        "url": "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/inception-v3/inception-v3.pb",
        "input_shapes": {"input": [1, 299, 299, 3]},
        "output_names": ["InceptionV3/Predictions/Reshape_1"],
    },
    "vgg16": "",
    "deeplab": {
        "url": "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/deeplab-v3-plus/deeplab-v3-plus-mobilenet-v2.pb",
        "input_shapes": {"sub_7": [1, 513, 513, 3]},
        "output_names": ["ResizeBilinear_2"],
    },
    "yolo": {
        "url": "http://cnbj1.fds.api.xiaomi.com/mace/miai-models/yolo-v3/yolo-v3.pb",
        "input_shapes": {"input_1": [1, 416, 416, 3]},
        "output_names": ["conv2d_59/BiasAdd", "conv2d_67/BiasAdd", "conv2d_75/BiasAdd"],
    },
    "classifier": {
        "url": "{}/classifier.pb".format(MODELS_DIR),
        "input_shapes": {"image": (1, 300, 300, 3)},
        "output_names": ["Edgetpu_M/prob", "Edgetpu_M/prob_openset"],
    },
    "detector": {
        "url": "{}/detector.pb".format(MODELS_DIR),
        "input_shapes": {"input" : [1, 320, 320, 3], "anchors": [1, 1, 19125, 4]},
        "output_names": ["AIG/classification_2", "AIG/clipped_boxes/concat"],
    },
}
MODELS = MODELS_INFO.keys()


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download models"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        required=True,
        help="Model to download",
        choices=MODELS,
    )

    args = parser.parse_args()
    return args


def prepare_dir():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)


def convert_tf_to_tflite(tf_file_name, model):
    # https://neuralet.com/docs/tutorials/tf-object-detection-api-model-quantization/
    import tensorflow as tf

    model_url = MODELS_INFO[model]["url"]
    model_name = model_url[model_url.rfind("/") + 1:model_url.rfind(".")]
    tflite_model_file = MODELS_DIR + "/" + model_name + ".tflite"

    input_shapes = MODELS_INFO[model]["input_shapes"]
    input_arrays = input_shapes.keys()
    output_arrays = MODELS_INFO[model]["output_names"]

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(tf_file_name,
        input_arrays=input_arrays,
        output_arrays=output_arrays,
        input_shapes=input_shapes)
    converter.allow_custom_ops = True

    tflite_model_quant = converter.convert()
    with open(tflite_model_file, "wb") as tflite_file:
        tflite_file.write(tflite_model_quant)

def process(model):
    import urllib.request

    if os.path.isfile(MODELS_INFO[model]["url"]):
        convert_tf_to_tflite(MODELS_INFO[model]["url"], model)
    else:
        local_file = "model.pb"
        urllib.request.urlretrieve(MODELS_INFO[model]["url"], local_file)
        convert_tf_to_tflite(local_file, model)
        os.remove(local_file)

if __name__ == '__main__':
    args = get_args()
    prepare_dir()

    process(args.model)
