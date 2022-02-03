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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-else-return
"""conv2d schedule on Qualcomm Adreno GPU"""
import tvm
from tvm import te
from tvm import autotvm
import numpy

from tvm.topi import nn
from tvm.topi.utils import simplify
from ..utils import get_const_tuple, traverse_inline

# OpenCL spec declares minimal value of max width/height dimension for 2D image.
# See CL_DEVICE_IMAGE2D_MAX_WIDTH in https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clGetDeviceInfo.html
OCL_IMAGE2D_MIN_SIZE = 16384

@autotvm.register_topi_compute("conv2d_nhwc.image2d")
def conv2d_nhwc(cfg, data, kernel, strides, padding, dilation, out_dtype="float16"):
    """Compute conv2d with NCHWc layout"""
    args={"shared" : False, "accumulator" : "float16"}
    return compute_conv2d_NHWC_HWIO(data, kernel, strides, padding, dilation, out_dtype, args=args)

@autotvm.register_topi_compute("conv2d_nhwc_acc32.image2d")
def conv2d_nhwc_acc32(cfg, data, kernel, strides, padding, dilation, out_dtype="float16"):
    """Compute conv2d with NCHWc layout"""
    args={"shared" : False, "accumulator" : "float32"}
    return compute_conv2d_NHWC_HWIO(data, kernel, strides, padding, dilation, out_dtype, args=args)

@autotvm.register_topi_schedule("conv2d_nhwc.image2d")
def schedule_conv2d_nhwc(cfg, outs):
    return schedule_conv2d_nhwc_impl(cfg, outs, tag="cast_from_acc16")

@autotvm.register_topi_schedule("conv2d_nhwc_acc32.image2d")
def schedule_conv2d_nhwc_acc32(cfg, outs):
    return schedule_conv2d_nhwc_impl(cfg, outs, tag="cast_from_acc32")

def schedule_conv2d_nhwc_impl(cfg, outs, tag):
    """Create the schedule for conv2d_nhwc"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    def _callback(op):
        if op.tag == tag:
            args={"shared" : False}
            schedule_conv2d_NHWC(cfg, s, op.output(0), args)

    traverse_inline(s, outs[0].op, _callback)
    return s


def compute_conv2d_NHWC_HWIO(Input, Filter, stride, padding, dilation, out_dtype=None, args={}):
    """Convolution operator in NHWC layout. """

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

    batch, in_height, in_width, in_channel = Input.shape
    kernel_h, kernel_w, _, out_channels = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    out_height_orig = out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width_orig = out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    in_channel_block = 4
    in_channel_tail = in_channel % in_channel_block
    in_channel_chunk = in_channel // in_channel_block
    num_filter_block = 4
    num_filter_tail = out_channels % num_filter_block
    num_filter_chunk = out_channels // num_filter_block

    pad_value = tvm.tir.const(0, Input.dtype)
    # compute:
    if in_channel_tail == 0:
        Input = te.compute(
            [batch, in_height, in_width, in_channel_chunk, in_channel_block],
            lambda nn, yy, xx, icc, icb: Input[nn, yy, xx, icc * in_channel_block + icb],
            name="input_pack",
            tag="input_pack",
        )
    else:
        in_channel_chunk += 1

        def _reorder_data(*indices):
            condition = []
            condition.append(indices[3] == in_channel_chunk - 1)
            condition.append(indices[4] >= in_channel_tail)
            condition = tvm.tir.all(*condition)
            return tvm.tir.if_then_else(
                    condition,
                    pad_value,
                    Input[indices[0], indices[1], indices[2], indices[3] * in_channel_block + indices[4]])
        Input = te.compute(
            [batch, in_height, in_width, in_channel_chunk, in_channel_block],
            _reorder_data,
            name="input_pack",
            tag="input_pack_expanded",
        )
    if num_filter_tail == 0 and in_channel_tail == 0:
        Filter = te.compute(
            [kernel_h, kernel_w, in_channel, num_filter_chunk, num_filter_block],
            lambda kh, kw, ic, nfc, nfb: Filter[kh, kw, ic, nfc * num_filter_block + nfb],
            name="filter_pack",
            tag="filter_pack",
        )
    else:
        num_filter_chunk += 1

        # HWIO
        def _reorder_weights(*indices):
            conditionA = []
            conditionA.append(indices[3] == num_filter_chunk - 1)
            conditionA.append(indices[4] >= num_filter_block)
            conditionAT = tvm.tir.all(*conditionA)

            conditionO = []
            conditionO.append(conditionAT)
            conditionO.append(indices[2] >= in_channel_chunk * in_channel_block + in_channel_tail)
            conditionOT = tvm.tir.any(*conditionO)
            return tvm.tir.if_then_else(
                    conditionOT,
                    pad_value,
                    Filter[indices[0], indices[1], indices[2], indices[3] * num_filter_block + indices[4]])
        Filter = te.compute(
            [kernel_h, kernel_w, in_channel, num_filter_chunk, num_filter_block],
            _reorder_weights,
            name="filter_pack",
            tag="filter_pack_expanded",
        )

    # can output shape be divded by 2 or even 4?
    # if it cannot be divided, need to extend for further help with split
    # theortically there should be addition padding for inputs, but it will be optimized by
    # cache_read InferBound. We must proceed pad here exactly to produce tensor which is
    # required for calculation of original out size, not more! In other case intermediate
    # tensor might be allcoated with less sizes while compute will try to fill the expanded
    # one - data discrepancy as a result
    # And in case of textures it is not a problem if we provide texture of less size because
    # 1. It is not important which valuses would be for extra calc - these calculations are
    #    required only for better utilizatin of GPU fit to working groups
    # 2. When we request pixel out opf bound, texture will handle this correctly. As mentioned
    #    above, the value itself is not important
    if out_height % 2 != 0:
        out_height += 1
    if out_width % 2 != 0:
        out_width += 1

    if out_height % 4 != 0:
        out_height += 2
    if out_width % 4 != 0:
        out_width += 2

    # compute graph
    pad_before = [0, pad_top, pad_left, 0, 0]
    pad_after = [0, pad_down, pad_right, 0, 0]
    # calculation of real used input size:
    input_latest_w = (out_width_orig - 1) * stride_w + (kernel_w - 1) * dilation_w + 1
    input_latest_h = (out_height_orig - 1) * stride_h + (kernel_h - 1) * dilation_h + 1
    if input_latest_w < in_width + pad_before[3] + pad_after[3]:
        pad_after[3] -= in_width + pad_before[3] + pad_after[3] - input_latest_w
    if input_latest_h < in_height + pad_before[2] + pad_after[2]:
        pad_after[2] -= in_height + pad_before[2] + pad_after[2] - input_latest_h

    temp = nn.pad(Input, pad_before, pad_after, name="pad_temp")

    rcc = te.reduce_axis((0, in_channel_chunk), name="rc")
    rcb = te.reduce_axis((0, in_channel_block), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    conv = te.compute(
        (batch, out_height, out_width, num_filter_chunk, num_filter_block),
        lambda nn, yy, xx, fc, fb: te.sum(
            (temp[nn, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rcc, rcb]
            * Filter[ry, rx, rcc * in_channel_block + rcb, fc, fb]).astype(args["accumulator"]),
            axis=[ry, rx, rcc, rcb],
        ),
        tag="conv2d_nhwc",
    )

    dummy_cast = te.compute((batch, out_height_orig, out_width_orig, num_filter_chunk, num_filter_block), lambda n,y,x,fc,fb: conv[n,y,x,fc,fb].astype(out_dtype), tag="dummy_cast")
    return te.compute((batch, out_height_orig, out_width_orig, out_channels), lambda n,y,x,c: dummy_cast[n,y,x,c//4,c%4], tag="cast_from_acc" + args["accumulator"][-2:])

def getDiv(value, start):
    div = 1
    for d in range(start,0,-1):
        if (value % d) == 0:
            div = d
            break
    return div

def schedule_conv2d_NHWC(cfg, s, output, args={}):
    """schedule optimized for batch size = 1"""
    dummy = output.op.input_tensors[0]
    conv = dummy.op.input_tensors[0]
    #latest = s.outputs[0].output(0)

    ##### space definition begin #####
    n, y, x, fc, fb = s[conv].op.axis
    ry, rx, rcc, rcb = s[conv].op.reduce_axis
    cfg.define_split("tile_fc", fc, num_outputs=3,
                filter=lambda entity: entity.size[1] <= 16 and entity.size[2] >= 2 and entity.size[2] < 256 )
    cfg.define_split("tile_y", y, num_outputs=3,
                filter=lambda entity: entity.size[1] <= 16 and entity.size[2] < 16 )
    cfg.define_split("tile_x", x, num_outputs=3,
                filter=lambda entity: entity.size[1] <= 16 and entity.size[2] < 16 )

    cfg.define_split("tile_rc", rcc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])

    cfg.multi_filter(filter=lambda entity: entity["tile_fc"].size[2] * entity["tile_y"].size[2] * entity["tile_x"].size[2] in range(32,1024))
    ##### space definition end #####

    pad_data, kernel = s[conv].op.input_tensors
    pack_data = pad_data.op.input_tensors[0]
    if s[pack_data].op.tag != "input_pack_expanded":
        s[pack_data].compute_inline()
    else:
        axes = s[pack_data].op.axis
        fused = s[pack_data].fuse(*axes[:-1])
        shape = get_const_tuple(pack_data.shape)
        ftc = numpy.prod(shape[:-1])
        div = getDiv(ftc, 64)
        block, thread = s[pack_data].split(fused, factor=div)
        s[pack_data].bind(block, te.thread_axis("blockIdx.x"))
        s[pack_data].bind(thread, te.thread_axis("threadIdx.x"))

    s[pad_data].compute_inline()
    if s[kernel].op.tag != "filter_pack_expanded":
        s[kernel].compute_inline()
    else:
        axes = s[kernel].op.axis
        fused = s[kernel].fuse(*axes[:-1])
        shape = get_const_tuple(kernel.shape)
        ftc = numpy.prod(shape[:-1])
        div = getDiv(ftc, 64)
        block, thread = s[kernel].split(fused, factor=div)
        s[kernel].bind(block, te.thread_axis("blockIdx.x"))
        s[kernel].bind(thread, te.thread_axis("threadIdx.x"))

    latest = s.outputs[0].output(0)

    # create cache stage
    def get_texture_storage(shape):
        if shape[0] * shape[1] * shape[2] < OCL_IMAGE2D_MIN_SIZE and shape[3] < OCL_IMAGE2D_MIN_SIZE:
            return "texture"
        elif shape[0] * shape[1] < OCL_IMAGE2D_MIN_SIZE and shape[2] * shape[3] < OCL_IMAGE2D_MIN_SIZE:
            return "texture:nhwc"
        else:
            return "texture:weight"

    AT = s.cache_read(pad_data, get_texture_storage(pad_data.shape), [conv])
    WT = s.cache_read(kernel, get_texture_storage(kernel.shape), [conv])
    def copy_to_texture(stage):
        axes = s[stage].op.axis
        fused = s[stage].fuse(*axes[:-1])
        block, thread = s[stage].split(fused, factor=32)
        s[stage].vectorize(axes[-1])
        s[stage].bind(block, te.thread_axis("blockIdx.x"))
        s[stage].bind(thread, te.thread_axis("threadIdx.x"))
    copy_to_texture(AT)
    copy_to_texture(WT)

    # tile and bind spatial axes
    n, y, x, fc, fb = s[dummy].op.axis

    kernel_scope, n = s[dummy].split(n, nparts=1)

    bf, vf, tf = cfg["tile_fc"].apply(s, dummy, fc)
    by, vy, ty = cfg["tile_y"].apply(s, dummy, y)
    bx, vx, tx = cfg["tile_x"].apply(s, dummy, x)

    by = s[dummy].fuse(n, by)
    s[dummy].bind(bf, te.thread_axis("blockIdx.z"))
    s[dummy].bind(by, te.thread_axis("blockIdx.y"))
    s[dummy].bind(bx, te.thread_axis("blockIdx.x"))
    s[dummy].bind(vf, te.thread_axis("vthread"))
    s[dummy].bind(vy, te.thread_axis("vthread"))
    s[dummy].bind(vx, te.thread_axis("vthread"))
    s[dummy].bind(tf, te.thread_axis("threadIdx.z"))
    s[dummy].bind(ty, te.thread_axis("threadIdx.y"))
    s[dummy].bind(tx, te.thread_axis("threadIdx.x"))
    s[dummy].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fb)
    s[dummy].vectorize(fb)

    s[conv].compute_at(s[dummy], tx)

    # tile reduction axes
    n, y, x, fc, fb = s[conv].op.axis

    ry, rx, rcc, rcb = s[conv].op.reduce_axis
    rco, rci = cfg["tile_rc"].apply(s, conv, rcc)
    ryo, ryi = cfg["tile_ry"].apply(s, conv, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, conv, rx)

    s[conv].reorder(rco, ryo, rxo, rci, ryi, rxi, n, y, x, fc, fb)
    s[conv].vectorize(fb)
    s[conv].unroll(rcb)
    # unroll
    s[dummy].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[dummy].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    s[latest].compute_root()

    N, OH, OW, OC = get_const_tuple(latest.shape)

    if OC % 4 == 0:
        n, oh, ow, oc = s[latest].op.axis
        occ, ocb = s[latest].split(oc, factor=4)
        s[latest].reorder(n, oh, ow, occ, ocb)
        s[latest].vectorize(ocb)
        fused = s[latest].fuse(n, oh, ow, occ)

        ftc = N * OH * OW * OC / 4
        div = getDiv(ftc, 128)
        block, thread = s[latest].split(fused, factor=div)

        s[latest].bind(block, te.thread_axis("blockIdx.z"))
        s[latest].bind(thread, te.thread_axis("threadIdx.z"))
    else:
        axes = s[latest].op.axis
        fused = s[latest].fuse(*axes[:-1])
        if OC < 32:
            block, thread = s[latest].split(fused, factor=32)
            s[latest].bind(block, te.thread_axis("blockIdx.x"))
            s[latest].bind(thread, te.thread_axis("threadIdx.x"))
        else:
            s[latest].bind(fused, te.thread_axis("blockIdx.x"))
            s[latest].bind(*axes[-1:], te.thread_axis("threadIdx.x"))

    if output != latest:
        s[output].compute_inline()

    N, OH, OW, OC = get_const_tuple(latest.shape)
    H, W, I, _ = get_const_tuple(kernel.op.input_tensors[0].shape)

    if isinstance(N, int):
        cfg.add_flop(2 * N * OH * OW * OC * I * H * W)
