import os
import numpy as np

import tvm
from tvm import relay
from tvm import autotvm
from tvm.contrib import ndk
from tvm.relay.op import register_mixed_precision_conversion

conv2d_acc = "float32"

# Pick a priority > 10 to overwrite defaults, higher priorities take precedence
@register_mixed_precision_conversion("nn.conv2d", level=11)
def conv2d_mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
    global conv2d_acc
    return [
        # always do main calculation in mixed_precision_type
        relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
        # the dtype for the accumulator
        conv2d_acc,
        # the output dtype for the operation (usually fp16)
        mixed_precision_type,
    ]

@register_mixed_precision_conversion("nn.dense", level=11)
def conv2d_mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
    global conv2d_acc
    return [
        # always do main calculation in mixed_precision_type
        relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
        # the dtype for the accumulator
        conv2d_acc,
        # the output dtype for the operation (usually fp16)
        mixed_precision_type,
    ]

def convert_to_dtype(mod, dtype):
    # downcast to float16
    if dtype == "float16" or dtype == "float16_acc32":
        global conv2d_acc
        conv2d_acc = "float16" if dtype == "float16" else "float32"
        from tvm.ir import IRModule
        mod = IRModule.from_expr(mod)
        seq = tvm.transform.Sequential(
            [
                relay.transform.InferType(),
                relay.transform.ToMixedPrecision()
            ]
        )
        with tvm.transform.PassContext(
                config={"relay.ToMixedPrecision.keep_orig_output_dtype": True},
                opt_level=3):
            mod = seq(mod)
    return mod

def advanced_time_evaluator(m, func_name, ctx, number=1, repeat=1, min_repeat_ms=0, time_to_work_ms=0, cooldown_interval_ms=0, f_preproc="", mod_func_name=None):
        import inspect
        import math
        def ms_to_s(ms):
            return ms / 1000
        if mod_func_name is None:
            one_run_time = m.module.time_evaluator(func_name, ctx, number=1,repeat=1,min_repeat_ms=0)().results[0]
        else:
            one_run_time = m.module.time_evaluator(func_name, ctx, number=1,repeat=1,min_repeat_ms=0)(mod_func_name).results[0]
        repeats_to_cooldown = max(round(ms_to_s(time_to_work_ms)/one_run_time), 1)

        def _time_evaluator(func_name, m, ctx, number=1, repeat=1, min_repeat_ms=0, cooldown_interval_ms=0, repeats_to_cooldown=1, f_preproc=""):
            def evaluator(mod_func_name):
                import time
                from tvm.runtime.module import BenchmarkResult
                results = []
                for _ in range(math.ceil(repeat / repeats_to_cooldown)):
                    time_f = m.module.time_evaluator(func_name, ctx, number=number, repeat=repeats_to_cooldown, min_repeat_ms=min_repeat_ms, f_preproc=f_preproc)
                    if mod_func_name is None:
                        results.append(time_f().results)
                    else:
                        results.append(time_f(mod_func_name).results)
                    time.sleep(ms_to_s(cooldown_interval_ms))
                return BenchmarkResult([np.mean(r) for r in results])
            return evaluator

        if inspect.signature(m.module.time_evaluator).parameters.get("cooldown_interval_ms"):
            time_f = m.module.time_evaluator(func_name, ctx, number=number, repeat=repeat, min_repeat_ms=min_repeat_ms, cooldown_interval_ms=cooldown_interval_ms, repeats_to_cooldown=repeats_to_cooldown, f_preproc=f_preproc)
        else:
            time_f = _time_evaluator(func_name, m, ctx, number=number, repeat=repeat, min_repeat_ms=min_repeat_ms, cooldown_interval_ms=cooldown_interval_ms, repeats_to_cooldown=repeats_to_cooldown, f_preproc=f_preproc)

        return time_f