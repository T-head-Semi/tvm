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
"""Manage command line arguments."""
from collections import namedtuple
from inspect import ArgSpec
import os
import sys
import argparse
import copy
import json

import yaml

from .common import (
    HHBException,
    HHB_REGISTERED_PARSER,
    hhb_version,
    AttributeDict,
    get_parameters_info,
    argument_filter_helper,
    generate_readme_file,
    find_index,
    hhb_exit,
    ARGS_DEST_TO_OPTIONS_STRING,
    ALL_ARGUMENTS_INFO,
    collect_arguments_info,
)

from .preprocess_manage import collect_preprocess_config, set_preprocess_params
from .quantization_manage import (
    collect_quantization_config,
    set_quantize_params_by_board,
)
from .codegen_manage import collect_codegen_config, set_codegen_config
from .hhbir_manage import get_input_info_from_relay, get_output_info_from_relay


ALL_SUBCOMMAND = ["import", "quantize", "codegen", "simulate", "benchmark", "profiler"]
HIDDEN_SUBCOMMAND = ["benchmark"]


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


ArgSpec = namedtuple(
    "ArgSpec",
    [
        "name",
        "abbr_name",
        "type",
        "default",
        "choices",
        "action",
        "nargs",
        "help",
        "hidden",
        "version",
    ],
)


class ArgSpecHelper(object):
    def __init__(
        self,
        name,
        abbr_name=None,
        type=None,
        default=None,
        choices=None,
        action=None,
        nargs=None,
        help=None,
        hidden=False,
        version=None,
    ) -> None:
        self.spec = ArgSpec(
            name=name,
            abbr_name=abbr_name,
            type=type,
            default=default,
            choices=choices,
            action=action,
            nargs=nargs,
            help=help,
            hidden=hidden,
            version=version,
        )
        if action == "store_true" and default is None:
            self.value = False
        else:
            self.value = default

    def show_help(self):
        """Show help string."""
        return self.spec.help


class ArgumentsBase(object):
    def __init__(self) -> None:
        self._exclusive_group = []

    @property
    def name(self):
        return "argument_base"

    def add_exclusive_group(self, *args):
        self._exclusive_group.append(args)

    def _get_actual_args(self, arg_spec: ArgSpec):
        position_args = []
        if arg_spec.abbr_name is not None:
            position_args.append(arg_spec.abbr_name)
        position_args.append(arg_spec.name)

        keyword_args = {}
        if arg_spec.type is not None:
            keyword_args["type"] = arg_spec.type
        if arg_spec.default is not None:
            keyword_args["default"] = arg_spec.default
        if arg_spec.choices is not None:
            keyword_args["choices"] = arg_spec.choices
        if arg_spec.action is not None:
            keyword_args["action"] = arg_spec.action
        if arg_spec.nargs is not None:
            keyword_args["nargs"] = arg_spec.nargs
        if not arg_spec.hidden:
            if arg_spec.help is not None:
                keyword_args["help"] = arg_spec.help
        else:
            keyword_args["help"] = argparse.SUPPRESS
        if arg_spec.version is not None:
            keyword_args["version"] = arg_spec.version

        return position_args, keyword_args

    def set_parser(self, parser: argparse.ArgumentParser):
        # set exclusive group args
        for group in self._exclusive_group:
            group_arg = parser.add_mutually_exclusive_group()

            for g in group:
                if not isinstance(g, ArgSpecHelper):
                    raise HHBException("Only ArgSpecHelpr obj can be put into exclusive group.")
                p_args, k_args = self._get_actual_args(g.spec)
                group_arg.add_argument(*p_args, **k_args)

        # set other normal args
        unroll_group = []
        for group in self._exclusive_group:
            unroll_group.extend(group)
        for arg in self.__dict__.values():
            if not isinstance(arg, ArgSpecHelper):
                continue
            if arg in unroll_group:
                continue

            p_args, k_args = self._get_actual_args(arg.spec)
            parser.add_argument(*p_args, **k_args)

    def get_option_string_to_argument(self):
        """Get {(abbr_name, name): argument} dict."""
        res = {}
        for arg in self.__dict__.values():
            if not isinstance(arg, ArgSpecHelper):
                continue
            if arg.spec.abbr_name:
                res[(arg.spec.abbr_name, arg.spec.name)] = arg
            else:
                res[(arg.spec.name,)] = arg
        return res

    def __str__(self) -> str:
        res = {}
        for key, arg in self.__dict__.items():
            if not isinstance(arg, ArgSpecHelper):
                continue

            res[key] = arg.value

        return str(res)


class CommonArguments(ArgumentsBase):
    def __init__(self) -> None:
        super(CommonArguments, self).__init__()

        self.config_file = ArgSpecHelper(
            name="--config-file",
            type=str,
            default=None,
            help="Configue more complex parameters for executing the model.",
        )
        self.generate_config = ArgSpecHelper(
            name="--generate-config", action="store_true", help="Generate  config file"
        )
        self.output = ArgSpecHelper(
            name="--output",
            abbr_name="-o",
            default="hhb_out",
            help="The directory that holds the outputs.",
        )
        self.trace = ArgSpecHelper(
            name="--trace",
            choices=["relay", "qnn", "csinn", "csinn_acc"],
            nargs="+",
            help="Generate unify tracing data.",
        )

    @property
    def name(self):
        return "common"


class ImportArguments(ArgumentsBase):
    def __init__(self) -> None:
        super(ImportArguments, self).__init__()
        self.input_name = ArgSpecHelper(
            name="--input-name",
            abbr_name="-in",
            type=str,
            default=None,
            help="Set the name of input node. If '--input-name'is None, "
            "default value is 'Placeholder'. Multiple values "
            "are separated by semicolon(;).",
        )
        self.input_shape = ArgSpecHelper(
            name="--input-shape",
            abbr_name="-is",
            type=str,
            default=None,
            help="Set the shape of input nodes. Multiple shapes are separated "
            "by semicolon(;) and the dims between shape are separated "
            "by space.",
        )
        self.output_name = ArgSpecHelper(
            name="--output-name",
            abbr_name="-on",
            type=str,
            help="Set the name of output nodes. Multiple shapes are " "separated by semicolon(;).",
        )

        from .frontend_manage import get_frontend_names

        self.model_format = ArgSpecHelper(
            name="--model-format",
            choices=get_frontend_names(),
            help="Specify input model format:{}".format(get_frontend_names()),
        )
        self.reorder_pixel_format = ArgSpecHelper(
            name="--reorder-pixel-format",
            action="store_true",
            help="If original model's input data pixel format is rgb, then covert it to bgr;"
            "otherwise, then convert it to rgb.",
        )

    @property
    def name(self):
        return "import"


class OptimizeArguments(ArgumentsBase):
    def __init__(self) -> None:
        super(OptimizeArguments, self).__init__()
        self.board = ArgSpecHelper(
            name="--board",
            default="unset",
            choices=[
                "th1520",
                "e907",
                "c906",
                "c908",
                "c920",
                "c920v2",
                "rvm",
                "x86_ref",
                "unset",
            ],
            help="Set target device, default is unset.",
        )
        self.opt_level = ArgSpecHelper(
            name="--opt-level",
            choices=[-1, 0, 1, 2, 3],
            default=0,
            type=int,
            help="Specify the optimization level, default is 0.",
            hidden=True,
        )
        self.hybrid_computing = ArgSpecHelper(
            name="--hybrid-computing",
            action="store_true",
            help="Supports hybrid computing on multiple devices. Currently supports "
            "CPU and NPU for th1520.",
        )
        self.link_lib = ArgSpecHelper(
            name="--link-lib",
            default="unset",
            choices=[
                "shl_pnna",
                "shl_th1520",
                "shl_c906",
                "shl_c908",
                "shl_c920",
                "shl_rvm",
                "unset",
            ],
            help="Set link library for -D. default is unset",
        )

    @property
    def name(self):
        return "optimize"


class QuantizeArguments(ArgumentsBase):
    def __init__(self) -> None:
        super(QuantizeArguments, self).__init__()

        self.calibrate_dataset = ArgSpecHelper(
            name="--calibrate-dataset",
            abbr_name="-cd",
            type=str,
            help="Provide with dataset for the input of model in reference step. "
            "Support dir or .npz .jpg .png .JPEG or .txt in which there are path "
            "of images. Note: only one image path in one line if .txt.",
        )
        self.quantization_scheme = ArgSpecHelper(
            name="--quantization-scheme",
            abbr_name="-qs",
            choices=[
                "q4_0",
                "q8_0",
                "int4_asym_w_sym",
                "uint8_asym",
                "int8_asym",
                "int8_sym",
                "int8_original",
                "int8_asym_w_sym",
                "int16_sym",
                "float16",
                "float16_w_int8",
                "float32",
                "unset",
            ],
            default="unset",
            help="Scheme of quantization. default is unset.",
        )
        self.auto_hybrid_quantization = ArgSpecHelper(
            name="--auto-hybrid-quantization",
            action="store_true",
            help="If set, quantize model automatically.",
        )
        self.quantization_loss_algorithm = ArgSpecHelper(
            name="--quantization-loss-algorithm",
            type=str,
            choices=["cos_similarity", "mse", "kl_divergence", "cross_entropy", "gini"],
            default="cos_similarity",
            help="How to calculate accuracy loss for every layer.",
        )
        self.quantization_loss_threshold = ArgSpecHelper(
            name="--quantization-loss-threshold",
            type=float,
            default=0.99,
            help="The threshold that will determin thich layer will be quantized with hybrid way."
            "If it is None, we will select threshold automatically.",
        )
        self.dump_quantization_loss = ArgSpecHelper(
            name="--dump-quantization-loss",
            action="store_true",
            help="If set, dump quantizaiton loss into file.",
        )
        self.loss_threshold_type = ArgSpecHelper(
            name="--loss-threshold-type",
            type=str,
            choices=["min", "max", "avg"],
            default="avg",
            help="The threshold type for quantizaiton loss, default is avg.",
            hidden=True,
        )
        self.from_quant_file = ArgSpecHelper(
            name="--from-quant-file",
            action="store_true",
            help="If set, get hybrid quantization infomation from model.quant.json",
            hidden=True,
        )
        self.ignore_hybrid_layer = ArgSpecHelper(
            name="--ignore-hybrid-layer",
            type=str,
            nargs="+",
            help="Ignore layers to be hybrid layers.",
            hidden=True,
        )
        self.hybrid_quantization_scheme = ArgSpecHelper(
            name="--hybrid-quantization-scheme",
            choices=[
                "int4_asym_w_sym",
                "uint8_asym",
                "int8_asym",
                "int8_sym",
                "int8_asym_w_sym",
                "int16_sym",
                "float16",
                "float32",
                "unset",
            ],
            default="unset",
            help="Scheme of hybrid quantization. default is unset.",
        )
        self.hybrid_layer_name = ArgSpecHelper(
            name="--hybrid-layer-name",
            type=str,
            nargs="+",
            help="Layer buffer name to use hybrid quantization.",
        )
        self.num_bit_activation = ArgSpecHelper(
            name="--num-bit-activation",
            type=int,
            choices=[0, 16, 32],
            default=0,
            help="The bit number that quantizes activation layer. default is 0, means unset.",
            hidden=True,
        )
        self.num_bit_input = ArgSpecHelper(
            name="--num-bit-input",
            type=int,
            choices=[0, 8, 16, 32],
            default=0,
            help="The bit number that quantizes input layer. default is 0, means unset.",
            hidden=True,
        )
        self.num_bit_weight = ArgSpecHelper(
            name="--num-bit-weight",
            type=int,
            choices=[0, 8, 16, 32],
            default=0,
            help="The bit number that quantizes weight layer. default is 0, means unset.",
            hidden=True,
        )
        self.dtype_input = ArgSpecHelper(
            name="--dtype-input",
            choices=["float", "int", "uint", "unset"],
            default="unset",
            help="The dtype of quantized input layer, default is uint.",
            hidden=True,
        )
        self.dtype_weight = ArgSpecHelper(
            name="--dtype-weight",
            choices=["float", "int", "uint", "unset"],
            default="unset",
            help="The dtype of quantized constant parameters, default is uint.",
            hidden=True,
        )
        self.dtype_activation = ArgSpecHelper(
            name="--dtype-activation",
            choices=["float", "int", "uint", "unset"],
            default="unset",
            help="The dtype of quantized activation layer, default is uint.",
            hidden=True,
        )
        self.calibrate_mode = ArgSpecHelper(
            name="--calibrate-mode",
            choices=[
                "maxmin",
                "pow2",
                "kl_divergence",
                "kl_divergence_tsing",
                "scale",
                "percentile",
            ],
            default="maxmin",
            help="How to calibrate while doing quantization, default is maxmin.",
            hidden=True,
        )
        self.low_bound_scale = ArgSpecHelper(
            name="--low-bound-scale",
            type=float,
            default=1,
            help="Enlarge the low bound of the data type of specified quantization mode to "
            "avoid overflow during calculation. It should be >=1",
        )
        self.high_bound_scale = ArgSpecHelper(
            name="--high-bound-scale",
            type=float,
            default=1,
            help="Reduce the high bound of the data type of specified quantization mode to "
            "avoid overflow during calculation. It should be in (0, 1]",
        )
        self.activate_quantized_type = ArgSpecHelper(
            name="--activate-quantized-type",
            choices=["asym", "sym", "unset"],
            default="unset",
            help="Select the algorithm of quantization, default is asym.",
            hidden=True,
        )
        self.weight_quantized_type = ArgSpecHelper(
            name="--weight-quantized-type",
            choices=["asym", "sym", "unset"],
            default="unset",
            help="Select the algorithm of quantization, default is asym.",
            hidden=True,
        )
        self.weight_scale = ArgSpecHelper(
            name="--weight-scale",
            choices=["max", "power2"],
            default="max",
            help="Select the mothod that quantizes weight value, default is max.",
            hidden=True,
        )
        self.fuse_conv_relu = ArgSpecHelper(
            name="--fuse-conv-relu",
            action="store_true",
            help="Fuse the convolution and relu layer.",
        )
        self.fuse_reshape_dense = ArgSpecHelper(
            name="--fuse-reshape-dense",
            action="store_true",
            help="Fuse the reshape and dense layer.",
        )
        self.channel_quantization = ArgSpecHelper(
            name="--channel-quantization",
            action="store_true",
            help="Do quantizetion across channel.",
            hidden=True,
        )
        self.broadcast_quantization = ArgSpecHelper(
            name="--broadcast-quantization",
            action="store_true",
            help="Broadcast quantization parameters for special ops.",
        )
        self.channel_quantization_ratio_threshold = ArgSpecHelper(
            name="--channel-quantization-ratio-threshold",
            type=float,
            default=0.0,
            help="Set optimize quantitative parameters threshold for channel quantization."
            + "The value can be selected in range [0, 1). 0 means this pass is not be used."
            + "0.3 is a recommended threshold if this pass is turned on.",
            hidden=True,
        )
        self.fuse_clip = ArgSpecHelper(
            name="--fuse-clip",
            action="store_true",
            help="Fuse clip's attr into pre layer's quantitative information. "
            + "This flag is only valid when quantization is used.",
        )
        self.fuse_mul_before_conv = ArgSpecHelper(
            name="--fuse-mul-before-conv",
            default=True,
            choices=[False, True],
            type=boolean_string,
            help="This parameter is used to merge mul before conv2d to conv2d.",
            hidden=True,
        )
        self.fuse_mul_after_conv = ArgSpecHelper(
            name="--fuse-mul-after-conv",
            default=True,
            choices=[False, True],
            type=boolean_string,
            help="This parameter is used to merge mul after conv2d to conv2d.",
            hidden=True,
        )
        self.fuse_add_before_conv = ArgSpecHelper(
            name="--fuse-add-before-conv",
            default=True,
            choices=[False, True],
            type=boolean_string,
            help="This parameter is used to merge add before conv2d to conv2d.",
            hidden=True,
        )
        self.fuse_add_after_conv = ArgSpecHelper(
            name="--fuse-add-after-conv",
            default=True,
            choices=[False, True],
            type=boolean_string,
            help="This parameter is used to merge add after conv2d to conv2d.",
            hidden=True,
        )
        self.fuse_zp2bias = ArgSpecHelper(
            name="--fuse-zp2bias",
            action="store_true",
            help="Merge conv2d/dense zp to bias.",
        )
        self.target_layout = ArgSpecHelper(
            name="--target-layout",
            default="NCHW",
            choices=["NCHW", "NHWC"],
            help="Set target layout.",
        )
        self.use_custom_fusion = ArgSpecHelper(
            name="--use-custom-fusion",
            action="store_true",
            help="This parameter is used to use custom fusion like fuse cache_matmul and layer norm.",
            hidden=True,
        )
        self.convert_to_relay = ArgSpecHelper(
            name="--convert-to-relay",
            action="store_true",
            help="This parameter is used to convert quanted qnn model to relay.",
            hidden=True,
        )
        self.quantization_tool = ArgSpecHelper(
            name="--quantization-tool",
            choices=["default", "ppq"],
            default="default",
            help="Model quantization tool, default is hhb.",
        )
        self.lsq = ArgSpecHelper(
            name="--lsq",
            action="store_true",
            help="Whether uses lsq quantization algorithm.",
        )
        self.lsq_lr = ArgSpecHelper(
            name="--lsq-lr",
            type=float,
            default=1e-5,
            help="Initial learning rate for lsq.",
        )
        self.lsq_steps = ArgSpecHelper(
            name="--lsq-steps",
            type=int,
            default=500,
            help="Training steps for lsq.",
        )
        self.quant_device = ArgSpecHelper(
            name="--quant-device",
            choices=["cpu", "cuda"],
            default="cpu",
            help="The device to quantize model, defaults to cpu.",
        )
        self.cali_batch = ArgSpecHelper(
            name="--cali-batch",
            type=int,
            default=16,
            help="The batch size for calibration.",
        )
        self.align_elementwise = ArgSpecHelper(
            name="--align-elementwise",
            choices=["None", "Align to Large"],
            default="Align to Large",
            help="The method that aligns the inputs quant config of elementwise ops.",
        )

    @property
    def name(self):
        return "quantize"


class HardwareArguments(ArgumentsBase):
    def __init__(self) -> None:
        super(HardwareArguments, self).__init__()

        self.hardware_sram_size = ArgSpecHelper(
            name="--hardware-sram-size",
            type=str,
            default=None,
            help="Set the size of sram. The unit must in [m, M, KB, kb].",
            hidden=True,
        )
        self.hardware_max_groups = ArgSpecHelper(
            name="--hardware-max-groups",
            type=int,
            default=0,
            help="This parameter is used to describe the maximum number of groups supported by hardware.",
            hidden=True,
        )
        self.hardware_max_out_channel = ArgSpecHelper(
            name="--hardware-max-out-channel",
            type=int,
            default=0,
            help="This parameter is used to describe the maximum number of output channel supported by hardware.",
            hidden=True,
        )
        self.hardware_max_kernel_size = ArgSpecHelper(
            name="--hardware-max-kernel-size",
            type=int,
            default=0,
            help="This parameter is used to describe the size of sram.",
            hidden=True,
        )
        self.hardware_contain_weight = ArgSpecHelper(
            name="--hardware-contain-weight",
            action="store_true",
            help="This parameter is used to describe whether the weight size should be contained when split ops.",
            hidden=True,
        )
        self.hardware_alignment = ArgSpecHelper(
            name="--hardware-alignment",
            type=int,
            choices=[1, 8, 16, 32],
            default=1,
            help="This parameter describes whether the hardware requires data alignment.",
            hidden=True,
        )
        self.matrix_extension_mlen = ArgSpecHelper(
            name="--matrix-extension-mlen",
            default=0,
            type=int,
            help="Specify XuanTie Matrix extension's MLEN bit, default is 0, unuse matrix extension.",
        )

    @property
    def name(self):
        return "hardware"


class SimulateArguments(ArgumentsBase):
    def __init__(self) -> None:
        super(SimulateArguments, self).__init__()

        self.simulate_data = ArgSpecHelper(
            name="--simulate-data",
            abbr_name="-sd",
            type=str,
            default=None,
            help="Provide with dataset for the input of model in reference step. "
            "Support dir or .npz .jpg .png .JPEG or .txt in which there are path "
            "of images. Note: only one image path in one line if .txt.",
        )

    @property
    def name(self):
        return "simulate"


class PostprocessArguments(ArgumentsBase):
    def __init__(self) -> None:
        super(PostprocessArguments, self).__init__()

        self.postprocess = ArgSpecHelper(
            name="--postprocess",
            type=str,
            default="top5",
            choices=["top5", "save", "save_and_top5"],
            help="Set the mode of postprocess: "
            "'top5' show top5 of output; "
            "'save' save output to file;"
            "'save_and_top5' show top5 and save output to file."
            " Default is top5",
        )
        self.show_session_run_time = ArgSpecHelper(
            name="--show-session-run-time",
            action="store_true",
            help="If set, print session run time.",
        )

    @property
    def name(self):
        return "postprocess"


class PreprocessArguments(ArgumentsBase):
    def __init__(self) -> None:
        super(PreprocessArguments, self).__init__()

        self.data_mean = ArgSpecHelper(
            name="--data-mean",
            abbr_name="-m",
            type=str,
            default="0",
            help="Set the mean value of input, multiple values are separated by space, "
            "default is 0.",
        )
        self.data_scale = ArgSpecHelper(
            name="--data-scale",
            abbr_name="-s",
            type=float,
            default=1,
            help="Scale number(mul) for inputs normalization(data=img*scale), default is 1.",
        )
        self.data_scale_div = ArgSpecHelper(
            name="--data-scale-div",
            abbr_name="-sv",
            type=float,
            default=1,
            help="Scale number(div) for inputs normalization(data=img/scale), default is 1.",
        )
        self.add_exclusive_group(self.data_scale, self.data_scale_div)
        self.data_resize = ArgSpecHelper(
            name="--data-resize",
            abbr_name="-r",
            type=int,
            default=None,
            help="Resize base size for input image to resize.",
        )
        self.pixel_format = ArgSpecHelper(
            name="--pixel-format",
            choices=["RGB", "BGR"],
            default="RGB",
            help="The pixel format of image data, defalut is RGB",
        )
        self.data_layout = ArgSpecHelper(
            name="--data-layout",
            choices=["NCHW", "NHWC", "CHW", "HWC", "Any"],
            default="NCHW",
            help='The input data layout, defalut is "NCHW"',
            hidden=True,
        )
        self.add_preprocess_node = ArgSpecHelper(
            name="--add-preprocess-node", action="store_true", help="Add preprocess node for model."
        )

    @property
    def name(self):
        return "preprocess"


class CodegenArguments(ArgumentsBase):
    def __init__(self) -> None:
        super(CodegenArguments, self).__init__()

        self.model_save = ArgSpecHelper(
            name="--model-save",
            choices=["run_only", "save_only", "save_and_run"],
            default="run_only",
            help="Whether save binary graph or run only.\n"
            "run_only: execute model only, not save binary graph.\n"
            "save_only: save binary graph only.\n"
            "save_and_run: execute and save model.",
        )
        self.model_priority = ArgSpecHelper(
            name="--model-priority",
            default=0,
            type=int,
            help="Set model priority, only for th1520 now.\n"
            "0 is lowest, 1 is medium, 2 is highest.",
        )
        self.without_preprocess = ArgSpecHelper(
            name="--without-preprocess",
            action="store_true",
            help="Do not generate preprocess codes.",
        )
        self.trace_strategy = ArgSpecHelper(
            name="--trace-strategy",
            choices=["normal", "advanced"],
            default="normal",
            help="Strategy to generate trace data.",
            hidden=True,
        )
        self.input_memory_type = ArgSpecHelper(
            name="--input-memory-type",
            choices=[0, 1, 2],
            type=int,
            nargs="+",
            help="Set the memory type for input tensor, support for multi-values.\n"
            "0: allocated by CPU and not aligned;\n"
            "1: allocated by CPU and aligned;\n"
            "2: dma buffer.",
        )
        self.output_memory_type = ArgSpecHelper(
            name="--output-memory-type",
            choices=[0, 1, 2],
            type=int,
            nargs="+",
            help="Set the memory type for output tensor, support for multi-values.\n"
            "0: allocated by CPU and not aligned;\n"
            "1: allocated by CPU and aligned;\n"
            "2: dma buffer.",
        )
        self.memory_type = ArgSpecHelper(
            name="--memory-type",
            choices=[0, 1, 2],
            type=int,
            help="Set the memory type for input and output tensors.\n"
            "0: allocated by CPU and not aligned;\n"
            "1: allocated by CPU and aligned;\n"
            "2: dma buffer.",
        )
        self.conv2d_algorithm = ArgSpecHelper(
            name="--conv2d-algorithm",
            choices=["direct", "winograd", "gemm", "unset"],
            default="unset",
            help="The recommend algorithm for conv2d implementation, default is unset.",
            hidden=True,
        )
        self.th1520_input_fix_size = ArgSpecHelper(
            name="--th1520-input-fix-size",
            type=str,
            default="0",
            help="Set input stride.",
            hidden=True,
        )
        self.ahead_of_time = ArgSpecHelper(
            name="--ahead-of-time",
            choices=["intrinsic", "unset"],
            default="unset",
            help="AOT to generate intrinsic.",
        )
        self.dynamic_shape = ArgSpecHelper(
            name="--dynamic-shape", action="store_true", help="If set, don't quantize the model."
        )
        self.with_makefile = ArgSpecHelper(
            name="--with-makefile",
            action="store_true",
            help="Generate corresponding makefile.",
            hidden=True,
        )
        self.with_py_wrapper = ArgSpecHelper(
            name="--with-py-wrapper",
            action="store_true",
            help="Generate python wrapper for model inference.",
            hidden=True,
        )

    @property
    def name(self):
        return "codegen"


class ProfilerArguments(ArgumentsBase):
    def __init__(self) -> None:
        super(ProfilerArguments, self).__init__()

        self.ir_type = ArgSpecHelper(
            name="--ir-type",
            choices=[
                "relay",
            ],
            default="relay",
            help="The ir type that will be profiled, default is relay",
        )
        self.indicator = ArgSpecHelper(
            name="--indicator",
            choices=["cal", "mem", "all"],
            default="cal",
            nargs="+",
            help="Select indicator to profile, default is cal(calculation).\n"
            "cal: calculation, how many operations to be executed for current op.\n"
            "mem: memory, how many memory to be used for current op.\n"
            "all: include all indicators above.",
        )
        self.output_type = ArgSpecHelper(
            name="--output-type",
            choices=["json", "print", "total", "all", "csv"],
            default="total",
            nargs="+",
            help="How to show results, default is show summary result.",
        )
        self.arch = ArgSpecHelper(
            name="--arch",
            choices=[
                "relay",
                "qnn",
                "npuperf",
            ],
            type=str,
            help="Target architecture for profiling.",
        )
        self.arch_config = ArgSpecHelper(
            name="--arch-config",
            type=str,
            help="Configurable parameters for specified architecture. Supported file type: .json.",
        )
        self.merge_trace = ArgSpecHelper(
            name="--merge-trace",
            action="store_true",
            help="Merge multi trace files into one file.",
        )
        self.topk = ArgSpecHelper(
            name="--topk",
            type=int,
            default=10,
            help="Show top k results.",
        )
        self.profile_method = ArgSpecHelper(
            name="--profile-method",
            choices=[
                "events_by_group",
                "events_all",
                "kernel_by_group",
                "kernel_all",
                "sorted_by_macc",
                "sorted_by_flops",
                "sorted_by_total_mem",
                "accuracy_loss",
            ],
            nargs="+",
            help="How to profile trace data, support for multi-values(seperate by space).",
        )

    @property
    def name(self):
        return "profiler"


class MainArguments(ArgumentsBase):
    def __init__(self) -> None:
        super(MainArguments, self).__init__()

        self.verbose = ArgSpecHelper(
            name="--verbose",
            abbr_name="-v",
            action="count",
            default=0,
            help="Increase verbosity",
        )
        self.help = ArgSpecHelper(
            name="--help",
            abbr_name="-h",
            action="store_true",
            help="Show this help information",
        )
        self.version = ArgSpecHelper(
            name="--version",
            action="version",
            version="{}\n".format(hhb_version()),
            help="Print the version and exit",
        )
        self.E = ArgSpecHelper(name="-E", action="store_true", help="Convert model into relay ir.")
        self.Q = ArgSpecHelper(name="-Q", action="store_true", help="Quantize the relay ir.")
        self.C = ArgSpecHelper(name="-C", action="store_true", help="codegen the model.")
        self.D = ArgSpecHelper(name="-D", action="store_true", help="deploy on platform.")
        self.S = ArgSpecHelper(name="-S", action="store_true", help="run elf to simulate.")
        self.simulate = ArgSpecHelper(
            name="--simulate", action="store_true", help="Simulate model on x86 device."
        )
        self.add_exclusive_group(self.E, self.Q, self.C, self.D, self.S, self.simulate)

        self.no_quantize = ArgSpecHelper(
            name="--no-quantize", action="store_true", help="If set, don't quantize the model."
        )
        self.model_file = ArgSpecHelper(
            name="--model-file",
            abbr_name="-f",
            nargs="+",
            help="Path to the input model file, can pass multi files",
        )
        self.llm = ArgSpecHelper(
            name="--llm",
            action="store_true",
            help="LLM name is useful",
            hidden=True,
        )
        self.save_temps = ArgSpecHelper(
            name="--save-temps", action="store_true", help="Save temp files."
        )
        self.generate_dataset = ArgSpecHelper(
            name="--generate-dataset",
            action="store_true",
            help="Generate dataset according to provided preprocess parameters.",
        )
        self.generate_readme = ArgSpecHelper(
            name="--generate-readme",
            action="store_true",
            help="Automatically generate README.md.",
            hidden=True,
        )

    @property
    def name(self):
        return "main"


class Config(object):
    def __init__(self, board="unset") -> None:
        self.common = CommonArguments()
        self.import_config = ImportArguments()
        self.optimize = OptimizeArguments()
        self.quantize = QuantizeArguments()
        self.hardware = HardwareArguments()
        self.simulate = SimulateArguments()
        self.postprocess = PostprocessArguments()
        self.preprocess = PreprocessArguments()
        self.codegen = CodegenArguments()
        self.profile = ProfilerArguments()
        self.main = MainArguments()

        self.set_config_by_board(board)

        self._cmd_config = None

    def convert_to_arguments(self) -> AttributeDict:
        args = AttributeDict()
        for v in self.__dict__.values():
            if isinstance(v, ArgumentsBase):
                for name, value in v.__dict__.items():
                    if isinstance(value, ArgSpecHelper):
                        args[name] = value.value
        return args

    def update_config_from_module(self, hhb_ir):
        mod, params = hhb_ir.get_model()
        input_name_list, input_shape_list, _ = get_input_info_from_relay(mod, params)
        output_shape_list, _ = get_output_info_from_relay(mod, params)

        # if self.import_config.input_name.value is None:
        self.import_config.input_name.value = input_name_list
        self.import_config.input_shape.value = input_shape_list
        if self.import_config.output_name.value is None:
            self.import_config.output_name.value = []
            for i in range(len(output_shape_list)):
                self.import_config.output_name.value.append("output" + str(i))

    def update_config_from_file(self, path: str):
        """Update arguments values in config.

        The specified json file can be generated by generate_hhb_default_config().
        """
        # init ARGS_DEST_TO_OPTIONS_STRING
        get_default_config()

        # config from specified file
        file_config = {}
        if path and os.path.isfile(path):
            with open(path, "r") as f:
                file_config = json.load(f)
        if not file_config:
            return

        for item in file_config.values():
            for dest, value in item.items():
                optstr = tuple(ARGS_DEST_TO_OPTIONS_STRING[dest])

                for arg in self.__dict__.values():
                    if not isinstance(arg, ArgumentsBase):
                        continue
                    optstr_to_arg = arg.get_option_string_to_argument()
                    if optstr in optstr_to_arg:
                        optstr_to_arg[optstr].value = value

    def get_all_arguments_info(self):
        if not ARGS_DEST_TO_OPTIONS_STRING:
            raise HHBException(
                "Please generate command line arguments by generate_cmd_config() first."
            )

        optinon_string_to_dest = {}
        for k, v in ARGS_DEST_TO_OPTIONS_STRING.items():
            optinon_string_to_dest[tuple(v)] = k

        res = {}
        for arg in self.__dict__.values():
            if not isinstance(arg, ArgumentsBase):
                continue
            res[arg.name] = {}
            opt_str_to_arg = arg.get_option_string_to_argument()
            for k, v in opt_str_to_arg.items():
                if k in optinon_string_to_dest:
                    dest_name = optinon_string_to_dest[k]
                    res[arg.name][dest_name] = v.value

        return res

    def set_config_by_board(self, board: str):
        if board not in self.optimize.board.spec.choices:
            raise HHBException(
                "Specified board: {} is not in {}".format(board, self.optimize.board.spec.choices)
            )
        self.optimize.board.value = board

        if board in ("x86_ref", "th1520"):
            self.quantize.quantization_scheme.value = "int8_asym"
        elif board in ("c906", "c920"):
            self.quantize.quantization_scheme.value = "float16"
        elif board in ("e907", "c908", "c920v2", "rvm"):
            self.quantize.quantization_scheme.value = "int8_asym_w_sym"
        else:
            self.quantize.quantization_scheme.value = "uint8_asym"

    def generate_cmd_config(self, with_io_info=True):
        """Genrate command line arguments."""

        def _update_config(orig_config, new_config):
            for sec, value in new_config.items():
                if sec in orig_config:
                    orig_config[sec].update(value)

        # This is the base config for cmd backend.
        base_config, _ = get_default_config()

        # The arguments that are set by user with Config obj
        # It should overwrite the base config first.
        obj_config = self.get_all_arguments_info()
        _update_config(base_config, obj_config)

        # unroll arguments
        unroll_config = AttributeDict()
        for value in base_config.values():
            unroll_config.update(value)

        extra_args = AttributeDict()
        all_filters = []
        if with_io_info:
            all_filters = [
                collect_preprocess_config,
                set_preprocess_params,
                collect_quantization_config,
                set_quantize_params_by_board,
                collect_codegen_config,
                set_codegen_config,
                parse_sram_size,
            ]
            if (
                self.import_config.input_shape.value is None
                or self.import_config.output_name.value is None
            ):
                raise HHBException(
                    "There is no input/output info, please set them directly or execute "
                    "update_config_from_module() firstly."
                )

            extra_args.input_shape = self.import_config.input_shape.value
            extra_args.input_num = len(extra_args.input_shape)
            extra_args.output_num = len(self.import_config.output_name.value)
        args_filter = ArgumentFilter(unroll_config)
        args_filter.filtered_args = unroll_config
        args_filter.filter_argument(all_filters, extra=extra_args)

        self._cmd_config = args_filter.filtered_args


@get_parameters_info(CommonArguments().name)
def add_common_argument(parser):
    """All common parameters"""
    CommonArguments().set_parser(parser)


@get_parameters_info(PreprocessArguments().name)
def add_preprocess_argument(parser):
    """All preprocess parameters"""
    PreprocessArguments().set_parser(parser)


@get_parameters_info(ImportArguments().name)
def add_import_argument(parser):
    """All parameters needed by 'import' subcommand"""
    ImportArguments().set_parser(parser)


@get_parameters_info(OptimizeArguments().name)
def add_optimize_argument(parser):
    """All parameters needed by optimization"""
    OptimizeArguments().set_parser(parser)


@get_parameters_info(QuantizeArguments().name)
def add_quantize_argument(parser):
    """All parameters needed by 'quantize' subcommand"""
    QuantizeArguments().set_parser(parser)


@get_parameters_info(HardwareArguments().name)
def add_hardware_argument(parser):
    """All hardware parameters"""
    HardwareArguments().set_parser(parser)


@get_parameters_info(SimulateArguments().name)
def add_simulate_argument(parser):
    """All parameters needed by 'simulate' subcommand"""
    SimulateArguments().set_parser(parser)


@get_parameters_info(PostprocessArguments().name)
def add_postprocess_argument(parser):
    """All postprocess parameters"""
    PostprocessArguments().set_parser(parser)


@get_parameters_info(MainArguments().name)
def add_main_argument(parser):
    """All commands that are compatible with previous version."""
    MainArguments().set_parser(parser)


@get_parameters_info(CodegenArguments().name)
def add_codegen_argument(parser):
    """All codegen parameters"""
    CodegenArguments().set_parser(parser)


@get_parameters_info(ProfilerArguments().name)
def add_profiler_argument(parser):
    """All profiler parameters"""
    ProfilerArguments().set_parser(parser)


@argument_filter_helper
def parse_node_name(filtered_args, extra=None):
    """
    Convert "input1;input2;input3..." into ["input1", "input2", "input3", ...]

    The name may be include multi name which is separated by semicolon(;), and this
    function will convert name to list.

    Parameters
    ----------
    filtered_args : AttributeDict
        filtered args
    """

    def convert_str2list(name):
        if not name:
            return list()
        if isinstance(name, (list, tuple)):
            return list(name)
        name_list = name.strip().split(";")
        name_list = list([n for n in name_list if n])
        name_list = [n.strip() for n in name_list]
        return list(name_list)

    if hasattr(filtered_args, "input_name"):
        filtered_args["input_name"] = convert_str2list(filtered_args["input_name"])
    if hasattr(filtered_args, "output_name"):
        filtered_args["output_name"] = convert_str2list(filtered_args["output_name"])


@argument_filter_helper
def parse_node_shape(filtered_args, extra=None):
    """
    Convert "1 3 224 224;1 3 256 256; ..." into [[1, 3, 224, 224], [1, 3, 256, 256] ...]

    There may be include multi shapes which is separated by semicolon(;), and this
    function will convert shape to list.

    Parameters
    ----------
    filtered_args : AttributeDict
        filtered args
    """

    def convert_str2list(shape):
        if not shape:
            return list()
        if isinstance(shape, (list, tuple)):
            return list(shape)
        if "," in shape:
            shape = shape.replace(",", " ")
        shape_list = []
        shape_str_list = shape.strip().split(";")
        shape_str_list = list([n for n in shape_str_list if n])
        for shape_str in shape_str_list:
            tmp_list = shape_str.strip().split(" ")
            tmp_list = [int(i) for i in tmp_list if i]
            shape_list.append(tmp_list)
        return shape_list

    if hasattr(filtered_args, "input_shape"):
        filtered_args["input_shape"] = convert_str2list(filtered_args["input_shape"])


@argument_filter_helper
def parse_sram_size(filtered_args, extra=None):
    """
    Convert "1M" to 2**20, "1KB" to 1024

    There may contain uppercase or lowercase unit, and this
    function will convert txt abbreviation to number.

    Parameters
    ----------
    filtered_args : AttributeDict
        filtered args
    """
    size_map = {
        "kb": 2**10,
        "m": 2**20,
    }

    def convert_str2int(txt_size):
        if isinstance(txt_size, int):
            return txt_size
        if txt_size:
            txt_size = txt_size.lower()
            if "kb" in txt_size:
                num = txt_size[: txt_size.find("kb")]
                sram_size = float(num) * size_map["kb"]
            elif "m" in txt_size:
                num = txt_size[: txt_size.find("m")]
                sram_size = float(num) * size_map["m"]
            else:
                raise Exception(
                    "Input argument --h-sram-size is invalid. The unit must in [m, M, KB, kb]."
                )

            return int(sram_size)
        return 0

    if hasattr(filtered_args, "hardware_sram_size"):
        filtered_args["hardware_sram_size"] = convert_str2int(filtered_args["hardware_sram_size"])


def update_arguments_by_file(args, origin_args):
    def _check_cmd_args(name, cmd_args):
        option_string = (
            ARGS_DEST_TO_OPTIONS_STRING[name] if name in ARGS_DEST_TO_OPTIONS_STRING else None
        )
        if option_string is None:
            return False
        else:
            for item in option_string:
                if item in cmd_args:
                    return True
            return False

    if not os.path.exists(args.config_file):
        hhb_exit("File does not exist: {}".format(args.config_file))
    with open(args.config_file, "r") as f:
        params_dict = yaml.safe_load(f.read())

    for name, value in params_dict.items():
        for arg_name, arg_value in value.items():
            if hasattr(args, arg_name) and not _check_cmd_args(arg_name, origin_args):
                args.__dict__[arg_name] = arg_value

            ALL_ARGUMENTS_INFO[name][arg_name] = arg_value


class ArgumentFilter(object):
    def __init__(self, args):
        self.origin_args = args  # parsed by argparse
        if isinstance(self.origin_args, argparse.Namespace):
            self.filtered_args = AttributeDict(**(self.origin_args.__dict__))
        else:
            self.filtered_args = AttributeDict(**self.origin_args)

        self._built_in_filter()

    def _built_in_filter(self):
        # do some basic filtering
        _func_handle_seq = [parse_node_name, parse_node_shape, parse_sram_size]

        self.filter_argument(_func_handle_seq)

    def filter_argument(self, func_handle_seq, extra=None):
        """Filter arguments help handler

        Parameters
        ----------
        func_handle_seq: List[function]
            A list of filter function, which will be executed one by one

        extra: Optional[AttributeDict]
            Extra arguments what will be used
        """
        if not func_handle_seq:
            return
        if not isinstance(func_handle_seq, (list, tuple)):
            raise HHBException(
                "Unsupport for type: {}, should be List[function] or Tuple[function]".format(
                    type(func_handle_seq)
                )
            )
        for func in func_handle_seq:
            func(self.filtered_args, extra=extra)


class CommandType(object):
    """Denote the kind of command mode"""

    MAINCOMMAND = 0
    SUBCOMMAND = 1


class ArgumentManage(object):
    def __init__(self, argv, args=None):
        self.origin_argv = argv  # obtained from command line

        self.have_help = self._check_help()
        if self.have_help:
            if "-h" in self.origin_argv:
                self.origin_argv.remove("-h")
            if "--help" in self.origin_argv:
                self.origin_argv.remove("--help")

        self.have_generate_readme = "--generate-readme" in self.origin_argv

    def _check_help(self):
        res = False
        if "-h" in self.origin_argv or "--help" in self.origin_argv:
            res = True
        return res

    def get_command_type(self) -> CommandType:
        """Determine the type of command line parameters

        Returns
        -------
        res : CommandType
            main comamnd(denoted by CommandType.MAINCOMMAND) or subcommand(denoted by CommandType.SUBCOMMAND)
        """
        res = CommandType.MAINCOMMAND
        for sub in ALL_SUBCOMMAND:
            if sub in self.origin_argv:
                res = CommandType.SUBCOMMAND
                break
        return res

    def check_cmd_arguments(self):
        """Check whether there are any other arguments before subcommand"""
        for sub in ALL_SUBCOMMAND:
            if sub in self.origin_argv:
                idx = self.origin_argv.index(sub)
                if idx != 1:
                    raise HHBException("do not allow any arguments before subcommand...")

        # "-sd" conflict with "-s"
        if "-sd" in self.origin_argv:
            idx = self.origin_argv.index("-sd")
            self.origin_argv[idx] = "--simulate-data"

    def set_main_command(self, parser):
        if not isinstance(parser, argparse.ArgumentParser):
            raise HHBException("invalid parser:{}".format(parser))
        add_main_argument(parser)
        add_import_argument(parser)
        add_optimize_argument(parser)
        add_quantize_argument(parser)
        add_hardware_argument(parser)
        add_preprocess_argument(parser)
        add_common_argument(parser)
        add_simulate_argument(parser)
        add_postprocess_argument(parser)
        add_codegen_argument(parser)

    def set_subcommand(self, parser):
        """Set parameters for subcommand"""
        if not isinstance(parser, argparse.ArgumentParser):
            raise HHBException("invalid parser:{}".format(parser))
        subparser = parser.add_subparsers(title="commands")
        for make_subparser in HHB_REGISTERED_PARSER:
            make_subparser(subparser)

    def print_help_info(self, parser):
        if not isinstance(parser, argparse.ArgumentParser):
            raise HHBException("invalid parser:{}".format(parser))
        self.origin_argv.append("-h")
        _ = parser.parse_args(self.origin_argv[1:])
        self.origin_argv.remove("-h")
        # self.set_subcommand(parser)
        all_valid_command = list([c for c in ALL_SUBCOMMAND if c not in HIDDEN_SUBCOMMAND])
        subparser = parser.add_subparsers(title="commands", metavar=str(set(all_valid_command)))
        for make_subparser in HHB_REGISTERED_PARSER:
            make_subparser(subparser)
        parser.print_help()

    def generate_readme(self, parser):
        if not isinstance(parser, argparse.ArgumentParser):
            raise HHBException("invalid parser:{}".format(parser))
        all_valid_command = list([c for c in ALL_SUBCOMMAND if c not in HIDDEN_SUBCOMMAND])
        subparser = parser.add_subparsers(title="commands", metavar=str(set(all_valid_command)))
        for make_subparser in HHB_REGISTERED_PARSER:
            make_subparser(subparser)

        # get output dir
        o_idx = -1
        if "-o" in self.origin_argv:
            o_idx = find_index(self.origin_argv, "-o")
        elif "--output" in self.origin_argv:
            o_idx = find_index(self.origin_argv, "--output")

        output_dir = "."
        if o_idx >= 0:
            output_dir = self.origin_argv[o_idx + 1]
        generate_readme_file(output_dir)

    def run_command(self, args_filter: ArgumentFilter, commad_type: CommandType):
        if not isinstance(args_filter, ArgumentFilter):
            raise HHBException("invalid ArgumentFilter:{}".format(args_filter))
        assert args_filter.filtered_args, "Error: empty parsed arguments."

        if commad_type == CommandType.SUBCOMMAND:
            assert hasattr(
                args_filter.filtered_args, "func"
            ), "Error: missing 'func' attribute for subcommand {}".format(args_filter.filtered_args)
        else:
            from .main_command_manage import driver_main_command

            args_filter.filtered_args.func = driver_main_command

        try:
            return args_filter.filtered_args.func(args_filter)
        except HHBException as err:
            sys.stderr.write("Error: {}".format(err))
            return 4


def get_default_config():
    arg_manage = ArgumentManage([])
    parser = argparse.ArgumentParser(prog="hhb_api", add_help=False)
    # add command line parameters
    arg_manage.set_main_command(parser)
    add_profiler_argument(parser)
    desc = collect_arguments_info(parser._actions)
    res = copy.deepcopy(ALL_ARGUMENTS_INFO)

    return res, desc


def generate_hhb_default_config(
    path="hhb_config_default.json", specified_groups=None, with_hidden=False
):
    """Generate defualt config.

    Parameters
    ----------
    path : str
        The path of file that holds default config.
    specified_groups : list or tuple
        Arguments in sepcified_groups are generated. Group name can be selected from
        ["common", "preprocess", "import", "optimize", "quantize", "hardware",
            "simulate", "postprocess", "main", "codegen", "profiler"]
    with_hidden : bool
        If it is true, more arguments will be generated in the config file.
    """
    res, desc = get_default_config()

    exclude_sections = [
        # "main",
        # "import",
        # "preprocess",
        # "common",
        # "postprocess",
        # "simulate",
    ]
    all_sections = ALL_ARGUMENTS_INFO.keys()
    if not specified_groups:
        specified_groups = list(all_sections)
    if not isinstance(specified_groups, (str, list, tuple)):
        raise HHBException("Unsupport for type: {}".format(type(specified_groups)))
    if isinstance(specified_groups, str):
        specified_groups = [
            specified_groups,
        ]
    inter_exclude = set(all_sections) - set(specified_groups)
    exclude_sections += list(inter_exclude)

    exclude_arguments = [
        # "opt_level",
        # "calibrate_dataset",
        # "config_file",
        # "generate_config"
    ]

    # ignore hidden arguments
    non_hidden_argument = []
    for item in desc:
        modify_item = item.name.strip().split(",")[-1].strip()
        if "--" not in modify_item:
            modify_item = modify_item.replace("-", "")
        else:
            modify_item = modify_item.replace("--", "")
            modify_item = modify_item.replace("-", "_")
        non_hidden_argument.append(modify_item)

    # delete some unused arguments
    for ses, value in ALL_ARGUMENTS_INFO.items():
        if ses in exclude_sections:
            res.pop(ses)
            continue
        for arg in value.keys():
            if arg in exclude_arguments:
                res[ses].pop(arg)
                continue
            if not with_hidden and arg not in non_hidden_argument:
                res[ses].pop(arg)
                continue

    with open(path, "w") as f:
        json.dump(res, f, indent=2)
