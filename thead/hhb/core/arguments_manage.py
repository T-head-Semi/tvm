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
from collections import OrderedDict
import os
import sys
import argparse

import yaml

from .common import (
    HHBException,
    HHB_REGISTERED_PARSER,
    hhb_version,
    AttributeDict,
    get_parameters_info,
    argument_filter_helper,
    ALL_ARGUMENTS_DESC,
    generate_readme_file,
    find_index,
    hhb_exit,
    ARGS_DEST_TO_OPTIONS_STRING,
    ALL_ARGUMENTS_INFO,
)


ALL_SUBCOMMAND = ["import", "quantize", "codegen", "simulate", "benchmark", "profiler"]
HIDDEN_SUBCOMMAND = ["benchmark"]


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


@get_parameters_info("common")
def add_common_argument(parser):
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        metavar="",
        help="Configue more complex parameters for executing the model.",
    )
    parser.add_argument("--generate-config", action="store_true", help="Generate  config file")


@get_parameters_info("preprocess")
def add_preprocess_argument(parser):
    """ All preprocess parameters"""
    parser.add_argument(
        "-m",
        "--data-mean",
        type=str,
        default="0",
        metavar="",
        help="Set the mean value of input, multiple values are separated by space, "
        "default is 0.",
    )
    parser.add_argument(
        "-s",
        "--data-scale",
        type=float,
        default="1",
        metavar="",
        help="Scale number for inputs normalization, default is 1.",
    )
    parser.add_argument(
        "-r",
        "--data-resize",
        type=int,
        default=None,
        metavar="",
        help="Resize base size for input image to resize.",
    )
    parser.add_argument(
        "--pixel-format",
        choices=["RGB", "BGR"],
        default="BGR",
        help="The pixel format of input data, defalut is BGR",
    )
    parser.add_argument(
        "--data-layout",
        choices=["NCHW", "NHWC", "CHW", "HWC", "Any"],
        default="NCHW",
        # help='The input data layout, defalut is "NCHW"',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--add-preprocess-node", action="store_true", help="Add preprocess node for model."
    )
    parser.add_argument(
        "--light-input-fix-size",
        type=str,
        default="0",
        metavar="",
        help=argparse.SUPPRESS,
    )


@get_parameters_info("import")
def add_import_argument(parser):
    """All parameters needed by 'import' subcommand"""
    # import frontends
    parser.add_argument(
        "-in",
        "--input-name",
        type=str,
        default=None,
        metavar="",
        help="Set the name of input node. If '--input-name'is None, "
        "default value is 'Placeholder'. Multiple values "
        "are separated by semicolon(;).",
    )
    parser.add_argument(
        "-is",
        "--input-shape",
        type=str,
        default=None,
        metavar="",
        help="Set the shape of input nodes. Multiple shapes are separated "
        "by semicolon(;) and the dims between shape are separated "
        "by space.",
    )
    parser.add_argument(
        "-on",
        "--output-name",
        type=str,
        metavar="",
        help="Set the name of output nodes. Multiple shapes are " "separated by semicolon(;).",
    )
    from .frontend_manage import get_frontend_names

    parser.add_argument(
        "--model-format",
        choices=get_frontend_names(),
        help="Specify input model format:{}".format(get_frontend_names()),
    )


@get_parameters_info("optimize")
def add_optimize_argument(parser):
    """All parameters needed by optimization"""
    parser.add_argument(
        "--board",
        default="unset",
        choices=[
            "anole",
            "light",
            "hlight",
            "asp",
            "i805",
            "c860",
            "c906",
            "c908",
            "x86_ref",
            "ch8601",
            "dp1k",
            "unset",
        ],
        help="Set target device, default is anole.",
    )
    parser.add_argument(
        "--opt-level",
        choices=[-1, 0, 1, 2, 3],
        default=3,
        type=int,
        # help="Specify the optimization level, default is 3.",
        help=argparse.SUPPRESS,
    )


@get_parameters_info("quantize")
def add_quantize_argument(parser):
    """All parameters needed by 'quantize' subcommand"""
    parser.add_argument(
        "-cd",
        "--calibrate-dataset",
        type=str,
        default=None,
        metavar="",
        help="Provide with dataset for the input of model in reference step. "
        "Support dir or .npz .jpg .png .JPEG or .txt in which there are path "
        "of images. Note: only one image path in one line if .txt.",
    )
    parser.add_argument(
        "--quantization-scheme",
        choices=[
            "int4_asym_w_sym",
            "uint8_asym",
            "int8_asym",
            "int8_sym",
            "int8_original",
            "int8_asym_w_sym",
            "int16_sym",
            "float16",
            "bfloat16",
            "unset",
        ],
        default="unset",
        help="Scheme of quantization. default is unset, and select scheme by --board.",
    )
    parser.add_argument(
        "--hybrid-quantization-scheme",
        choices=[
            "int4_asym_w_sym",
            "uint8_asym",
            "int8_asym",
            "int8_sym",
            "int8_original",
            "int8_asym_w_sym",
            "int16_sym",
            "float16",
            "bfloat16",
            "unset",
        ],
        default="unset",
        help="Scheme of hybrid quantization. default is unset.",
    )
    parser.add_argument(
        "--hybrid-layer-name",
        type=str,
        nargs="+",
        help="Layer buffer name to use hybrid quantization.",
    )
    parser.add_argument(
        "--num-bit-activation",
        type=int,
        choices=[0, 16, 32],
        default=0,
        # help="The bit number that quantizes activation layer. default is 0, means unset.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-bit-input",
        type=int,
        choices=[0, 8, 16, 32],
        default=0,
        # help="The bit number that quantizes input layer. default is 0, means unset.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-bit-weight",
        type=int,
        choices=[0, 8, 16, 32],
        default=0,
        # help="The bit number that quantizes weight layer. default is 0, means unset.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dtype-input",
        choices=["float", "int", "uint", "unset"],
        default="unset",
        # help="The dtype of quantized input layer, default is uint.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dtype-weight",
        choices=["float", "int", "uint", "unset"],
        default="unset",
        # help="The dtype of quantized constant parameters, default is uint.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dtype-activation",
        choices=["float", "int", "uint", "unset"],
        default="unset",
        # help="The dtype of quantized activation layer, default is uint.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--calibrate-mode",
        choices=["maxmin", "pow2", "kl_divergence", "kl_divergence_tsing"],
        default="maxmin",
        # help="How to calibrate while doing quantization, default is maxmin.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--activate-quantized-type",
        choices=["asym", "sym", "unset"],
        default="unset",
        # help="Select the algorithm of quantization, default is asym.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--weight-quantized-type",
        choices=["asym", "sym", "unset"],
        default="unset",
        # help="Select the algorithm of quantization, default is asym.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--weight-scale",
        choices=["max", "power2"],
        default="max",
        # help="Select the mothod that quantizes weight value, default is max.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fuse-conv-relu",
        action="store_true",
        help="Fuse the convolution and relu layer.",
    )
    parser.add_argument(
        "--fuse-reshape-dense",
        action="store_true",
        help="Fuse the reshape and dense layer.",
    )
    parser.add_argument(
        "--channel-quantization",
        action="store_true",
        help="Do quantizetion across channel.",
    )
    parser.add_argument(
        "--broadcast-quantization",
        action="store_true",
        help="Broadcast quantization parameters for special ops.",
    )
    parser.add_argument(
        "--channel-quantization-ratio-threshold",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
        # help="Set optimize quantitative parameters threshold for channel quantization."
        # + "The value can be selected in range [0, 1). 0 means this pass is not be used."
        # + "0.3 is a recommended threshold if this pass is turned on.",
    )
    parser.add_argument(
        "--fuse-clip",
        action="store_true",
        help="Fuse clip's attr into pre layer's quantitative information. "
        + "This flag is only valid when quantization is used.",
    )
    parser.add_argument(
        "--fuse-mul-before-conv",
        default=True,
        choices=[False, True],
        type=boolean_string,
        help=argparse.SUPPRESS,
        # help="This parameter is used to merge mul before conv2d to conv2d.",
    )
    parser.add_argument(
        "--fuse-mul-after-conv",
        default=True,
        choices=[False, True],
        type=boolean_string,
        help=argparse.SUPPRESS,
        # help="This parameter is used to merge mul after conv2d to conv2d.",
    )
    parser.add_argument(
        "--fuse-add-before-conv",
        default=True,
        choices=[False, True],
        type=boolean_string,
        help=argparse.SUPPRESS,
        # help="This parameter is used to merge add before conv2d to conv2d.",
    )
    parser.add_argument(
        "--fuse-add-after-conv",
        default=True,
        choices=[False, True],
        type=boolean_string,
        help=argparse.SUPPRESS,
        # help="This parameter is used to merge add after conv2d to conv2d.",
    )
    parser.add_argument(
        "--fuse-zp2bias",
        action="store_true",
        help="Merge conv2d/dense zp to bias.",
    )
    parser.add_argument(
        "--target-layout",
        default="NCHW",
        choices=["NCHW", "NHWC"],
        help="Set target layout.",
    )
    parser.add_argument(
        "--use-custom-fusion",
        action="store_true",
        help=argparse.SUPPRESS,
        # help="This parameter is used to use custom fusion like fuse cache_matmul and layer norm.",
    )
    parser.add_argument(
        "--convert-to-relay",
        action="store_true",
        help=argparse.SUPPRESS,
        # help="This parameter is used to convert quanted qnn model to relay.",
    )
    parser.add_argument(
        "--h-sram-size",
        type=str,
        default=None,
        metavar="",
        help=argparse.SUPPRESS,
        # help="Set the size of sram. The unit must in [m, M, KB, kb].",
    )
    parser.add_argument(
        "--h-max-groups",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
        # help="This parameter is used to describe the maximum number of groups supported by hardware.",
    )
    parser.add_argument(
        "--h-max-out-channel",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
        # help="This parameter is used to describe the maximum number of output channel supported by hardware.",
    )
    parser.add_argument(
        "--h-max-kernel-size",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
        # help="This parameter is used to describe the size of sram.",
    )
    parser.add_argument(
        "--h-contain-weight",
        action="store_true",
        help=argparse.SUPPRESS,
        # help="This parameter is used to describe whether the weight size should be contained when split ops.",
    )


@get_parameters_info("simulate")
def add_simulate_argument(parser):
    """All parameters needed by 'simulate' subcommand"""
    parser.add_argument(
        "-sd",
        "--simulate-data",
        type=str,
        default=None,
        metavar="",
        help="Provide with dataset for the input of model in reference step. "
        "Support dir or .npz .jpg .png .JPEG or .txt in which there are path "
        "of images. Note: only one image path in one line if .txt.",
    )


@get_parameters_info("postprocess")
def add_postprocess_argument(parser):
    """All postprocess parameters"""
    parser.add_argument(
        "--postprocess",
        type=str,
        default="top5",
        choices=["top5", "save", "save_and_top5"],
        help="Set the mode of postprocess: "
        "'top5' show top5 of output; "
        "'save' save output to file;"
        "'save_and_top5' show top5 and save output to file."
        " Default is top5",
    )


@get_parameters_info("main")
def add_main_argument(parser):
    """All commands that are compatible with previous version."""
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument("-h", "--help", action="store_true", help="Show this help information")
    parser.add_argument(
        "--version",
        action="version",
        version="{}\n".format(hhb_version()),
        help="Print the version and exit",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-E", action="store_true", help="Convert model into relay ir.")
    group.add_argument("-Q", action="store_true", help="Quantize the relay ir.")
    group.add_argument("-C", action="store_true", help="codegen the model.")
    group.add_argument("--simulate", action="store_true", help="Simulate model on x86 device.")
    # group.add_argument(
    #     "--generate-config", action="store_true", help="Generate config file for HHB"
    # )
    parser.add_argument(
        "--no-quantize", action="store_true", help="If set, don't quantize the model."
    )
    parser.add_argument(
        "-f",
        "--model-file",
        nargs="+",
        metavar="",
        help="Path to the input model file, can pass multi files",
    )
    parser.add_argument("--save-temps", action="store_true", help="Save temp files.")
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Generate dataset according to provided preprocess parameters.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="",
        default="hhb_out",
        help="The directory that holds the outputs.",
    )
    parser.add_argument(
        "--generate-readme",
        action="store_true",
        help=argparse.SUPPRESS,
        # help="Automatically generate README.md."
    )


@get_parameters_info("codegen")
def add_codegen_argument(parser):
    parser.add_argument(
        "--model-save",
        choices=["run_only", "save_only", "save_and_run"],
        default="run_only",
        help="Whether save binary graph or run only.\n"
        "run_only: execute model only, not save binary graph.\n"
        "save_only: save binary graph only.\n"
        "save_and_run: execute and save model.",
    )
    parser.add_argument(
        "--without-preprocess",
        action="store_true",
        help="Do not generate preprocess codes.",
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        # help="Create multithread codes."
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trace-strategy",
        choices=["normal", "advanced"],
        default="normal",
        # help="Strategy to generate trace data.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--input-memory-type",
        choices=[0, 1, 2],
        type=int,
        nargs="+",
        help="Set the memory type for input tensor, support for multi-values.\n"
        "0: allocated by CPU and not aligned;\n"
        "1: allocated by CPU and aligned;\n"
        "2: dma buffer.",
    )
    parser.add_argument(
        "--output-memory-type",
        choices=[0, 1, 2],
        type=int,
        nargs="+",
        help="Set the memory type for output tensor, support for multi-values.\n"
        "0: allocated by CPU and not aligned;\n"
        "1: allocated by CPU and aligned;\n"
        "2: dma buffer.",
    )
    parser.add_argument(
        "--memory-type",
        choices=[0, 1, 2],
        type=int,
        help="Set the memory type for input and output tensors.\n"
        "0: allocated by CPU and not aligned;\n"
        "1: allocated by CPU and aligned;\n"
        "2: dma buffer.",
    )
    parser.add_argument(
        "--dynamic-bc-reg", action="store_true", help="Emit bc_map file to reduce elf size on RTOS."
    )


@get_parameters_info("profiler")
def add_profiler_argument(parser):
    parser.add_argument(
        "--ir-type",
        choices=["relay", "light"],
        default="relay",
        help="The ir type that will be profiled, default is relay",
    )
    parser.add_argument(
        "--indicator",
        choices=["cal", "mem", "cycle", "all"],
        default="cal",
        nargs="+",
        help="Select indicator to profile, default is cal(calculation).\n"
        "cal: calculation, how many operations to be executed for current op.\n"
        "mem: memory, how many memory to be used for current op.\n"
        "cycle: how many cycles to execute op.\n"
        "all: include all indicators above.",
    )
    parser.add_argument(
        "--output-type",
        choices=["json", "binary", "print", "total", "all"],
        default="total",
        nargs="+",
        help="How to show results, default is show summary result.",
    )
    parser.add_argument(
        "--npu-frequency",
        default=1000000000,
        type=int,
        help="NPU frequency(HZ).",
    )


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
        "kb": 2 ** 10,
        "m": 2 ** 20,
    }

    def convert_str2int(txt_size):
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

    if hasattr(filtered_args, "h_sram_size"):
        filtered_args["h_sram_size"] = convert_str2int(filtered_args["h_sram_size"])


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
        self.filtered_args = AttributeDict(**(self.origin_args.__dict__))

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

    # 这是一个叫做返回值注解的符号。它通过允许将元数据附加到描述其参数和返回值的函数来扩展该功能。
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
