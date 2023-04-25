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
""" HHB Command Line Tools """

# argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。argparse模块的作用是用于解析命令行参数。
import argparse

# logging 是python用于记录日志的标准模块
import logging

# sys是python用于python解释器相关的变量和方法，这点于os模块不同
import sys

# os是python用于为os模块提供了多数操作系统的功能接口函数
import os

# 可以选择用sys.path.insert(0,‘/path’),这样新添加的目录会优先于其他目录被import检查, sys.path.append(),则是放在列表尾部。注意：sys.path是个列表
# os.path.dirname((__file__)是得到当前文件的绝对路径
# https://blog.csdn.net/weixin_38470851/article/details/80367143
# 这句话的本意就是使得该文件所在目录优先于其他sys被找到
sys.path.insert(0, os.path.dirname(__file__))

# 导入./core文件夹中的各种库。
from core.arguments_manage import ArgumentManage, CommandType, HHBException, ArgumentFilter
from core.arguments_manage import update_arguments_by_file
from core.common import ALL_ARGUMENTS_INFO, import_module_for_register, collect_arguments_info
from core.common import ArgInfo, ALL_ARGUMENTS_DESC

# 在内部字典中将lvl与名字levelName联系在一起，用以将数字型级别映射为文本型表示。例如 Formatter格式化一条信息。这个函数也可以用于自定义级别。
LOG = 25
logging.addLevelName(LOG, "LOG")

# _main函数，argv
def _main(argv):
    """ HHB commmand line interface. """
    arg_manage = ArgumentManage(argv)
    arg_manage.check_cmd_arguments()

    parser = argparse.ArgumentParser(
        prog="HHB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="HHB command line tools",
        epilog=__doc__,
        allow_abbrev=False,
        add_help=False,
    )

    # add command line parameters
    curr_command_type = arg_manage.get_command_type()
    if curr_command_type == CommandType.SUBCOMMAND:
        arg_manage.set_subcommand(parser)
    else:
        arg_manage.set_main_command(parser)
        ALL_ARGUMENTS_DESC["main_command"] = collect_arguments_info(parser._actions)

    # print help info
    if arg_manage.have_help:
        arg_manage.print_help_info(parser)
        return 0

    # generate readme file
    if arg_manage.have_generate_readme:
        arg_manage.generate_readme(parser)
        return 0

    # parse command line parameters
    args = parser.parse_args(arg_manage.origin_argv[1:])
    if args.config_file:
        update_arguments_by_file(args, arg_manage.origin_argv[1:])
    args_filter = ArgumentFilter(args)

    # config logger
    logging.basicConfig(
        format="[%(asctime)s] (%(name)s %(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("HHB")
    logger.setLevel(25 - args.verbose * 10)

    # run command
    arg_manage.run_command(args_filter, curr_command_type)


def main():
    ALL_MODULES_FOR_REGISTER = [   
        "importer",
        "quantizer",
        "codegen",
        "simulate",
        "benchmark",
        "profiler",
    ]
    import_module_for_register(ALL_MODULES_FOR_REGISTER)

    argv = sys.argv
    sys.exit(_main(argv))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl-C detected.")
