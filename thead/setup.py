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
# pylint: disable=invalid-name, exec-used
"""Setup HHB package."""

from setuptools import find_packages
from setuptools import setup

__version__ = "2.9.5"


def get_package_data_files():
    # template files
    return [
        "config/*",
        "config/process/include/*",
        "config/process/src/*",
        "config/template/*",
        "config/wrapper/*",
        "prebuilt/runtime/riscv_linux/libprebuilt_runtime.a",
        "prebuilt/runtime/x86_linux/libprebuilt_runtime.a",
        "prebuilt/runtime/cmd_parse/cmd_parse.h",
        "prebuilt/decode/install/lib/rv/*",
        "prebuilt/decode/install/lib/x86/*",
        "3rdparty/dlpack/include/dlpack/dlpack.h",
        "install_nn2/*",
        "install_nn2/x86/include/*",
        "install_nn2/x86/include/csinn/*",
        "install_nn2/x86/include/graph/*",
        "install_nn2/x86/include/shl_public/*",
        "install_nn2/x86/lib/*",
        "install_nn2/c906/include/*",
        "install_nn2/c906/include/csinn/*",
        "install_nn2/c906/include/graph/*",
        "install_nn2/c906/include/shl_public/*",
        "install_nn2/c906/lib/*",
        "install_nn2/c908/include/*",
        "install_nn2/c908/include/csinn/*",
        "install_nn2/c908/include/graph/*",
        "install_nn2/c908/include/shl_public/*",
        "install_nn2/c908/lib/*",
        "install_nn2/c920/include/*",
        "install_nn2/c920/include/csinn/*",
        "install_nn2/c920/include/graph/*",
        "install_nn2/c920/include/shl_public/*",
        "install_nn2/c920/lib/*",
        "install_nn2/c920v2/include/*",
        "install_nn2/c920v2/include/csinn/*",
        "install_nn2/c920v2/include/graph/*",
        "install_nn2/c920v2/include/shl_public/*",
        "install_nn2/c920v2/lib/*",
        "install_nn2/th1520/include/*",
        "install_nn2/th1520/include/csinn/*",
        "install_nn2/th1520/include/graph/*",
        "install_nn2/th1520/include/shl_public/*",
        "install_nn2/th1520/lib/*",
        "install_nn2/pnna_x86/include/*",
        "install_nn2/pnna_x86/include/csinn/*",
        "install_nn2/pnna_x86/include/graph/*",
        "install_nn2/pnna_x86/include/shl_public/*",
        "install_nn2/pnna_x86/lib/*",
    ]


setup(
    name="hhb",
    version=__version__,
    description="HHB: XuanTie deployment tools for Deep Learning Systems",
    zip_safe=False,
    entry_points={"console_scripts": ["hhb = hhb.main:main"]},
    install_requires=[
        "hhb-tvm",
        "scikit-build",
        "tqdm",
        "pyyaml",
        "opencv-python",
        "tflite",
        "protobuf",
        "ppq==0.6.6",
        "onnxsim",
        "onnx",
        "jsbeautifier",
        "transformers",
        "pandas",
    ],
    packages=find_packages(),
    package_dir={"hhb": "hhb"},
    package_data={"hhb": get_package_data_files()},
    url="https://github.com/T-head-Semi/tvm",
)
