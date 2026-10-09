                                       
                                         
import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
import shutil

from setuptools import setup, find_packages
import subprocess
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)

                                                            
this_dir = os.path.dirname(os.path.abspath(__file__))
                                                                                        
FORCE_CXX11_ABI = os.getenv("FORCE_CXX11_ABI", "FALSE") == "TRUE"

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

                                                                                
                                                                                
                                   
MODES = ["oflex"]

def get_ext():
    cc_flag = []

    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    print("\n\nCUDA_HOME = {}\n\n".format(CUDA_HOME))

                                                              
    multi_threads = True
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        print("CUDA version: ", bare_metal_version, flush=True)
        if bare_metal_version < Version("11.6"):
            warnings.warn("CUDA version ealier than 11.6 may leads to performance mismatch.")
        if bare_metal_version < Version("11.2"):
            multi_threads = False
            
    cc_flag.extend(["-gencode", "arch=compute_89,code=sm_89"])
    if multi_threads:
        cc_flag.extend(["--threads", "4"])

                                                                               
                                     
                                                                                                                        
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    sources = dict(
        core=[
            "csrc/selective_scan/cus/selective_scan.cpp",
            "csrc/selective_scan/cus/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cus/selective_scan_core_bwd.cu",
        ],
        nrow=[
            "csrc/selective_scan/cusnrow/selective_scan_nrow.cpp",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd2.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd3.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd4.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd2.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd3.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd4.cu",
        ],
        ndstate=[
            "csrc/selective_scan/cusndstate/selective_scan_ndstate.cpp",
            "csrc/selective_scan/cusndstate/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cusndstate/selective_scan_core_bwd.cu",
        ],
        oflex=[
            "csrc/selective_scan/cusoflex/selective_scan_oflex.cpp",
            "csrc/selective_scan/cusoflex/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cusoflex/selective_scan_core_bwd.cu",
        ],
    )

    names = dict(
        core="selective_scan_cuda_core",
        nrow="selective_scan_cuda_nrow",
        ndstate="selective_scan_cuda_ndstate",
        oflex="selective_scan_cuda_oflex",
    )

    ext_modules = [
        CUDAExtension(
            name=names.get(MODE, None),
            sources=sources.get(MODE, None),
            extra_compile_args={
                "cxx": ["/O2", "/std:c++20", "/Zc:preprocessor"],
                "nvcc": [
                            "-O3",
                            "-std=c++20",
                            "-U__CUDA_NO_HALF_OPERATORS__",
                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                            "--expt-relaxed-constexpr",
                            "--expt-extended-lambda",
                            "--use_fast_math",
                            "--ptxas-options=-v",
                            "-lineinfo",
                            "-Xcompiler=/Zc:preprocessor",
                        ]
                        + cc_flag
            },
            include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
        )
        for MODE in MODES
    ]

    return ext_modules

ext_modules = get_ext()
setup(
    name="selective_scan",
    version="0.0.2",
    packages=[],
    author="Tri Dao, Albert Gu, $@#Anonymous#@$ ",
    author_email="tri@tridao.me, agu@cs.cmu.edu, $@#Anonymous#EMAIL@$",
    description="selective scan",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/state-spaces/mamba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": _bdist_wheel, "build_ext": BuildExtension} if ext_modules else {"bdist_wheel": _bdist_wheel,},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
        "einops",
    ],
)
