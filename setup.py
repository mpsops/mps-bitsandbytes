"""
Setup script for MPS BitsAndBytes

Builds the native Metal extension for GPU-accelerated quantization.
"""

import os
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class ObjCppBuildExt(build_ext):
    """Build extension for Objective-C++ with PyTorch."""

    def build_extensions(self):
        import torch
        from torch.utils import cpp_extension

        self.compiler.src_extensions.append('.mm')

        original_compile = self.compiler._compile

        def objcpp_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.mm'):
                extra_postargs = ['-x', 'objective-c++'] + list(extra_postargs or [])
            return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = objcpp_compile

        for ext in self.extensions:
            ext.include_dirs.extend(cpp_extension.include_paths())
            ext.library_dirs.extend(cpp_extension.library_paths())
            ext.libraries.extend(['c10', 'torch', 'torch_cpu', 'torch_python'])

        super().build_extensions()


def get_extensions():
    if sys.platform != "darwin":
        return []

    return [Extension(
        name="mps_bitsandbytes._C",
        sources=["mps_bitsandbytes/csrc/mps_bitsandbytes.mm"],
        extra_compile_args=["-std=c++17", "-O3", "-DNDEBUG"],
        extra_link_args=["-framework", "Metal", "-framework", "Foundation"],
    )]


setup(
    name="mps-bitsandbytes",
    version="0.2.0",
    description="4-bit NF4 and 8-bit quantization for PyTorch on Apple Silicon",
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={
        "mps_bitsandbytes": [
            "kernels/*.metal",
            "kernels/*.metallib",
        ],
    },
    include_package_data=True,
    install_requires=["torch>=2.0"],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
        "transformers": ["transformers>=4.30.0", "accelerate>=0.20.0"],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": ObjCppBuildExt},
    python_requires=">=3.10",
)
