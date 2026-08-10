import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


_musa_home = Path(os.environ.get("MUSA_HOME", "/usr/local/musa")).resolve()
_musa_lib = _musa_home / "lib"
if not (_musa_lib / "libmusart.so").exists():
    _musa_lib = _musa_home / "lib64"


class CleanOptionalBuildExt(build_ext):
    """Never package a stale optional binary after a failed rebuild."""

    def build_extension(self, extension):
        if extension.optional:
            output = Path(self.get_ext_fullpath(extension.name))
            if output.is_file():
                output.unlink()
        super().build_extension(extension)


setup(
    cmdclass={"build_ext": CleanOptionalBuildExt},
    ext_modules=[
        Extension(
            "musa_perf_collector.online._native_dispatch",
            ["src/musa_perf_collector/online/native_dispatch.c"],
            extra_compile_args=["-O3", "-std=c11"],
        ),
        Extension(
            "musa_perf_collector.online._native_musa_trace",
            ["src/musa_perf_collector/online/native_musa_trace.c"],
            include_dirs=[str(_musa_home / "include")],
            library_dirs=[str(_musa_lib)],
            runtime_library_dirs=[str(_musa_lib)],
            libraries=["musart"],
            extra_compile_args=["-O3", "-std=c11"],
            optional=True,
        ),
    ]
)
