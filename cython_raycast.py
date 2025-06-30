from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy

ext = Extension(
        "*",
        ["run/*.pyx"],
        include_dirs = [numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),

setup(
    name='Raycast_cython_test',
    ext_modules=cythonize(ext),
)
