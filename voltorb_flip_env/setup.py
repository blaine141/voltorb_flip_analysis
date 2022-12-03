
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "fast_board_generator",
        ["voltorb_flip/fast_board_generator.pyx"],
        extra_compile_args=['/openmp', '-O2'],
    )
]

setup(
	name='voltorb_flip_env', # name of the package
	version='0.0.1', # version of this release
	install_requires=['gym', 'numpy'], # specifies the minimal list of libraries required to run the package correctly
	ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)


