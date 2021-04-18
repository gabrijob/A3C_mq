from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension("A3CProcesses", ["A3CProcesses.pyx"],
        include_dirs=["./"],
        #libraries=["tensorflow_framework"], 
        library_dirs=["./"]),
]

setup(
    name='A3C Processes API',
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
