#  Copyright (C) 2012 Matt Hagy <hagy@gatech.edu>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

name = 'rtchemstats'
version = '0.0.1'

import sys
import os

from distutils.core import setup, Extension
from distutils.sysconfig import customize_compiler
from distutils.ccompiler import new_compiler

try:
    import numpy as np
    from numpy.distutils.misc_util import get_numpy_include_dirs
except ImportError:
    print name + ' requires numpy'
    exit(1)

try:
    from Cython.Compiler.Main import (compile as cython_compile,
                                      CompilationOptions as CythonCompilationOptions,
                                      default_options as cython_default_options)
    from Cython.Compiler.Errors import PyrexError
    have_cython = True
except ImportError,e:
    have_cython = False

os.chdir(os.path.dirname(__file__) or os.getcwd())

def msg(s, *args):
    sys.stderr.write((s % args if args else s) + '\n')
    sys.stderr.flush()

def warn_msg(*args):
    msg(*args)

def error_msg(*args):
    msg(*args)
    error_exit()

def error_exit():
    exit(1)

def ensure_file(path):
    if not os.path.isfile(path):
        error_msg('missing path %s', path)
    return path

def run_cython(cython_file, c_file):
    assert have_cython
    msg('Cythonizing %s -> %s', cython_file, c_file)
    options = CythonCompilationOptions(cython_default_options)
    options.output_file = c_file
    try:
        result = cython_compile([cython_file], options)
    except (EnvironmentError, PyrexError), e:
        error_msg(str(e))
    else:
        if result.num_errors > 0:
            error_exit()

def cython_extension(name, extra_c_files=[]):
    base_path = name.replace('.','/')
    cython_file = ensure_file(base_path + '.pyx')
    cython_def_file = base_path + '.pxd'
    c_file = base_path + '.c'

    if (not os.path.exists(c_file) or
        os.path.getmtime(c_file) <
        max(os.path.getmtime(path) for path in [cython_file, cython_def_file]
            if os.path.exists(path))):
        if have_cython:
            run_cython(cython_file, c_file)
        else:
            ensure_file(c_file)
            warn_msg('%s stale : %s has been updated and cython not available',
                     c_file, cython_file)

    return Extension(name=name,
                     sources=[c_file],
                     extra_objects=map(prepare_extra_c_file, extra_c_files),
                     include_dirs=get_numpy_include_dirs())

def prepare_extra_c_file(info):
    c_file = info['filename']
    compile_args = info.get('compile_args', [])
    cc = new_compiler(verbose=3)
    customize_compiler(cc)
    [o_file] = cc.compile([c_file], '.',
                         extra_postargs=compile_args)
    return o_file

extensions = [
    cython_extension('rtchemstats.cython.util'),
    cython_extension('rtchemstats.cython.pair1d'),
    cython_extension('rtchemstats.cython.pair2d'),
    cython_extension('rtchemstats.cython.bond'),
    cython_extension('rtchemstats.cython.tcf'),
    ]

setup(
    name=name,
    version=version,
    url='https://github.com/matthagy/rtchemstats',
    author='Matt Hagy',
    author_email='hagy@gatech,.edu',
    description='Compute statistics description of chemical systems in real-time',
    long_description='''
Python library to compute statistical description of chemical systems
in real-time without the need to write trajectories to disk. This allows
one to collect high resolution distribution functions with roughly 4 orders
of magnitude less disk space. In practice 100GB trajectory files have
been replaced by 50MB state files.

This librariy is based around StatComputers which extract the necesary
information from a sequence of simulation configurations to compute
a specific distribution function. StatComputers are memory efficient
in that they extract the minimimal information from each configuration.
The internal analysis code is implemented in Cython for efficency.
StatComputers are also restartable, in that they can be pickled to disk
as part of a simulation restart file.

StatComputers are implemented for the following distribution functions
 o Isotropic pair correlation function (i.e. h(r) = g(r) - 1)
 o 2-Dimensional pair correlation function
 o 2-Dimensional orienation correlation function
 o Bond angle about a common atom distribution
 o Mean squared displacment function
 o Velocity autocorrelation function
 o Reversible bond duration distribution
''',
    classifiers = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Cython",
    'Topic :: Education',
    'Topic :: Scientific/Engineering :: Chemistry',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Utilities'
    ],
    packages = ['rtchemstats'],
    ext_modules = extensions
    )
