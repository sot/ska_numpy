# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup, Extension

import numpy as np
from Cython.Build import cythonize

fastss_ext = Extension("*",
                       ['Ska/Numpy/fastss.pyx'],
                       include_dirs=[np.get_include()])
try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='Ska.Numpy',
      author='Tom Aldcroft',
      description='Numpy utilities',
      author_email='aldcroft@head.cfa.harvard.edu',
      py_modules=['Ska.Numpy'],
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      ext_modules=cythonize([fastss_ext]),
      zip_safe=False,
      packages=['Ska', 'Ska.Numpy', 'Ska.Numpy.tests'],
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
