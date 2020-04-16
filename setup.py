# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys
from setuptools import setup, Extension

# Special case here to allow `python setup.py --version` to run without
# requiring cython and numpy to be installed.
if '--version' in sys.argv[1:]:
    cythonize = lambda arg: None
    fastss_ext = None
else:
    from Cython.Build import cythonize
    import numpy as np
    fastss_ext = Extension("Ska.Numpy.fastss",
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
