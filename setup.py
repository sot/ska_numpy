from setuptools import setup, Extension

import numpy as np
from Cython.Build import cythonize
from Ska.Numpy import __version__

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
      version=__version__,
      ext_modules=cythonize([fastss_ext]),
      zip_safe=False,
      packages=['Ska', 'Ska.Numpy', 'Ska.Numpy.tests'],
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
