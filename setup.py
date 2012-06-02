from setuptools import setup

import numpy as np
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Ska.Numpy.version import version

cmdclass = {'build_ext': build_ext}
fastss_ext = Extension("Ska.Numpy.fastss",
                       ['Ska/Numpy/fastss.pyx'],
                       include_dirs=[np.get_include()])

setup(name='Ska.Numpy',
      author='Tom Aldcroft',
      description='Numpy utilities',
      author_email = 'aldcroft@head.cfa.harvard.edu',
      py_modules = ['Ska.Numpy'],
      version=version,
      cmdclass=cmdclass,
      ext_modules=[fastss_ext],
      zip_safe=False,
      namespace_packages=['Ska'],
      packages=['Ska', 'Ska/Numpy'],
      package_dir={'Ska': 'Ska', 'Ska.Numpy': 'Ska/Numpy'},
      package_data={}
      )
