# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys
from setuptools import setup, Extension

from ska_helpers.setup_helper import duplicate_package_info
from testr.setup_helper import cmdclass

# Special case here to allow `python setup.py --version` to run without
# requiring cython and numpy to be installed.
if '--version' in sys.argv[1:]:
    ext_modules = None
else:
    from Cython.Build import cythonize
    import numpy as np
    fastss_ext = Extension("ska_numpy.fastss",
                           ['ska_numpy/fastss.pyx'],
                           include_dirs=[np.get_include()])
    ext_modules = cythonize([fastss_ext])

name = "ska_numpy"
namespace = "Ska.Numpy"

packages = ["ska_numpy", "ska_numpy.tests"]
package_dir = {name: name}

duplicate_package_info(packages, name, namespace)
duplicate_package_info(package_dir, name, namespace)

setup(name=name,
      author='Tom Aldcroft',
      description='Numpy utilities',
      author_email='taldcroft@cfa.harvard.edu',
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      ext_modules=ext_modules,
      zip_safe=False,
      package_dir=package_dir,
      packages=packages,
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
