from setuptools import setup
from Ska.Numpy.version import version


setup(name='Ska.Numpy',
      author = 'Tom Aldcroft',
      description='Numpy utilities',
      author_email = 'taldcroft@cfa.harvard.edu',
      py_modules = ['Ska.Numpy'],
      version=version,
      zip_safe=False,
      namespace_packages=['Ska'],
      packages=['Ska', 'Ska/Numpy'],
      package_dir={'Ska': 'Ska', 'Ska.Numpy': 'Ska/Numpy'},
      package_data={}
      )
