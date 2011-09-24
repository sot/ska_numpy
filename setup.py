from setuptools import setup
setup(name='Ska.Numpy',
      author = 'Tom Aldcroft',
      description='Numpy utilities',
      author_email = 'taldcroft@cfa.harvard.edu',
      py_modules = ['Ska.Numpy'],
      version='0.06',
      zip_safe=False,
      namespace_packages=['Ska'],
      packages=['Ska'],
      package_dir={'Ska' : 'Ska'},
      package_data={}
      )
