from setuptools import setup
setup(name='Ska.Numpy',
      author = 'Tom Aldcroft',
      description='Numpy utilities',
      author_email = 'taldcroft@cfa.harvard.edu',
      py_modules = ['Ska.Numpy'],
      version='0.06',
      zip_safe=False,
      namespace_packages=['Ska'],
      packages=['Ska', 'Ska/Numpy'],
      package_dir={'Ska': 'Ska', 'Ska.Numpy': 'Ska/Numpy'},
      package_data={}
      )
