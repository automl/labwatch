import labwatch
from setuptools import setup

classifiers = """
Intended Audience :: Science/Research
Natural Language :: English
Programming Language :: Python
Topic :: Utilities
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Software Development :: Libraries :: Python Modules
License :: OSI Approved :: MIT License
"""

requires = [
    'numpy >= 1.7',
    'sacred',
    'pymongo',
    'ConfigSpace'
]

setup(name='labwatch',
      version=labwatch.__version__,
      description='Hyperparameter optimization extension to Sacred',
      long_description=open('README.md').read(),
      classifiers=list(filter(None, classifiers.split('\n'))),
      author=labwatch.__authors__,
      author_email='kleinaa@cs.infomatik.uni-freiburg.de, springj@cs.uni-freiburg.de',
      url=labwatch.__url__,
      packages=['labwatch', 'labwatch.utils', 'labwatch.optimizers', 'labwatch.converters'],
      include_package_data=True,
      tests_require=['mock', 'mongomock', 'pytest'],
      install_requires=requires
)
