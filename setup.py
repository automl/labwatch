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
    'pymongo'
]

setup(name='labwatch',
      version=labwatch.__version__,
      description='Bayesian optimization extension to sacred',
      long_description='',
      classifiers=list(filter(None, classifiers.split('\n'))),
      author=labwatch.__authors__,
      author_email='springj@informatik.uni-freiburg.de',
      url='',
      keywords='',
      packages=['labwatch'],
      include_package_data=True,
      tests_require=['mock', 'mongomock', 'pytest'],
      install_requires=requires
)
