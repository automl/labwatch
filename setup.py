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

try:
    from labwatch import __about__
    about = __about__.__dict__
except ImportError:
    about = dict()
    exec(open("labwatch/__about__.py").read(), about)



setup(name='labwatch',
      version=about['__version__'],
      description='Hyperparameter optimization extension to Sacred',
      long_description=open('README.md').read(),
      classifiers=list(filter(None, classifiers.split('\n'))),
      author=about['__authors__'],
      author_email='kleinaa@cs.infomatik.uni-freiburg.de, springj@cs.uni-freiburg.de',
      url=about['__url__'],
      packages=['labwatch', 'labwatch.utils', 'labwatch.optimizers', 'labwatch.converters'],
      include_package_data=True,
      tests_require=['mock', 'mongomock', 'pytest'],
      install_requires=requires
)
