# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

def readme():
    with open(path.join(here, 'README.md')) as f:
        return f.read()

setup(name='paysage',
      version='0.1.4',
      description='Machine learning with energy based models in python',
      url='https://github.com/drckf/paysage',
      author='Unlearn.AI, Inc.',
      author_email='drckf@unlearnai.com',
      license='MIT',
      packages=find_packages(),
      package_data={
        '': ['*.json', '*.py']
        },
      include_package_data=True,
      install_requires=[
          'h5py',
          'matplotlib',
          'numexpr',
          'numpy',
          'pandas',
          'pytest',
          'scikit-learn',
          'scipy',
          'seaborn',
          'tables',
          'cytoolz'
          ],
      tests_require=[
          'pytest'
      ],
      python_requires='~=3.6',
      zip_safe=False)
