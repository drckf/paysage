from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='paysage',
      version='0.1',
      description='Machine learning with energy based models in python',
      url='https://github.com/drckf/paysage',
      author='Unlearn.AI, Inc.',
      author_email='drckf@unlearnai.com',
      license='MIT',
      packages=['paysage'],
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
