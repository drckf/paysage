from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='paysage',
      version='0.1',
      description='Machine learning with energy based models in python',
      url='https://bitbucket.org/drckf/paysage',
      author='Charles K. Fisher',
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
      zip_safe=False)
