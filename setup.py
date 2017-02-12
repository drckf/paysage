from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='paysage',
      version='0.0',
      description='Machine learning with energy based models in python',
      url='https://bitbucket.org/drckf/paysage',
      author='Charles K. Fisher',
      author_email='charleskennethfisher@gmail.com',
      license='MIT',
      packages=['paysage'],
      install_requires=[
          'h5py',
          'matplotlib',
          'numba',
          'numexpr',
          'numpy',
          'pandas',
          'scikit-learn',
          'scipy',
          'seaborn',
          'tables'
      ],
      zip_safe=False)
