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
          'pytest',
          'scikit-learn',
          'scipy',
          'seaborn',
          'tables'
          ],
      tests_require=[
          'pytest'
      ],
      zip_safe=False)

# PyTorch is not available through PyPi yet (but it is available through conda)
# And, it is not available for windows
# Need to deal with the dependencies directly
import pip
from sys import platform as _platform

if _platform == "linux" or _platform == "linux2":
    pip.main(['install', "https://s3.amazonaws.com/pytorch/whl/cu75/torch-0.1.10.post2-cp35-cp35m-linux_x86_64.whl"])
    pip.main(['install', "torchvision"])
elif _platform == "darwin":
    pip.main(['install', "https://s3.amazonaws.com/pytorch/whl/torch-0.1.10.post1-cp35-cp35m-macosx_10_6_x86_64.whl "])
    pip.main(['install', "torchvision"])
elif _platform == "win32":
   raise RuntimeError("PyTorch is not available for windows")
