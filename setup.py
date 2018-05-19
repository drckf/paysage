from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='paysage',
      version='0.2.0',
      description='Machine learning with energy based models in python',
      url='https://github.com/drckf/paysage',
      author='Unlearn.AI',
      author_email='drckf@unlearn.ai',
      license='MIT',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
      ],
      packages=['paysage'],
      install_requires=[
          'matplotlib',
          'numexpr',
          'numpy',
          'pandas',
          'pytest',
          'scipy',
          'seaborn',
          'tables',
          'torchvision',
          'cytoolz'
          ],
      tests_require=[
          'pytest'
      ],
      python_requires='>=3.6',
      zip_safe=False)
