from setuptools import setup

def readme():
    with open('README') as f:
        return f.read()

setup(name='s3c',
      version='0.0.1',
      description='Specific Contributions to Community Changes',
      long_description=readme(),
      url='http://github.com/geeklhem/s3c',
      author='Guilhem Doulcier',
      author_email='guilhem.doulcier@ens.fr',
      license='GPLv3',
      packages=['s3c'],
      zip_safe=False,
      install_requires=[
          'matplotlib',
          'numpy',
          'pandas',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      #scripts=['bin/s3c'],
)
