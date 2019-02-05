# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='hyppopy',
    version='0.0.1',
    description='Hyper-Parameter Optimization Toolbox for Blackboxfunction Optimization',
    long_description=readme,
    # if you want, put your own name here
    # (this would likely result in people sending you emails)
    author='Sven Wanner',
    author_email='s.wanner@dkfz.de',
    url='',
    license=license,
    packages=find_packages(exclude=('bin', '*test*', 'doc', 'hyppopy')),
    # the requirements to install this project.
    # Since this one is so simple this is empty.
    install_requires=['dicttoxml>=1.7.4', 'hyperopt>=0.1.1', 'matplotlib>=3.0.2', 'numpy>=1.16.0',
	'Optunity>=1.1.1', 'pytest>=4.1.1', 'scikit-learn>=0.20.2', 'scipy>=1.2.0', 'sklearn>=0.0', 'Sphinx>=1.8.3',
	'xmlrunner>=1.7.7', 'xmltodict>=0.11.0', 'Yapsy>=1.11.223', 'visdom>=0.1.8.8'],
    # a more sophisticated project might have something like:
    #install_requires=['numpy>=1.11.0', 'scipy>=0.17', 'scikit-learn']

    # after running setup.py, you will be able to call hypopy_exe
    # from the console as if it was a normal binary. It will call the function
    # main in bin/hypopy_exe.py
    entry_points={
        'console_scripts': ['hyppopy_exe=bin.hypopy_exe:main'],
    }
)

