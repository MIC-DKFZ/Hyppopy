# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

VERSION = "0.2.0"

ROOT = os.path.dirname(os.path.realpath(__file__))

new_init = []
with open(os.path.join(ROOT, *("hyppopy", "__init__.py")), "r") as infile:
	for line in infile:
		new_init.append(line)
for n in range(len(new_init)):
	if new_init[n].startswith("__version__"):
		split = line.split("=")
		new_init[n] = "__version__ = '" + VERSION + "'\n"
with open(os.path.join(ROOT, *("hyppopy", "__init__.py")), "w") as outfile:
	outfile.writelines(new_init)


setup(
    name='hyppopy',
    version=VERSION,
    description='Hyper-Parameter Optimization Toolbox for Blackboxfunction Optimization',
    long_description=readme,
    # if you want, put your own name here
    # (this would likely result in people sending you emails)
    author='Sven Wanner',
    author_email='s.wanner@dkfz.de',
    url='',
    license=license,
    packages=find_packages(exclude=('*test*', 'doc')),
	package_data={
	   'hyppopy.plugins': ['*.yapsy-plugin']
    },
    # the requirements to install this project.
    # Since this one is so simple this is empty.
    install_requires=[
	'dicttoxml>=1.7.4',
	'xmltodict>=0.11.0',
	'hyperopt>=0.1.1',
	'Optunity>=1.1.1',
	'numpy>=1.16.0',
	'matplotlib>=3.0.2',
	'scikit-learn>=0.20.2',
	'scipy>=1.2.0',
	'Sphinx>=1.8.3',
	'xmlrunner>=1.7.7',
	'Yapsy>=1.11.223',
	'pandas>=0.24.1',
	'seaborn>=0.9.0',
	'deap>=1.2.2',
	'bayesian-optimization>=1.0.1'
	],
)

