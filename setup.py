import os
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

VERSION = "0.5.0.6"

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
    packages=find_packages(exclude=('tests', 'doc')),
    # the requirements to install this project.
    # Since this one is so simple this is empty.
    install_requires=[
		'bayesian-optimization>=1.0.1',
		'hyperopt>=0.1.2',
		'matplotlib>=3.0.3',
		'numpy>=1.16.2',
		'optuna>=0.9.0',
		'Optunity>=1.1.1',
		'pandas>=0.24.2',
		'pytest>=4.3.1',
		'scikit-learn>=0.20.3',
		'scipy>=1.2.1',
		'visdom>=0.1.8.8'
	],
)

