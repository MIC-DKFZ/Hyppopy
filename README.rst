Hyppopy
========================

The friendly Hyppo that helps you to find your inner blackbox optimum.

It is adapted from `this github example <https://github.com/kennethreitz/samplemod/tree/master>`_
(`Learn more <http://www.kennethreitz.org/essays/repository-structure-and-python>`_) and tries to follow the instructions on `this guide <http://docs.python-guide.org/en/latest/writing/structure/>`_.

The actual code can be found in :py:mod:`samplepackage.thinker`. It implements a toy class
which "thinks" random thoughts.
The tests for this class are located in :py:mod:`samplepackage.tests`. Python will automatically
discover tests if you name them test_*.py
Look at :py:meth:`samplepackage.tests.test_advanced` for a simple unittest example.

Install Dependencies
--------------------
either use::

  pip install -r requirements.txt

or::

  make init


Install This Package
--------------------
You have two choices:

#. "normal" install. This will install the current version.
#. install a development version. Here the package will be installed, but only as a link to your current codebase. See `here <http://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install>`_ for a better explanation :-)

After you installed your package, it can be imported via::

  import samplepackage

in your python console.

Install Development Version:
^^^^^^^^^^^^^^^^^^^^^^^^^^^

either use::

  python setup.py develop

or::

  make install_develop

"Normal" Install
^^^^^^^^^^^^^^^

either use::

  python setup.py install

or::

  make install


run tests
--------------------

either run::

  python -m unittest discover

or::

  make test


execute tutorial
--------------------
In your console, navigate to the tutorials subfolder and start ipython notebook.


run scripts
--------------------
Scripts can be found in subfolder bin/. They are declared as entry points
(see setup.py in the project root). This means you can call them by calling the entry points
directly in console!


build documentation
--------------------

First the documentation has to be refreshed by typing::

    sphinx-apidoc -e -f samplepackage -o doc/

or::

    make documentation

in the projects root folder (the one with setup.py). This will automatically create
all the files necessary for sphinx (the documentation builder) to create the
html documentation.
Then, navigate to doc/ and type::

    make html

To build the documentation. Note that sphinx needs to be available in your python
installation (e.g. install requirements.txt as mentioned above).
The documentation main page can then be found in doc/_build/html/index.html
