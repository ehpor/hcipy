Installation
============

Using pip
---------

HCIPy is available on PyPI, so installation can be as easy as:

.. code-block:: shell

    pip install hcipy --upgrade

It is also possible to install the latest development version from Github, by:

.. code-block:: shell
    
    git clone https://github.com/ehpor/hcipy
    cd hcipy
    pip install -e .

The ``-e`` flag tells ``pip`` to install HCIPy in editable mode, ie. edits to the source files in the ``hcipy`` directory will change the library. This allows for easy updates by simply pulling the git repository:

.. code-block:: shell

    git pull
    python setup.py egg_info

You can check if HCIPy is installed correctly by importing it in Python:

.. code-block:: python

    >>> import hcipy
    >>> print(hcipy.__version__)

If you want to install the latest development verison, but do not intend to receive continuous updates (not advised), you can also install in non-editable mode:

.. code-block:: shell

    git clone https://github.com/ehpor/hcipy
    pip install ./hcipy
    rm -rf hcipy

Dependencies
------------

All Python dependencies will be installed alongside HCIPy by the installer. These include:

* **numpy** (for all numerical calculations)
* **scipy** (for advanced linear algebra)
* **matplotlib** (for visualisations)
* **Pillow** (for gif file output)
* **pyyaml** (for the configuration file)
* **astropy** (for fits file reading and writing)
* **imageio** (for image writing)
* **xxhash** (for efficiently hashing Grids)
* **numexpr** (for speedup of lengthy numerical operations)

However, there is one dependency that ``pip`` does not automatically install. This is FFMpeg, which is used for writing out animations to video files. Installation of FFMpeg is not necessary, unless you want to use the ``FFMpegWriter`` class in HCIPy.

Most Linux systems should have this preinstalled, but on MacOS and Windows, you need to install it yourself, either from their `website <https://www.ffmpeg.org/>`_ or other sources. You also need to make sure that the ``ffmpeg`` binary can be found by HCIPy. You can do this by adding the ``ffmpeg/bin`` directory to the ``PATH`` environment variable, if this is not already done by the FFMpeg installer.

Running the test suite
----------------------

For running the test suite, you'll need to install the test dependencies as well:

.. code-block:: shell

    pip install -e ".[dev]"

or

.. code-block:: shell

    pip intall hcipy[dev]

depending on the way you originally installed HCIPy. You can then run the full test suite by running the following in the HCIPy directory.

.. code-block:: shell

    pytest ./tests

This should execute all simple tests in around 3-5 minutes, depending on your computer. It skips the longer tests that perform more extensive verification checks. All tests, including slow ones, can be run as well

.. code-block:: shell

    pytest ./tests --runslow

This should take about 10-20 minutes, depending on your computer.

A single submodule can be tested as well, for example:

.. code-block:: shell

    pytest ./tests/test_atmosphere.py

Building the documentation
--------------------------

A pre-built version of the documentation is always available `online <https://docs.hcipy.org>`_, in fact, you're most likely reading it from that website right now. 

The HCIPy documentation uses Sphinx to build an HTML website containing the documentation. If you want to build a local copy, you'll first have to install the documentation dependencies first:

.. code-block:: shell

    pip install -e ".[doc]"

Building the documentation also requires FFMpeg to be installed. You can now build the documentation:

.. code-block:: shell

    cd doc
    make html

This will execute all tutorials, and compile the documentation. The first time you build the documentation, it may take 5-10 minutes, depending on your computer. Afterwards, a refresh should take about a minute. The final HTML documentation is then available in ``doc/_build/html``.

The documentation can be rebuilt from scratch by cleaning it first.

.. code-block:: shell

    make clean
    make html