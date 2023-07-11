Project setup
=============

This page describes the technical infrastructure we have set up to develop and maintain HCIPy.

Folder structure and files
--------------------------

This section explains the content of the repository:

* **hcipy**. This folder contains all of the code for the HCIPy package.
* **doc**. This folder contains the full documentation.
* **tests**. This folder contains the unit tests, separated by submodule.
* **examples**. This folder contains old examples. These examples should be rewritten into tutorials for the documentation. Do not add new examples here.
* **pyproject.toml**. This file is necessary for pip installation.
* **setup.py**. This file was used for pip installation, but is now deprecated. It is being kept for backwards compatibility.
* **README.md**. Main repository readme file.
* **LICENSE**. A copy of the MIT license, which should be redistributed with any copy of HCIPy.
* **.azure-pipelines.yml**. Configuration file for the continuous integration service.
* **.flake8**. Configuration file for the Python linting.
* **.coveragerc**. Configuration file for test coverage.
* **.gitignore**. List of files to ignore for the version control system.

.. _git-style:

Git
---

Development for HCIPy takes place completely on `Github <https://github.com/ehpor/hcipy>`__. We're using issues to track bugs. We're using the feature-branching workflow. For a tutorial on how to contribute a new feature, please refer to the :doc:`contributing guide <contributing_guide>`.

We prefer git commits that are appropriately sized. That is, use a single commit for one logical change to the code base. Do not commit a whole feature consisting of many-hundreds of new lines of code in a single commit. Useful commits should show the line of development separated in logical parts.

Git is also used to generate a version string, available as ``hcipy.__version__``. We use the `setuptools_scm <https://pypi.org/project/setuptools-scm/>`__. This means that for an editable installation of HCIPy, the version needs to be updated after you pull the repository.

.. code-block:: shell

    python setup.py egg_info

.. _coding-style:

The version string will be either released version (ie. ``{tag}`, with ``{tag}`` being for example ``0.5.1``), a development version with commit hash (ie. ``{tag}.dev{distance}+g{commit_hash}`` where ``{distance}`` is the number of commits since the last release), or a dirty version string (ie. ``{tag}.dev{distance}+g{commit_hash}.d{yyyymmdd}`` where ``{yyyymmdd}`` is the current date).  The version string of HCIPy for this documentation is listed at the top of the navigation bar.

Coding style
------------

We adhere mostly to `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__, the official style guide for Python code. Deviations from PEP8, plus some additional rules, are listed below:

* **No double blank lines**. Deviating from PEP8, at most one blank line may be used to separate functions, classes or logical sections inside functions.
* **Long lines allowed in moderation**. Deviating from PEP8, HCIPy code sets the limit at <100 characters, with lines longer than this allowed sporadically.
* **No print statements**. Inside the HCIPy package code, no print statements should be used to avoid printing stuff during normal operation.
* **No commented-out code**. Commented code should be removed, as these lines can always be retrieved using git.
* **Docstrings in numpydoc**. All functions and classes that are exposed to code outside of HCIPy, should have a docstring written in the numpydoc format.

Continuous integration and test suite
-------------------------------------

HCIPy currently supports Python 3.7+ on Linux, MacOS and Windows. To make sure that it keeps working as expected on all these configurations, we have set up automatic testing on each of these. We currently use `Azure <https://dev.azure.com/ehpor/hcipy/_build?definitionId=1>`__ to run our integrated testing suite. The configuration file for this service is **.azure-pipelines.yml**, which shows Azure how to install HCIPy and its dependencies, and on which operating systems and with which Python versions to test HCIPy. The testing itself is done via the `pytest <https://docs.pytest.org/en/latest/>`__ library.

The test suite tests all major features of HCIPy. In the name of efficiency, HCIPy does not contain much so-called unit tests, but rather tests high-level behaviour, which implicitly tests low-level functions as well. For example, for the atmospheric model, we test whether then variance of the outcoming wavefront conforms to analytic formula, and whether the variance projection on a certain Zernike mode conforms to analytic formula, rather than testing the smallest possible unit of the atmospheric model. Tests are located in **tests/test_[submodule_name].py**, separated by submodule. All tests can be run with:

.. code-block:: shell

    pytest ./tests

To reduce the time taken for the suite of tests, we have separated them into comprehensive (ie. fast) and in-depth (ie. slow) tests. The slow tests are not run on the CI service, but can be run manually on your own machine by passing the flag ``--runslow`` to the pytest script:

.. code-block:: shell

    pytest ./tests --runslow

Test coverage is reported for all branches and pull requests on `Coveralls <https://coveralls.io/github/ehpor/hcipy>`__, based on the tests performed on Linux with Python 3.7 by Travis-CI.

Documentation
-------------

The documentation is located in the **doc** folder as a bunch of ``.rst`` files and Jupyter notebooks which are built into a static HTML website by `Sphinx <https://www.sphinx-doc.org>`__. The documentation is built on Linux with the latest Python version for every push to the Github repository. Every build for a push to the master branch is automatically uploaded to `<https://docs.hcipy.org/dev>`__.

The documentation can be build via a Makefile provided by Sphinx:

.. code-block:: shell

    python setup.py egg_info
    cd doc
    make html
    cd ..

To build the documentation from scratch, you can clean the intermediate and output files with:

.. code-block:: shell

    cd doc
    make clean
    cd ..

Tutorials
~~~~~~~~~

While most files are normal reStructuredText (``rst``) files, tutorials are compiled a little bit differently. Currently, these Jupyter notebook (``.ipynb``) files are compiled into reStructuredText files at the end of the **doc/conf.py** configuration file for Sphinx. This may not be the best way, as Sphinx extensions are designed to cover this use case. Currently, the development effort involved in changing this does not outweigh the, most likely very minor, maintenance risk.

During compilation, the notebook is executed and scraped for the title, description, level and thumbnail figure. The first cell in the notebook must start with ``# [title]``, which will be taken as the title of the tutorial. The next non-empty line is taken as the description of the tutorial. The level and thumbnail figure are taken from the metadata of the notebook, which can be edited with ``Edit->Edit Notebook Metadata`` menu option in the Jupyter Notebook Viewer. The ``level`` property should be one of ``("beginner", "intermediate", "advanced", "expert")``. The thumbnail is a rescaled and cropped version of the last figure in the notebook, unless the ``thumbnail_figure_index`` property is in the notebook metadata. This property indicates the index in the list of figures from the notebook, using Python indexing conventions (ie. 0 is the first figure, -2 is the second-to-last figure, etc...).

All tutorials are compiled to reStructuredText files in the **doc/tutorials** folder. This also includes **doc/tutorials/index.rst**.

API Documentation
~~~~~~~~~~~~~~~~~

The documentation for the API is contained in the docstrings for each function and class in HCIPy that is meant to be used outside of the package. The docstring are written using the `numpydoc format <https://numpydoc.readthedocs.io/en/latest/format.html>`__. These function and class docstrings are compiled by Sphinx using the `sphinx-automodapi <https://github.com/astropy/sphinx-automodapi>`__ extension. This creates reStructuredText files in the **doc/api** folder for each submodule in HCIPy.

Website
-------

The website served on `<hcipy.org>`__ is contained in a separate repository `hcipy-webpage <https://github.com/ehpor/hcipy-webpage>`__. This website serves the documentation for the latest stable, all previous and the development version of HCIPy. The repository contains the main website (`hcipy.org <https://hcipy.org>`__) and the framework for the documentation site (`docs.hcipy.org <https://docs.hcipy.org>`__). The built documentation is not in the repository, due to eventual size constraints of the Github repository. The current website is hosted using `Amazon S3 <https://aws.amazon.com/s3/>`__ (Amazon Simple Storage Service).
