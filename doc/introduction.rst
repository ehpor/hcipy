Introduction
===============

What is HCIPy?
------------------

HCIPy is a framework written in Python for high contrast imaging simulation work. It implements adaptive optics simulation, coronagraphy and optical diffraction calculations.

Installation
----------------------

Installation is done as any standard Python package. First clone the git repository with::

    git clone https://gitlab.strw.leidenuniv.nl/por/hcipy.git

Installation can then be done by::

    python setup.py develop

If you don't intend to receive continous updates, you can also do::

    python setup.py install

To receive updates, just pull the git repository, as::

    git pull

Contributing
------------

Contributions are strongly encouraged. Contributions in this case must be library style code, using the PEP8 code style conventions. Simulation code for personal use will not be accepted.

This package is developed using a `feature branching model <https://www.atlassian.com/git/tutorials/comparing-workflows#feature-branch-workflow>`_. This means that each feature will be developed on its own branch to be merged later into the *master* branch. To start contribution on a new feature of fix, do::

    git checkout -b new_feature

This will start a new branch named ``new_feature``. A feature must be an accurate description of the feature you want to add. Normal ``git`` commits can then be used to develop the feature locally. Use::

    git push -u origin new_feature

to add your feature to the repository on Gitlab for all others to see. Use::

    git push

afterwards  if you want to push updates to the server. When you are done with development, issue a merge request on the Gitlab website. The merge request will be granted if there are no conflicts with existing code and formatting requirements.