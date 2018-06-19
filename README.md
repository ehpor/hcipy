[![Build Status](https://travis-ci.org/ehpor/hcipy.svg?branch=master)](https://travis-ci.org/ehpor/hcipy)
[![Build status](https://ci.appveyor.com/api/projects/status/gs01jf6mgpskwm1d/branch/master?svg=true)](https://ci.appveyor.com/project/ehpor/hcipy/branch/master)
[![Coverage Status](https://coveralls.io/repos/ehpor/hcipy/badge.svg?branch=master)](https://coveralls.io/r/ehpor/hcipy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/hcipy/badge/?version=latest)](https://hcipy.readthedocs.io/en/latest/?badge=latest)

# HCIPy

This Python package provides a common framework for performing optical propagation simulations, meant for high contrast imaging.

> **Caution**: this library is still under heavy development. Many functions are untested and may not work as expected. Testing and adding functionality will continue over the next few weeks.

# Installation
First clone the git repository to your own computer as:

```
git clone https://gitlab.strw.leidenuniv.nl/por/hcipy.git
```
Installation is performed as any python package:

```
python setup.py install
```

For continuous updates however, it is easier to avoid having to uninstall and reinstall every time there is an update. For updates to work seamlessly, do:

```
python setup.py develop
```

This will install `hcipy` as a symbolic link to the current directory. Updates can then be done by:

```
git pull
```

# Documentation

A complete documentation is currently a work in progress, and is written in reStructuredText and compiled using Sphinx. The documentation can be built by:

```
cd doc
make clean
make html
```

Afterwards, the built documentation should now be visible in ``doc/_build/html`` as a static website.

# Contributing

We are working in a [feature branching workflow](https://www.atlassian.com/git/tutorials/comparing-workflows#feature-branch-workflow). This means that each feature will be developed on its own branch to be merged later into the *master* branch. To start contribution on a new feature:

```
git checkout -b new_feature
```
where `new_feature` is the name of the feature that you want to develop. You can then use normal `git` commands for commits. Use
```
git push -u origin new_feature
```
to add your feature to the repository on Gitlab for all other to see. Use
```
git push
```
afterwards if you want to push updates to the server. When you are done with development, issue a merge request on the Gitlab website. The merge request will be granted if there are no conflicts with existing code and formatting.
