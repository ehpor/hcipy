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