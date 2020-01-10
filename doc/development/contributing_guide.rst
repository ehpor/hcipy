Contributing guide
==================

You don't have to be an expert or know the internal workings of HCIPy to provide a worthwhile contribution to HCIPy. We welcome bug reports and/or fixes, changes to documentation, new optical elements and other improvements.

Reporting bugs or requesting enhancements
-----------------------------------------

Bug reports or enhancement requests can be made on `Github <https://github.com/ehpor/>`__. If possible, bug reports should contain a code example or an error traceback to give us information on how to fix the bug. Enhancement requests are requests for a new feature in HCIPy. Depending on the amount of effort it would take to implement this feature, we may decide to implement it. Otherwise, it'll get put on a list for later implementation, when the need becomes greater, or you'll have to implement the feature yourself.

Contributing code
-----------------

The first step in making a code contribution to HCIPy is to get in touch with us. Create an issue with the bug you want to fix, or the feature you want to add. You can indicate in this Github issue that you'd be willing to do the work. Getting in touch early on ensures that your contribution will not be in vain. For example, your bug fix could be for code that we intend to remove or rewrite soon anyway, or your enhancement could be something that we are working on already. If you think the bug or enhancement is low impact, you may choose to ignore this advice.

We are working in a feature-branching workflow. This means that pushing to the master branch directly is prohibited. Pull requests from feature branches are used to merge changes into the master branch. 

1.  Depending on if you are a collaborator on the HCIPy repository, you may need to fork HCIPy. You can do this by clicking ``Fork`` in the repository on Github. If you are already a collaborator, you do not have fork the HCIPy repository.

2.  Create your *feature branch*:

    .. code-block:: shell

        git checkout -b my_new_feature
    
    where ``my_new_feature`` is a short name for the new feature or bug report.

3.  Commit your changes as usual with git. Do not commit a whole feature consisting of many-hundreds of new lines of code in a single commit. Useful commits should show the line of development separated in logical parts. See :ref:`git style<git-style>` for more information on our preferences.

4.  Push the *feature branch* to the remote repository:

    .. code-block:: shell

        git push origin my_new_feature

5.  Create a new *pull request* on Github. Please mention any Github issue that this solves, so that the pull request and issue are linked by Github.

6.  We will review the changes.

There are a few important things to take note of when introducing new code:

* Follow the existing :ref:`HCIPy coding style<coding-style>`.

* Add docstrings for each class and function that is available outside of the HCIPy package.

* Add unit tests for added code. The test coverage should not have decreased due to your contribution without a good reason.

* Make sure that the documentation still builds, and that the full testing suite still run as expected.

* Any new optical element should have its ``backward()`` function implemented, unless the physical object does not support such an operation. Please refer to the :doc:`"Creating your own optical elements" tutorial<../tutorials/CreatingYourOwnOpticalElements/CreatingYourOwnOpticalElements>` on how to do this.

* For tutorials: make sure that you clean all outputs of the notebook. This is to avoid growing the size of the repository too quickly.
