How to make a new release?
==========================

This page is intended for the maintainer of HCIPy, and contains step-by-step instructions on how to release a new version of HCIPy.

1.  Pull the latest version of the master branch.

    .. code-block:: shell

        git checkout master
        git pull

    Make sure that all unit tests are functioning without errors, including slow tests:

    .. code-block:: shell

        pytest ./tests --runslow

    Also make sure that the latest CI for the master branch on `Azure Pipelines <https://dev.azure.com/ehpor/hcipy/_build?definitionId=1>`__ is passing. Build the documentation and check if it is building without errors or problematic warnings.

    .. code-block:: shell

        cd doc
        make clean
        make html
        cd ..

2. Write release notes mimicking other release notes. Add those release notes to the :doc:`changelog <../changelog>` in the documentation and commit these changes.

3.  Add tag information on Github. You can do this by releasing a new version. Add the written release notes, mimicking previous release notes. After you've released the new version, a new tag will have been added on the Github repository.

4.  Fetch the tags on your local git repository. Update the version information, and check that the version was changed:

    .. code-block:: shell

        git fetch
        python setup.py egg_info
        python setup.py --version

5.  Re-build the documentation:

    .. code-block:: shell

        cd doc
        make clean
        make html
        cd ..

    Load the built documentation (in *doc/_build/html/index.html*) locally, and make sure that the version number has changed (in the upper left), and that everything looks okay. Then upload to the documentation website:

    .. code-block:: shell

        aws s3 sync --acl public-read --cache-control max-age=2629800,public doc/_build/html s3://docs.hcipy.org/0.5.1

    where ``0.5.1`` has been changed to the correct version number.

6.  Build the source distribution and wheels:

    .. code-block:: shell

        python3 -m build

    Then submit to PyPI:

    .. code-block:: shell

        python -m twine upload dist/*

    Enter username and password, and everything will be uploaded. Then add the source distribution and wheel to the Github release as assets.

7.  Update all links on the website (*www/index.html*, *www/news.html* and *docs/stable/index.html*) and add release to list of releases. Upload website to AWS S3:

    .. code-block:: shell

        aws s3 sync --acl public-read --delete --cache-control max-age=604800,public www s3://hcipy.org
        aws s3 sync --acl public-read --cache-control max-age=604800,public docs s3://docs.hcipy.org

8.  Purge the `CloudFlare <https://cloudflare.com>`__ cache for `hcipy.org <https://hcipy.org>`__. This step is not necessary. Without it the website will update in at maximum seven days, due to caching of the old website by CloudFlare.

9.  Update this document with any issues, problems or peculiarities that you encountered for later reference.
