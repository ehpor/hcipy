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

    Also make sure that the latest CI for the master branch is passing. Build the documentation and check if it is building without errors or problematic warnings.

    .. code-block:: shell

        cd doc
        make clean
        make html
        cd ..

2. Write release notes mimicking other release notes. Add those release notes to the :doc:`changelog <../changelog>` in the documentation and commit these changes.

3.  Fetch the tags on your local git repository and update the version information:

    .. code-block:: shell

        git fetch
        python setup.py egg_info
        python setup.py --version

4.  Re-build the documentation:

    .. code-block:: shell

        cd doc
        make clean
        make html
        cd ..

    Load the built documentation (in *doc/_build/html/index.html*) locally, and make sure that the version number has changed (in the upper left), and that everything looks okay. Then upload to the documentation website:

    .. code-block:: shell

        aws s3 sync --acl public-read --cache-control max-age=2629800,public doc/_build/html s3://docs.hcipy.org/0.5.1

    where ``0.5.1`` has been changed to the correct version number.

5.  Publish a new release on GitHub. Navigate to the repository's Releases page, click "Draft a new release", select the appropriate tag (or create a new one, e.g. ``v0.5.1``), add the release notes, and click "Publish release".

    Upon publishing the release, a GitHub Actions workflow will automatically:

    - Build the source distribution and wheels.
    - Publish the distribution to PyPI using `Trusted Publishing <https://docs.pypi.org/trusted-publishers/>`__.
    - Attach the built distribution files to the GitHub Release.

    No manual upload to PyPI is required.

6.  Update all links on the website (*www/index.html*, *www/news.html* and *docs/stable/index.html*) and add release to list of releases. Upload website to AWS S3:

    .. code-block:: shell

        aws s3 sync --acl public-read --delete --cache-control max-age=604800,public www s3://hcipy.org
        aws s3 sync --acl public-read --cache-control max-age=604800,public docs s3://docs.hcipy.org

7.  Purge the `CloudFlare <https://cloudflare.com>`__ cache for `hcipy.org <https://hcipy.org>`__. This step is not necessary. Without it the website will update in at maximum seven days, due to caching of the old website by CloudFlare.

8.  Update this document with any issues, problems or peculiarities that you encountered for later reference.