Changelog
=========

0.3.0 (Jan 21, 2020)
--------------------

This HCIPy release contains full Stokes polarization support, some performance improvements and bug fixes. See the list of major changes below for a summary. This version supports Python 3.5+. This and later releases may still work with Python 2.7, but this is not a supported use case.

List of major changes
~~~~~~~~~~~~~~~~~~~~~

* Many new tutorials (everyone)
* Windows support for GifWriter (`@jamienoss <https://github.com/jamienoss>`_)
* Support for partially polarized wavefronts using Stokes vectors (`@stevenbos <https://github.com/stevenbos>`_, `@dsdoelman <https://github.com/dsdoelman>`_ & `@ehpor <https://github.com/ehpor>`_)
* Addition of general Jones matrix optical elements (`@dsdoelman <https://github.com/dsdoelman>`_, `@stevenbos <https://github.com/stevenbos>`_ & `@ehpor <https://github.com/ehpor>`_)
* Addition of wave plates (`@dsdoelman <https://github.com/dsdoelman>`_, `@stevenbos <https://github.com/stevenbos>`_ & `@ehpor <https://github.com/ehpor>`_)
* Addition of linear and circular polarizers and beam splitters (`@dsdoelman <https://github.com/dsdoelman>`_ & `@stevenbos <https://github.com/stevenbos>`_)
* Addition of a vector vortex coronagraph (`@ehpor <https://github.com/ehpor>`_)
* Addition of a vector Zernike wavefront sensor (`@dsdoelman <https://github.com/dsdoelman>`_)
* Automated testing of all polarization optical elements (`@stevenbos <https://github.com/stevenbos>`_, `@dsdoelman <https://github.com/dsdoelman>`_ & `@ehpor <https://github.com/ehpor>`_)
* Bug fix in the Zernike wavefront sensor (`@dsdoelman <https://github.com/dsdoelman>`_)
* Addition of a material refractive index catalog (`@syhaffert <https://github.com/syhaffert>`_ & `@ehpor <https://github.com/ehpor>`_)
* Addition of better microlenses (`@syhaffert <https://github.com/syhaffert>`_)
* Addition of better single/few mode fibers (`@syhaffert <https://github.com/syhaffert>`_)
* Fixed APP optimization code (`@dsdoelman <https://github.com/dsdoelman>`_)
* Improved handling of agnostic optical elements (`@ehpor <https://github.com/ehpor>`_)
* Improved of documentation, including developer notes (`@ehpor <https://github.com/ehpor>`_)
* Removal of much old, unused code (`@ehpor <https://github.com/ehpor>`_)
* Subpixel interpolation of atmospheric wavefronts is now default (`@ehpor <https://github.com/ehpor>`_)
* Speed improvements for calculating telescope apertures of up to 12x (`@ehpor <https://github.com/ehpor>`_)
* Evaluation of field generators can now be supersampled by any factor (`@ehpor <https://github.com/ehpor>`_)
* Bug fix where data files were not found on pip-installed versions of HCIPy (`@ehpor <https://github.com/ehpor>`_)

0.2 (Aug 7, 2019)
-----------------

This HCIPy release contains lots of new functionality, performance improvements and bug fixes. See the list of major changes below for a summary. Currently Python 2.7, 3.5-3.7 are supported. Our intention is to drop support for Python 2.7 in the near future.

List of major changes
~~~~~~~~~~~~~~~~~~~~~

* Many bug fixes (everyone)
* Many additions to the documentation (everyone)
* Atmospheric layers are now tested to conform to analytical formulas for the total wavefront error and variance for each Zernike mode (`@ehpor <https://github.com/ehpor>`_ & `@vkooten <https://github.com/vkooten>`_)
* The InfiniteAtmosphericLayer now supports subpixel interpolation to improve predictive control simulations (`@ehpor <https://github.com/ehpor>`_)
* The phase shift for an AtmosphericModel can now be retrieved (`@ehpor <https://github.com/ehpor>`_)
* Calculating of influence functions for a tilted DM (`@ehpor <https://github.com/ehpor>`_)
* Polygonal and hexagonal apertures can now be rotated with arbitrary angles (`@ehpor <https://github.com/ehpor>`_)
* An optical element that performs (de)magnification was added (`@syhaffert <https://github.com/syhaffert>`_)
* Coronagraphs are now included in automatic testing for starlight suppression capabilities (`@ehpor <https://github.com/ehpor>`_ & `@cukeller <https://github.com/cukeller>`_)
* HiCAT and LUVOIR-A pupils and Lyot stops were added (`@kstlaurent <https://github.com/kstlaurent>`_ & `@RemiSoummer <https://github.com/RemiSoummer>`_)
* A segmented deformable mirror was added (`@ivalaginja <https://github.com/ivalaginja>`_)
* Much improved (modulated) Pyramid WFS models (`@syhaffert <https://github.com/syhaffert>`_)
* Added tip-tilt mirror (`@syhaffert <https://github.com/syhaffert>`_)
* Improved Zernike WFS model based on semi-analytical Lyot coronagraph propagation (`@ehpor <https://github.com/ehpor>`_)
* A mode basis can now be sparse (`@ehpor <https://github.com/ehpor>`_)
* All segmented pupils can now also return the pupil by segment (`@ehpor <https://github.com/ehpor>`_, `@kstlaurent <https://github.com/kstlaurent>`_, `@ivalaginja <https://github.com/ivalaginja>`_ & `@RemiSoummer <https://github.com/RemiSoummer>`_)
* Reduced memory usage of evaluate_supersampled() (`@ehpor <https://github.com/ehpor>`_)
* Removal of deprecated atmospheric model (`@ehpor <https://github.com/ehpor>`_)
* Improved Fresnel propagation model that avoids/corrects for aliasing (`@syhaffert <https://github.com/syhaffert>`_ & `@ehpor <https://github.com/ehpor>`_)
* Automated testing of Fraunhofer, Fresnel and ASP propagators (`@ehpor <https://github.com/ehpor>`_ & `@syhaffert <https://github.com/syhaffert>`_)
* Introduction of grid-agnostic optical elements (`@ehpor <https://github.com/ehpor>`_)
* Added a knife-edge Lyot coronagraph model (`@ehpor <https://github.com/ehpor>`_)
* All telescope pupils are now included in the automated testing framework (`@ehpor <https://github.com/ehpor>`_)
* Faster calculation of Zernike modes using q-recursive algorithm (`@ehpor <https://github.com/ehpor>`_)
* Accelerated APP optimization based on Douglas-Rachford operator splitting (`@cukeller <https://github.com/cukeller>`_)
* Add methods for linear and nearest interpolation for Fields (`@ehpor <https://github.com/ehpor>`_)

0.1 (Jul 5, 2018)
-----------------

This is the first open-source release of HCIPy.