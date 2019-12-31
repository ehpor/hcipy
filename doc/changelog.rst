Changelog
=========

0.3 (Jan x, 2020)
-----------------



0.2 (Aug 7, 2019)
-----------------

This HCIPy release contains lots of new functionality, performance improvements and bug fixes. See the list of major changes below for a summary. Currently Python 2.7, 3.5-3.7 are supported. Our intention is to drop support for Python 2.7 in the near future.

List of major changes:
~~~~~~~~~~~~~~~~~~~~~~

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