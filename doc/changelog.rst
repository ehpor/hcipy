Changelog
=========

0.5.1 (Oct 13, 2022)
--------------------

What's Changed
~~~~~~~~~~~~~~

* Return mask on all exit paths by `@ehpor <https://github.com/ehpor>`__ in `#156 <https://github.com/ehpor/hcipy/pull/156>`__


**Full Changelog**: `v0.5.0...v0.5.1 <https://github.com/ehpor/hcipy/compare/v0.5.0...v0.5.1>`__

0.5.0 (Aug 18, 2022)
--------------------

This HCIPy release provides further speed improvements for Fourier transforms, among some other minor improvements and bugfixes. See the list of all changes below for a summary. This release can now also be installed with Conda. We dropped support for Python 3.6. This version supports Python 3.7+.

List of major changes
~~~~~~~~~~~~~~~~~~~~~

* Further speed improvements using NumExpr and in-place operations (`@ehpor <https://github.com/ehpor>`__).
* Autodetect FFTs when doing general Fourier transforms (`@ehpor <https://github.com/ehpor>`__)
* Renamed some of the aperture functions to be more uniform (`@ehpor <https://github.com/ehpor>`__)
* Added file IO for Field, Grid and ModeBasis (`@ehpor <https://github.com/ehpor>`__)
* Allow the use of 32bit floating point Wavefronts (`@ehpor <https://github.com/ehpor>`__)
* Added the LUVOIR-B, GMT, TMT, ELT, Hale, HabEx, Hubble and JWST telescope apertures (`@ivalaginja <https://github.com/ivalaginja>`__, `@syhaffert <https://github.com/syhaffert>`__ & `@ehpor <https://github.com/ehpor>`__)
* Improved the speed of spiders and segmented apertures by 4x (`@ehpor <https://github.com/ehpor>`__)
* Knife edge coronagraph now works in all four directions (`@syhaffert <https://github.com/syhaffert>`__ & `@ehpor <https://github.com/ehpor>`__))
* Improved support for animations (`@ehpor <https://github.com/ehpor>`__)
* Added focal-length parameter to Lyot coronagraph (`@syhaffert <https://github.com/syhaffert>`__)
* Added a telescope pupil introductory tutorial (`@ehpor <https://github.com/ehpor>`__)
* Allow step-index fibers to be put at arbitrary positions (`@syhaffert <https://github.com/syhaffert>`__)
* Fixed the PyWFS tutorial (`@syhaffert <https://github.com/syhaffert>`__)
* Added CaF2 to the materials (`@syhaffert <https://github.com/syhaffert>`__)
* Fixed deprecation warnings for Numpy 1.20 (`@ehpor <https://github.com/ehpor>`__)
* Added release for Conda (`@ehpor <https://github.com/ehpor>`__)
* Reading specific extensions of Fits files (`@syhaffert <https://github.com/syhaffert>`__)
* CI and linting automation maintenance and upgrades (`@ehpor <https://github.com/ehpor>`__)
* Added support for Python 3.10 and removed support for Python 3.6 (`@ehpor <https://github.com/ehpor>`__)
* Added conda-forge installation (`@ehpor <https://github.com/ehpor>`__)

0.4.0 (Feb 22, 2021)
--------------------

This HCIPy release provides significant speed improvements for Fourier transforms, propagations and the vortex coronagraphs, among some other minor improvements and bugfixes. See the list of all changes below for a summary. We dropped support for Python 3.5. This version supports Python 3.6+.

List of major changes
~~~~~~~~~~~~~~~~~~~~~

* Major bug fix in caching algorithm for optical elements (`@ehpor <https://github.com/ehpor>`__)
* Major improvement in computational efficiency of FFTs and MFTs (`@ehpor <https://github.com/ehpor>`__)
* FFTs now use MKL when installed (`@ehpor <https://github.com/ehpor>`__)
* Fourier transforms now retain bit depth and do not automatically cast to double precision (`@ehpor <https://github.com/ehpor>`__)
* A new class FourierFilter for efficient correlations and convolutions  (`@ehpor <https://github.com/ehpor>`__)
* The vortex coronagraphs now use smooth windowing for their multiscale Fourier transforms (`@ehpor <https://github.com/ehpor>`__)
* Added VLT aperture (`@dsdoelman <https://github.com/dsdoelman>`__, `@ehpor <https://github.com/ehpor>`__, `@syhaffert <https://github.com/syhaffert>`__ & `@jmilou <https://github.com/jmilou>`__)
* The perfect coronagraph can now handle polarized wavefronts (`@ehpor <https://github.com/ehpor>`__)
* Optical elements can now be pickled for multiprocessing (`@ehpor <https://github.com/ehpor>`__)
* Detectors can now do subsampling of the incoming light  (`@spbos <https://github.com/spbos>`__ & `@ehpor <https://github.com/ehpor>`__)
* Fixed the OD wavefront sensors (`@syhaffert <https://github.com/syhaffert>`__)
* Fixed bug for the ZernikeWavefrontSensor when physical units were used (`@yinzi-xin <https://github.com/yinzi-xin>`__ & `@ehpor <https://github.com/ehpor>`__)
* Two new tutorials (`@jmilou <https://github.com/jmilou>`__ & `@ehpor <https://github.com/ehpor>`__)
* Tutorial notebooks are now allowed to use up to 10mins per cell (`@ehpor <https://github.com/ehpor>`__)
* Added support for Python 3.8 and 3.9 (`@ehpor <https://github.com/ehpor>`__)
* Migration to Azure Pipelines for CI tests on all operating systems (`@ehpor <https://github.com/ehpor>`__)
* Miscellaneous minor bug fixes (`@syhaffert <https://github.com/syhaffert>`__, `@jmilou <https://github.com/jmilou>`__ & `@ehpor <https://github.com/>`__)

0.3.1 (Apr 2, 2020)
-------------------

This HCIPy release fixes a major bug in the caching algorithm for backwards propagation, among some other minor improvements. See the list of all changes below for a summary. This version supports Python 3.5+.

List of major changes
~~~~~~~~~~~~~~~~~~~~~

* Bug fix in the caching algorithm when using backwards propagation for the first time (`@ehpor <https://github.com/ehpor>`__)
* Add input/output to FITS and ASDF files for Field, Grid and ModeBasis (`@ehpor <https://github.com/ehpor>`__)
* Fixed normalization for the Fourier, Gaussian-Hermite and Gaussian-Laguerre mode bases (`@ehpor <https://github.com/ehpor>`__)
* Allow anamorphic magnification (`@ehpor <https://github.com/ehpor>`__)
* Allow variable wind speed for atmospheric layers (`@syhaffert <https://github.com/syhaffert>`__)
* Add plotting utilities for pupils and PSFs (`@ehpor <https://github.com/ehpor>`__)
* Fix FFMpegWriter for MacOS (`@ehpor <https://github.com/ehpor>`__ & `@stevenbos <https://github.com/stevenbos>`__)
* Minor documentation clarifications (`@ehpor <https://github.com/ehpor>`__)
* Increased test coverage (`@ehpor <https://github.com/ehpor>`__)

0.3.0 (Jan 21, 2020)
--------------------

This HCIPy release contains full Stokes polarization support, some performance improvements and bug fixes. See the list of major changes below for a summary. This version supports Python 3.5+. This and later releases may still work with Python 2.7, but this is not a supported use case.

List of major changes
~~~~~~~~~~~~~~~~~~~~~

* Many new tutorials (everyone)
* Windows support for GifWriter (`@jamienoss <https://github.com/jamienoss>`__)
* Support for partially polarized wavefronts using Stokes vectors (`@stevenbos <https://github.com/stevenbos>`__, `@dsdoelman <https://github.com/dsdoelman>`__ & `@ehpor <https://github.com/ehpor>`__)
* Addition of general Jones matrix optical elements (`@dsdoelman <https://github.com/dsdoelman>`__, `@stevenbos <https://github.com/stevenbos>`__ & `@ehpor <https://github.com/ehpor>`__)
* Addition of wave plates (`@dsdoelman <https://github.com/dsdoelman>`__, `@stevenbos <https://github.com/stevenbos>`__ & `@ehpor <https://github.com/ehpor>`__)
* Addition of linear and circular polarizers and beam splitters (`@dsdoelman <https://github.com/dsdoelman>`__ & `@stevenbos <https://github.com/stevenbos>`__)
* Addition of a vector vortex coronagraph (`@ehpor <https://github.com/ehpor>`__)
* Addition of a vector Zernike wavefront sensor (`@dsdoelman <https://github.com/dsdoelman>`__)
* Automated testing of all polarization optical elements (`@stevenbos <https://github.com/stevenbos>`__, `@dsdoelman <https://github.com/dsdoelman>`__ & `@ehpor <https://github.com/ehpor>`__)
* Bug fix in the Zernike wavefront sensor (`@dsdoelman <https://github.com/dsdoelman>`__)
* Addition of a material refractive index catalog (`@syhaffert <https://github.com/syhaffert>`__ & `@ehpor <https://github.com/ehpor>`__)
* Addition of better microlenses (`@syhaffert <https://github.com/syhaffert>`__)
* Addition of better single/few mode fibers (`@syhaffert <https://github.com/syhaffert>`__)
* Fixed APP optimization code (`@dsdoelman <https://github.com/dsdoelman>`__)
* Improved handling of agnostic optical elements (`@ehpor <https://github.com/ehpor>`__)
* Improved of documentation, including developer notes (`@ehpor <https://github.com/ehpor>`__)
* Removal of much old, unused code (`@ehpor <https://github.com/ehpor>`__)
* Subpixel interpolation of atmospheric wavefronts is now default (`@ehpor <https://github.com/ehpor>`__)
* Speed improvements for calculating telescope apertures of up to 12x (`@ehpor <https://github.com/ehpor>`__)
* Evaluation of field generators can now be supersampled by any factor (`@ehpor <https://github.com/ehpor>`__)
* Bug fix where data files were not found on pip-installed versions of HCIPy (`@ehpor <https://github.com/ehpor>`__)

0.2 (Aug 7, 2019)
-----------------

This HCIPy release contains lots of new functionality, performance improvements and bug fixes. See the list of major changes below for a summary. Currently Python 2.7, 3.5-3.7 are supported. Our intention is to drop support for Python 2.7 in the near future.

List of major changes
~~~~~~~~~~~~~~~~~~~~~

* Many bug fixes (everyone)
* Many additions to the documentation (everyone)
* Atmospheric layers are now tested to conform to analytical formulas for the total wavefront error and variance for each Zernike mode (`@ehpor <https://github.com/ehpor>`__ & `@vkooten <https://github.com/vkooten>`__)
* The InfiniteAtmosphericLayer now supports subpixel interpolation to improve predictive control simulations (`@ehpor <https://github.com/ehpor>`__)
* The phase shift for an AtmosphericModel can now be retrieved (`@ehpor <https://github.com/ehpor>`__)
* Calculating of influence functions for a tilted DM (`@ehpor <https://github.com/ehpor>`__)
* Polygonal and hexagonal apertures can now be rotated with arbitrary angles (`@ehpor <https://github.com/ehpor>`__)
* An optical element that performs (de)magnification was added (`@syhaffert <https://github.com/syhaffert>`__)
* Coronagraphs are now included in automatic testing for starlight suppression capabilities (`@ehpor <https://github.com/ehpor>`__ & `@cukeller <https://github.com/cukeller>`__)
* HiCAT and LUVOIR-A pupils and Lyot stops were added (`@kstlaurent <https://github.com/kstlaurent>`__ & `@RemiSoummer <https://github.com/RemiSoummer>`__)
* A segmented deformable mirror was added (`@ivalaginja <https://github.com/ivalaginja>`__)
* Much improved (modulated) Pyramid WFS models (`@syhaffert <https://github.com/syhaffert>`__)
* Added tip-tilt mirror (`@syhaffert <https://github.com/syhaffert>`__)
* Improved Zernike WFS model based on semi-analytical Lyot coronagraph propagation (`@ehpor <https://github.com/ehpor>`__)
* A mode basis can now be sparse (`@ehpor <https://github.com/ehpor>`__)
* All segmented pupils can now also return the pupil by segment (`@ehpor <https://github.com/ehpor>`__, `@kstlaurent <https://github.com/kstlaurent>`__, `@ivalaginja <https://github.com/ivalaginja>`__ & `@RemiSoummer <https://github.com/RemiSoummer>`__)
* Reduced memory usage of evaluate__supersampled() (`@ehpor <https://github.com/ehpor>`__)
* Removal of deprecated atmospheric model (`@ehpor <https://github.com/ehpor>`__)
* Improved Fresnel propagation model that avoids/corrects for aliasing (`@syhaffert <https://github.com/syhaffert>`__ & `@ehpor <https://github.com/ehpor>`__)
* Automated testing of Fraunhofer, Fresnel and ASP propagators (`@ehpor <https://github.com/ehpor>`__ & `@syhaffert <https://github.com/syhaffert>`__)
* Introduction of grid-agnostic optical elements (`@ehpor <https://github.com/ehpor>`__)
* Added a knife-edge Lyot coronagraph model (`@ehpor <https://github.com/ehpor>`__)
* All telescope pupils are now included in the automated testing framework (`@ehpor <https://github.com/ehpor>`__)
* Faster calculation of Zernike modes using q-recursive algorithm (`@ehpor <https://github.com/ehpor>`__)
* Accelerated APP optimization based on Douglas-Rachford operator splitting (`@cukeller <https://github.com/cukeller>`__)
* Add methods for linear and nearest interpolation for Fields (`@ehpor <https://github.com/ehpor>`__)

0.1 (Jul 5, 2018)
-----------------

This is the first open-source release of HCIPy.