[![PyPI Status](https://img.shields.io/pypi/v/hcipy.svg)](https://pypi.org/project/hcipy/)
[![Build Status](https://img.shields.io/travis/ehpor/hcipy/master.svg?logo=travis)](https://travis-ci.org/ehpor/hcipy)
[![Build status](https://img.shields.io/appveyor/ci/ehpor/hcipy/master.svg?logo=appveyor)](https://ci.appveyor.com/project/ehpor/hcipy/branch/master)
[![Coverage Status](https://img.shields.io/coveralls/github/ehpor/hcipy.svg)](https://coveralls.io/r/ehpor/hcipy)
[![Documentation Status](https://img.shields.io/readthedocs/hcipy.svg)](https://hcipy.readthedocs.io)
![License](https://img.shields.io/github/license/ehpor/hcipy.svg)

# High Contrast Imaging for Python (HCIPy)

HCIPy is an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments for astronomy.

The library defines wavefronts and optical elements for defining an optical system, and provides both Fraunhofer and Fresnel diffraction propgators. Polarization is supported using Jones calculus, with polarizers and waveplates included out of the box. It implements atmospheric turbulence using thin infinitely-long phase screens, and can model scintillation using Fresnel propagation between individual layers. Many wavefront sensors are implemented including a Shack-Hartmann and Pyramid wavefront sensor. Implemented coronagraphs include the vortex, Lyot and APP coronagraph.

By including simulation of both adaptive optics and coronagraphy into a single framework, HCIPy allows simulations including feedback from post-coronagraphic focal-plane wavefront sensors to the AO system.

For documentation, see https://hcipy.readthedocs.io.

# Installation

HCIPy can be installed from PyPI as usual:
```
pip install hcipy
```
To install the latest development version from Github:
```
git clone https://github.com/ehpor/hcipy
cd hcipy
python setup.py develop
```
HCIPy can then be updated by running:
```
git pull
python setup.py egg_info
```
inside the hcipy repository.

# Citing

If you use HCIPy for your own research, please include the following acknowledgement in your publication:
> This research made use of HCIPy, an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments (Por et al. 2018).

The BibTeX citation can be found below:
```
@inproceedings{por2018hcipy,
   author = {Por, E.~H. and Haffert, S.~Y. and Radhakrishnan, V.~M. and Doelman, D.~S. and Van Kooten, M. and Bos, S.~P.},
   title = "{High Contrast Imaging for Python (HCIPy): an open-source adaptive optics and coronagraph simulator}",
   booktitle = {Adaptive Optics Systems VI},
   year = 2018,
   series = {Proc. {{SPIE}}},
   volume = 10703
}
```

# Documentation

A complete documentation is currently a work in progress. The current documentation can be found at https://hcipy.readthedocs.io. Alternatively, you can build the documentation yourself by executing:

```
cd doc
make clean
make html
```

Afterwards, the built documentation should now be visible in ``doc/_build/html`` as a static website.

# Testing

HCIPy includes a unit testing framework to make sure that no bugs creep in for previously written code. You can run the full test suite by calling
```
pytest ./tests
```
in the root directory of HCIPy. To run all tests, including tests that may take longer, call instead
```
pytest ./tests --runslow
```

# Contributions

If you have something to add, or want something added to HCIPy, please let us know. We actively support external contributions to HCIPy, whether small or large. To contribute a new feature:

1. Fork it if you are not a collaborator yet.
2. Create your feature branch (``git checkout -b my_new_feature``)
3. Commit your changes as normal.
4. Push to the branch (``git push origin my_new_feature``)
5. Create a new Pull Request on GitHub.
6. We will review the feature.

If you encounter any problems during this process, do not hesitate to contact us.