[![PyPI Status](https://img.shields.io/pypi/v/hcipy.svg?logo=pypi&logoColor=white)](https://pypi.org/project/hcipy/)
[![Conda Status](https://img.shields.io/conda/vn/conda-forge/hcipy?logo=anaconda&logoColor=white)](https://anaconda.org/conda-forge/hcipy)
[![Build Status](https://img.shields.io/azure-devops/build/ehpor/hcipy/1/master?logo=azure-pipelines)](https://dev.azure.com/ehpor/hcipy/_build?definitionId=1)
[![Coverage Status](https://img.shields.io/codecov/c/gh/ehpor/hcipy?logo=codecov&logoColor=white)](https://codecov.io/gh/ehpor/hcipy)
[![Documentation Status](https://img.shields.io/badge/docs-latest%20build-brightgreen?logo=read-the-docs&logoColor=white)](https://docs.hcipy.org/dev)
[![License](https://img.shields.io/github/license/ehpor/hcipy.svg?logo=open-source-initiative&logoColor=white)](https://opensource.org/licenses/MIT)

# HCIPy: High Contrast Imaging for Python

![HCIPy banner](https://github.com/ehpor/hcipy/raw/master/doc/hcipy_banner.png "HCIPy banner")

HCIPy is an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments for astronomy.

The library defines wavefronts and optical elements for defining an optical system, and provides both Fraunhofer and Fresnel diffraction propgators. Polarization is supported using Jones calculus, with polarizers and waveplates included out of the box. It implements atmospheric turbulence using thin infinitely-long phase screens, and can model scintillation using Fresnel propagation between individual layers. Many wavefront sensors are implemented including a Shack-Hartmann and Pyramid wavefront sensor. Implemented coronagraphs include the vortex, Lyot and APP coronagraph.

By including simulation of both adaptive optics and coronagraphy into a single framework, HCIPy allows simulations including feedback from post-coronagraphic focal-plane wavefront sensors to the AO system.

The main website is hosted at <https://hcipy.org>. For documentation, see <https://docs.hcipy.org>.

## Team

HCIPy was originally developed by a small team of astronomers at Leiden Observatory, but has since received external constributions from scientists and software developers around the world. For a current list, please visit our [website](https://hcipy.org/team.html).

## Citing

If you use HCIPy for your own research, we ask you to cite the HCIPy proceeding ([Por et al. 2018](https://doi.org/10.1117/12.2314407)). If there is no appropriate place in the body text to cite the proceeding, please include something along the lines of the following in your acknowledgements:

> This research made use of HCIPy, an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments ([Por et al. 2018](https://doi.org/10.1117/12.2314407)).

## Contributions

If you have something to add, or want something added to HCIPy, please let us know using a Github issue. We actively support external contributions to HCIPy, small and large. Please look at the [contributing guide](https://docs.hcipy.org/dev/development/contributing_guide.html) for more information.
