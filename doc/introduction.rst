Introduction
===============

What is HCIPy?
------------------

HCIPy is an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments for astronomy.

The library defines wavefronts and optical elements for defining an optical system, and provides both Fraunhofer and Fresnel diffraction propgators. Polarization is supported using Jones calculus, with polarizers and waveplates included out of the box. It implements atmospheric turbulence using thin infinitely-long phase screens, and can model scintillation using Fresnel propagation between individual layers. Many wavefront sensors are implemented including a Shack-Hartmann and Pyramid wavefront sensor. Implemented coronagraphs include the vortex, Lyot and APP coronagraph.

By including simulation of both adaptive optics and coronagraphy into a single framework, HCIPy allows simulations including feedback from post-coronagraphic focal-plane wavefront sensors to the AO system.
