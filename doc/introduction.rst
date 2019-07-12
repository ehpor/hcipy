Introduction
===============

What is HCIPy?
------------------


HCIPy is an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments for astronomy.

The library defines wavefronts and optical elements for defining an optical system, and provides both Fraunhofer and Fresnel diffraction propgators. Polarization is supported using Jones calculus, with polarizers and waveplates included out of the box. It implements atmospheric turbulence using thin infinitely-long phase screens, and can model scintillation using Fresnel propagation between individual layers. Many wavefront sensors are implemented including a Shack-Hartmann and Pyramid wavefront sensor. Implemented coronagraphs include the vortex, Lyot and APP coronagraph.

By including simulation of both adaptive optics and coronagraphy into a single framework, HCIPy allows simulations including feedback from post-coronagraphic focal-plane wavefront sensors to the AO system.

Installation
----------------------

HCIPy can be installed from PyPI as usual:

.. code-block:: shell

    pip install hcipy

To install the latest development version from Github, however, do:

.. code-block:: shell

    git clone https://github.com/ehpor/hcipy
	cd hcipy
    python setup.py develop

If you don't intend to receive continous updates, you can also do:

.. code-block:: shell

    python setup.py install

To receive updates, just pull the git repository, as:

.. code-block:: shell

    git pull

Citing
------

If you use HCIPy for your own research, please include the following acknowledgement in your publication:

    This research made use of HCIPy, an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments (Por et al. 2018).

The BibTeX citation can be found below:

.. code-block:: bib

	@inproceedings{por2018hcipy,
		author = {Por, E.~H. and Haffert, S.~Y. and Radhakrishnan, V.~M. and Doelman, D.~S. and Van Kooten, M. and Bos, S.~P.},
		title = "{High Contrast Imaging for Python (HCIPy): an open-source adaptive optics and coronagraph simulator}",
		booktitle = {Adaptive Optics Systems VI},
		year = 2018,
		series = {Proc. {{SPIE}}},
		volume = 10703,
		doi = {10.1117/12.2314407},
		URL = {https://doi.org/10.1117/12.2314407}
	}

The Team
--------

HCIPy was originally developed by a small team of astronomers at Leiden Observatory, but has since received external constributions from scientists and software developers around the world. Development takes place on Github at http://github.com/ehpor/hcipy. If you have something to add, or want something added to HCIPY, please let us know. We actively support external contributions to HCIPy, whether small or large.

Direct contributors to HCIPy code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 * Emiel Por (@ehpor)
 * Sebastiaan Haffert (@syhaffert)
 * Steven Bos (@spbos)
 * Maaike van Kooten (@vkooten)
 * Vikram Radhakrishnan (@VikramRadhakrishnan)
 * David Doelman (@dsdoelman)
 * Christoph Keller (@cukeller)
 * Iva Laginja (@ivalaginja)
 * Remi Soummer (@RemiSoummer)
 * Kathryn St Laurent (@kstlaurent)
 * Matthew Kenworthy (@mkenworthy)
 * David Kleingeld (@dskleingeld)

Testing and QA
^^^^^^^^^^^^^^
 * Aditya Sengupta (@aditya-sengupta)
 * Rebecca Jensen-Clem (@rmjc)
 * Jamie Noss (@jamienoss)
 * Fedde Fagginger Auer (@fjfaggingerauer)
 * Joost Wardenier (@jpw96)