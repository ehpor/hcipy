
Tutorials
=========

These tutorials demonstrate the features of HCIPy in the context of a standard workflow.

.. toctree::
    :maxdepth: 1
    :hidden:

    BroadbandTelescopePSF/BroadbandTelescopePSF
    PyramidWFS/PyramidWFS
    SegmentedDMs/SegmentedDMs



.. only:: html

    .. container:: tutorial_item

        :doc:`Making a broadband telescope point spread function <BroadbandTelescopePSF/BroadbandTelescopePSF>`

        .. container:: tutorial_thumbnail

            .. figure:: BroadbandTelescopePSF/thumb.png

        .. container:: tutorial_description
			
            We will introduce the basic elements in HCIPy and produce a broadband point spread function for the Magellan telescope.

.. only:: html

    .. container:: tutorial_item

        :doc:`Wavefront sensing with a Pyramid wavefront sensor <PyramidWFS/PyramidWFS>`

        .. container:: tutorial_thumbnail

            .. figure:: PyramidWFS/thumb.png

        .. container:: tutorial_description
			
            We will simulate a closed-loop adaptive optics system, based on the the Magellan Adaptive Optics Extreme (MagAO-X) system, that uses an unmodulated pyramid wavefront sensor with a 2k-MEMS DM.

.. only:: html

    .. container:: tutorial_item

        :doc:`Segmented deformable mirrors <SegmentedDMs/SegmentedDMs>`

        .. container:: tutorial_thumbnail

            .. figure:: SegmentedDMs/thumb.png

        .. container:: tutorial_description
			
            We will use segmented deformable mirrors and simulate the PSFs that result from segment pistons and tilts. We will compare this functionality against Poppy, another optical propagation package.
