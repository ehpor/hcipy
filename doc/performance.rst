Performance
==========

This section describes performance considerations for HCIPy, including how to optimize computations and use GPU backends for accelerated processing.

Introduction
------------

HCIPy simulations can be computationally intensive, especially for high-resolution optical systems with many propagations. Understanding and optimizing performance is crucial for efficiently running simulations of complex high contrast imaging instruments.

This guide covers two major performance optimization areas in HCIPy: the Field system which provides flexibility through the Array API, and the Fourier transform machinery which automatically selects optimal algorithms for different optical propagation scenarios.

Field Performance and Backend Selection
---------------------------------------

HCIPy provides two Field implementations: :class:`hcipy.field.OldStyleField` and :class:`hcipy.field.NewStyleField`.

.. note::
   **NewStyleField is currently experimental.** While it provides better performance and flexibility through Array API compliance, many :class:`hcipy.optics.OpticalElement` subclasses do not yet support it. Support for NewStyleField is actively being added to the codebase, but until complete compatibility is achieved, you may encounter limitations when using NewStyleField with certain optical elements. If you encounter issues, you can fall back to OldStyleField by setting ``config.core.use_new_style_fields = False``.

NewStyleField leverages the `Python Array API standard <https://data-apis.org/array-api/latest/>`_, enabling it to work with multiple backends (NumPy, CuPy, JAX) rather than being limited to a single implementation. This allows users to explicitly select the optimal backend for their specific hardware and use case.

Comparison of Field Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**OldStyleField:**
- Subclasses numpy.ndarray directly
- Limited to NumPy operations
- Inflexible architecture with known subclassing issues

**NewStyleField:**
- Implements the Python Array API standard
- Works with multiple backends (NumPy, CuPy, JAX)
- Users explicitly select backends rather than having them automatically chosen
- More flexible and future-proof architecture

GPU Acceleration
^^^^^^^^^^^^^^^^

HCIPy supports GPU acceleration through the `Array API <https://data-apis.org/array-api/latest/>`_ interface. When GPU-accelerated backends like CuPy are available, users can select them to significantly accelerate computations.

To use GPU acceleration:

1. Install CuPy:

   .. code-block:: bash

      pip install cupy

2. Configure HCIPy to use NewStyleField by editing your configuration file (typically located at ``~/.hcipy/config.yaml``):

   .. code-block:: yaml

      core:
        use_new_style_fields: true

   .. important::
      The choice between OldStyleField and NewStyleField is made at the time HCIPy is imported and **cannot be changed afterwards**. You must set this configuration before importing HCIPy in your Python script.

3. Create fields using CuPy arrays:

   .. code-block:: python

      import cupy as cp
      from hcipy import make_pupil_grid, Field

      # Create a field using CuPy arrays for GPU acceleration
      grid = make_pupil_grid(512)
      field = Field(cp.ones(grid.size), grid)

      # Operations will automatically use the GPU
      result = field * 2 + 1

Note that backend selection is explicit - HCIPy does not automatically switch to GPU backends even when available. Users must intentionally create fields using CuPy arrays to utilize GPU acceleration.

Fourier Transform Performance
-----------------------------

HCIPy implements multiple Fourier transform methods to efficiently handle different optical scenarios. Each method has specific strengths and requirements:

**FFT (:class:`~hcipy.fourier.FastFourierTransform`):**
Calculates FFTs with zero-padding and cropping capabilities. Requires the input grid to be regular in Cartesian coordinates. Supports arbitrary dimensionality.

Key features:
- Fastest for regular grid-to-grid transformations with FFT-compatible sizes
- Supports zero-padding (via the ``q`` parameter) to increase sampling in the Fourier domain
- Supports cropping (via the ``fov`` parameter) to limit the field of view in the output
- Optional ``emulate_fftshifts`` mode provides 3x performance boost but with 10x accuracy degradation

**MFT (:class:`~hcipy.fourier.MatrixFourierTransform`):**
Based on Soummer et al. 2007. Requires both input and output grids to be separated in Cartesian coordinates. Currently supports one- and two-dimensional grids.

Key features:
- Works between arbitrary grids (not limited to FFT-compatible sizes)
- Essential when propagating between non-FFT-compatible grids
- Can precompute transformation matrices for 20-30% speedup (uses more memory)
- Can allocate intermediate arrays for 5-10% speedup (uses more memory)
- More flexible than FFT but potentially slower for certain grid combinations

**ZoomFFT (:class:`~hcipy.fourier.ZoomFastFourierTransform`):**
A specialization of the Chirp Z-transform. Requires both input and output grids to be regularly spaced in Cartesian coordinates, but unlike FFT, the spacing can be arbitrary.

Key features:
- Efficiently evaluates small regions of Fourier space without computing the full transform
- Asymptotically faster than MFT for large grids (typically 1k x 1k or larger in each dimension)
- Supports arbitrary dimensionality
- Ideal for high-resolution analysis of specific regions in the focal plane

**NaiveFT (:class:`~hcipy.fourier.NaiveFourierTransform`):**
Calculates the Fourier integral using brute force. Can operate on any coordinate types and is written as generally as possible to act as a "ground truth" for other transforms.

Key features:
- Works on any coordinate system (most general)
- Transformation matrices can be very large, quickly overwhelming computers even for small grids
- Multiple orders of magnitude slower than FFT or MFT
- Uses multiple orders of magnitude more memory
- Should only be used as a last resort for validation or ground truth calculations

.. warning::
   In almost all cases, you should not use NaiveFT. The :class:`~hcipy.fourier.FastFourierTransform` and :class:`~hcipy.fourier.MatrixFourierTransform` classes are multiple orders of magnitude faster and use multiple orders of magnitude less memory. Only use NaiveFT if you know what you are doing and have no other choice.

Automatic Method Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^

HCIPy uses the :py:func:`hcipy.fourier.make_fourier_transform` function to automatically select the most efficient Fourier transform method based on input and output grid characteristics. This function analyzes grid properties and chooses the algorithm that will provide the best performance for that specific propagation.

The decision process considers:
- Grid dimensions and whether they are FFT-friendly (powers of 2)
- The relationship between input and output grid sizes
- Required oversampling factors
- Field of view requirements
- Whether grids are regularly or irregularly spaced
- Grid coordinate system compatibility (Cartesian, separated, etc.)

By default, ``make_fourier_transform`` estimates the execution time of each available method based on theoretical complexity estimates. You can also use the ``planner='measure'`` option to perform actual timing measurements for more accurate selection, though this takes longer.

Performance Tuning
^^^^^^^^^^^^^^^^^^

The optimal switching point between Fourier transform methods varies significantly by hardware. A method that is fastest on one machine may be slower on another due to differences in CPU architecture, memory bandwidth, and cache sizes.

To optimize Fourier transform performance for your specific hardware, use the performance tuning tool:

.. code-block:: bash

   hcipy_tune_fourier

This tool measures the actual performance of different Fourier transform methods across a range of grid sizes and parameters. It runs benchmarks for FFT, MFT, and ZoomFFT operations, then fits performance models to determine the optimal switching points.

The tuning process typically produces output like this:

.. code-block:: text

   Fit results:
     fft:
       a: 1.234
       b: 0.987
       c: -2.109
     mft:
       a: 0.876
       b: 1.234
       c: -1.098
     zfft:
       a: 0.567
       b: 1.543
       c: -0.876

These coefficients describe the performance characteristics of each method on your hardware.

Configuration File Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the tuned parameters, add them to your HCIPy configuration file. The configuration file is typically located at ``~/.hcipy/config.yaml`` or in your project's configuration directory.

Add the tuning results under the ``fourier`` section:

.. code-block:: yaml

   fourier:
     fft:
       a: 1.234
       b: 0.987
       c: -2.109
     mft:
       a: 0.876
       b: 1.234
       c: -1.098
     zfft:
       a: 0.567
       b: 1.543
       c: -0.876

Once configured, HCIPy will automatically use these performance models to select the optimal Fourier transform method for each propagation, ensuring you get the best performance without manual tuning.

When to Re-tune
^^^^^^^^^^^^^^^

You should re-run the performance tuning tool when:
- You upgrade your hardware (CPU, memory)
- You change operating systems or Python versions
- You install different versions of NumPy, SciPy, or other numerical libraries
- You notice performance degradation in your simulations

Performance Best Practices
--------------------------

To get the best performance from HCIPy simulations:

**Use NewStyleField**
Enable NewStyleField in your configuration file for better performance and flexibility:

.. code-block:: yaml

   core:
     use_new_style_fields: true

Remember that this setting must be configured **before** importing HCIPy, as the choice is made at import time.

**Select Appropriate Backends**
Choose the most suitable backend for your hardware:
- Use NumPy for general CPU computations
- Use CuPy for GPU acceleration when available
- Consider JAX for additional optimization features

**Tune Fourier Transforms**
Run ``hcipy_tune_fourier`` to optimize the automatic method selection for your hardware. This is especially important after hardware changes.

**Use Appropriate Grid Sizes**
Select grid sizes that are efficient for your hardware:
- FFT operations are fastest with sizes that are powers of 2
- Very large grids consume significant memory and may cause swapping
- Balance accuracy requirements with computational cost

**Monitor Memory Usage**
Large grids, especially with GPU backends, can quickly exhaust available memory:
- Monitor GPU memory usage when using CuPy
- Consider using smaller grids or batch processing for memory-intensive simulations
- Be aware that complex fields require twice the memory of real fields

**Grid Size Guidelines**
- For FFT operations, prefer grid sizes that are powers of 2 (512, 1024, 2048, etc.)
- Consider the trade-off between resolution and computation time
- Use ZoomFFT for high-resolution analysis of limited regions instead of increasing the entire grid size
