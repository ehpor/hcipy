Performance
===========

Overview
--------

Optics simulations can be computationally intensive, especially for high-resolution optical systems with many propagations. Understanding and optimizing performance is crucial for efficiently running simulations of complex high contrast imaging instruments.

This guide covers two major areas that determine the performance of your simulations when using HCIPy.

1. **Field implementation** - The implementation of the main computation object in HCIPy.
2. **Fourier Transforms** - The main building block of optical propagations.

Field Implementation
--------------------

Originally, HCIPy used :class:`~hcipy.field.OldStyleField`, which subclasses :class:`numpy.ndarray` directly. While this approach provided seamless integration with NumPy operations, it came with several significant limitations. The direct subclassing of numpy.ndarray created architectural constraints that made it difficult to extend functionality and led to known subclassing issues. Most importantly, OldStyleField was limited to NumPy operations only, preventing users from leveraging alternative backends such as GPUs or other array computing libraries.

To address these limitations, HCIPy introduced :class:`~hcipy.field.NewStyleField`, which implements the `Python Array API standard <https://data-apis.org/array-api/latest/>`_. The Python Array API is a standardized interface for array computing that enables interoperability between different array libraries. It defines a common set of operations and behaviors that array libraries can implement, allowing code written to the standard to work with any compliant backend without modification.

The Array API provides several key benefits for HCIPy. First, it enables true backend agnosticism - the same HCIPy code can run on NumPy for CPU computation, CuPy for NVIDIA GPU acceleration, JAX for automatic differentiation and accelerator support, or any other Array API-compliant library. Second, it provides a cleaner, more maintainable architecture that avoids the subclassing issues inherent in OldStyleField. Third, it positions HCIPy for future compatibility as more scientific Python libraries adopt the Array API standard.

NewStyleField works by wrapping array data from any Array API-compliant backend rather than subclassing numpy.ndarray. This means that when you create a Field with a CuPy array, all operations on that Field will use CuPy's GPU-accelerated implementations. When you create a Field with a JAX array, you get JAX's capabilities for automatic differentiation and compilation. The Field itself remains agnostic to which backend is being used, delegating all array operations to the underlying Array API implementation.

.. note::
   **NewStyleField is currently experimental.** While it provides better performance and flexibility through Array API compliance, many :class:`~hcipy.optics.OpticalElement` subclasses do not yet support it. Support for NewStyleField is actively being added to the codebase, but until complete compatibility is achieved, you may encounter limitations when using NewStyleField with certain optical elements.

Using GPU Acceleration
^^^^^^^^^^^^^^^^^^^^^^

The following example demonstrates how to use GPU acceleration with HCIPy. This example uses CuPy, but you can substitute any Array API-compliant backend such as JAX or PyTorch.

First, configure HCIPy to use NewStyleField by creating or editing your configuration file at ``~/.hcipy/config.yaml``:

.. code-block:: yaml

   core:
     use_new_style_fields: true

.. important::
   The choice between OldStyleField and NewStyleField is made at the time HCIPy is imported and **cannot be changed afterwards**. You must set this configuration before importing HCIPy in your Python script.

Now you can create fields using CuPy arrays for GPU acceleration:

.. code-block:: python

   import cupy as cp
   from hcipy import make_pupil_grid, Field

   # Create a field using CuPy arrays
   grid = make_pupil_grid(512)
   field = Field(cp.ones(grid.size), grid)

Operations on this field will automatically use the GPU:

.. code-block:: python

   # Get the backend module for backend-agnostic operations
   xp = field.__array_interface__()

   # Perform GPU-accelerated computations
   result = xp.sin(field) * 2 + 1

In this example, the ``field`` object wraps a CuPy array, so all mathematical operations are executed on the GPU. The ``__array_interface__()`` method returns the backend module (in this case, CuPy), which you can use for backend-agnostic operations that work with both CPU and GPU arrays.

Note that backend selection is explicit - HCIPy does not automatically switch to GPU backends even when available. You must intentionally create fields using a backend that uses GPU acceleration. This design gives you full control over where computations occur and allows you to mix CPU and GPU operations in the same script by creating different fields with different backends.

Fourier Transform Performance
-----------------------------

HCIPy implements *multiple* Fourier transform methods to efficiently handle different optical scenarios. The following table provides a quick comparison:

+--------------------------------------------------+---------------------------------------------------+-------------------------------------------------------------+
| Transform                                        | Grid Requirements                                 | Main Feature                                                |
+==================================================+===================================================+=============================================================+
| :class:`~hcipy.fourier.FastFourierTransform`     | Regularly-spaced Cartesian, FFT-compatible sizes  | Fastest for standard grid-to-grid transforms                |
+--------------------------------------------------+---------------------------------------------------+-------------------------------------------------------------+
| :class:`~hcipy.fourier.MatrixFourierTransform`   | Separated Cartesian, arbitrary spacing            | Flexible transforms between any separated grids             |
+--------------------------------------------------+---------------------------------------------------+-------------------------------------------------------------+
| :class:`~hcipy.fourier.ZoomFastFourierTransform` | Regularly-spaced Cartesian grids                  | Asymptotically faster than an MFT, but only for huge grids. |
+--------------------------------------------------+---------------------------------------------------+-------------------------------------------------------------+
| :class:`~hcipy.fourier.NaiveFourierTransform`    | Any input and output grids                        | Most general but extremely slow; verification only          |
+--------------------------------------------------+---------------------------------------------------+-------------------------------------------------------------+


Automatic Method Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^

HCIPy uses the :py:func:`~hcipy.fourier.make_fourier_transform` function to automatically select the most efficient Fourier transform method based on input and output grid characteristics. This function analyzes grid properties and chooses the algorithm that will provide the best performance for that specific propagation.

By default, :py:func:`~hcipy.fourier.make_fourier_transform` estimates the execution time of each available method based on theoretical complexity estimates. You can also use the ``planner='measure'`` option to perform actual timing measurements for more accurate selection, though this takes longer.

The estimated optimal switching point between Fourier transform methods varies significantly by hardware. A method that is fastest on one machine may be slower on another due to differences in CPU architecture, memory bandwidth, and cache sizes. Currently, the execution time is modeled using measured execution times on a Macbook M2 Pro. To optimize Fourier transform performance for your specific hardware, use the performance tuning tool:

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

These coefficients describe the performance characteristics of each method on your hardware. To use the newly tuned parameters, add them to your HCIPy configuration file. The configuration file is typically located at ``~/.hcipy/config.yaml`` or in your project's configuration directory. Add the tuning results under the ``fourier`` section.

Once configured, HCIPy will automatically use your tuned performance models to select the optimal Fourier transform method for each propagation.
