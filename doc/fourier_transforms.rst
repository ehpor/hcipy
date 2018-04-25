.. currentmodule:: hcipy.fourier

Fourier Transforms
==================

Overview
--------

The library implements a variety of Fourier transform algorithms, each of which has its own requirements on the grids it operates on. The class :class:`FourierTransform` serves as a base class for all Fourier transform implementations. 

.. autosummary::
	:toctree: _autosummary

	FourierTransform

.. autosummary::
	:toctree: _autosummary

	make_fourier_transform
	multiplex_for_tensor_fields

Fast Fourier Transform
----------------------

.. autosummary::
	:toctree: _autosummary

	FastFourierTransform
	make_fft_grid

.. autosummary::
	:toctree: _autosummary

	ConvolveFFT
	RotateFFT
	FilterFFT

Matrix Fourier Transform
-------------------------

.. autosummary::
	:toctree: _autosummary

	MatrixFourierTransform

Naive Fourier Transform
-----------------------

.. autosummary::
	:toctree: _autosummary

	NaiveFourierTransform