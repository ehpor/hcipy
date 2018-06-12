.. currentmodule:: hcipy.atmosphere

Atmospheric model
==============================

Overview
--------

The atmospheric model implemented in hcipy has two base classes. An :class:`AtmosphericLayer` class represents a single layer at a certain height above the ground. A :class:`MultiLayerAtmosphere` class represents a stack of these layers, and performs propagations between and through this stack.

.. autosummary::
	:toctree: _autosummary

	AtmosphericLayer
	MultiLayerAtmosphere

Finite atmospheric layer
--------------------------

.. autosummary::
	:toctree: _autosummary

	FiniteAtmosphericLayer

Infinite atmospheric layer
----------------------------

.. autosummary::
	:toctree: _autosummary

	InfiniteAtmosphericLayer

Simple adaptive optics
------------------------

.. autosummary::
	:toctree: _autosummary

	ModalAdaptiveOpticsLayer


Stored atmospheric layer
-------------------------

Not implemented yet.