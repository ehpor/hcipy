.. currentmodule:: hcipy.field

Fields
================

What is a Field?
----------------

HCIPy introduces the notion of Fields, a physical quantity, typically a number or tensor, that has a value for each point in some coordinate system. More specifically, in HCIPy a :class:`Field` is the combination of a :class:`Grid`, which defines specific point in some coordinate system, and a list of values or tensors for each point in that :class:`Grid`.

.. autosummary::
    :toctree: _autosummary

    Field

:class:`Field` is derived from a ``numpy.ndarray``. As such it can be treated as one most of the time. In addition, it includes the following methods and properties:

.. autosummary::
   :toctree: _autosummary
   
   Field.at
   Field.tensor_order
   Field.is_scalar_field
   Field.is_vector_field
   Field.is_valid_field
   Field.shaped

Grids
-----

A :class:`Grid` contains the information on how to handle coordinates from specific coordinate system. It combines this information with actual coordinates, implemented in class :class:`CoordsBase`. Therefore, a grid can transform coordinates from one system to another. The :class:`Grid` provides a base class for each grid with a specific coordinate system.

.. autosummary::
   :toctree: _autosummary

   Grid



.. autosummary::
   :toctree: _autosummary

   CartesianGrid
   PolarGrid

Coordinate Systems
------------------

.. autosummary::
   :toctree: _autosummary

   UnstructuredCoords
   RegularCoords
   SeparatedCoords

Utility functions
-----------------

.. autosummary::
   :toctree: _autosummary
   
   make_pupil_grid
   make_focal_grid
   make_hexagonal_grid
   make_chebyshev_grid

Field generators
----------------

Plotting
--------
