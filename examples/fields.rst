.. currentmodule:: hcipy.field

Fields
================

What is a Field?
----------------

HCIPy introduces the notion of Fields, a physical quantity, typically a number or tensor, that has a value for each point in some coordinate system. More specifically, in HCIPy a :class:`Field` is the combination of a :class:`Grid`, which defines the sampling in N-dimensional space, and a list of values or tensors for each point in that :class:`Grid`.

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

To make it easier to work with tensor- and vector fields, we implemented the following methods:

.. autosummary::
   :toctree: _autosummary

   field_einsum

Coordinates
---------------------

In HCIPy we separate the way of generating points (the actual values on the axes) from the interpretation of those values (the coordinate system). We will first discuss the implementation of the coordinates before combining these with the context of a coordinate system. Coordinates are implemented by :class:`CoordsBase`.

.. autosummary::
   :toctree: _autosummary

   CoordsBase

Coordinates (currently) come in three different flavours: Unstructured, Separated and Regular. These three flavours are implemented as derived classes. The base class allows for testing which flavour the coordinates are using the following methods:

.. autosummary::
   :toctree: _autosummary

   CoordsBase.is_regular
   CoordsBase.is_separated

:class:`CoordsBase` also provides basic functionality for generating points:

.. autosummary::
   :toctree: _autosummary

   CoordsBase.copy
   CoordsBase.size
   CoordsBase.__getitem__
   CoordsBase.__len__
   CoordsBase.reverse

It also overloads math operators for more direct interaction with the values themselves.

.. autosummary::
   :toctree: _autosummary

   CoordsBase.__add__
   CoordsBase.__iadd__
   CoordsBase.__radd__
   CoordsBase.__sub__
   CoordsBase.__isub__
   CoordsBase.__mul__
   CoordsBase.__imul__
   CoordsBase.__rmul__
   CoordsBase.__div__
   CoordsBase.__idiv__

The three flavours are:

.. autosummary::
   :toctree: _autosummary

   UnstructuredCoords
   SeparatedCoords
   RegularCoords

An :class:`UnstructuredCoords` is just a raw list of points. This general case should be avoided in practice if at all possible. A :class:`SeparatedCoords` gives the coordinates some structure. The coordinates are the tensor product of the one-dimensional coordinates. This structure often occurs in nature, and many algorithms can be implemented much more efficiently compared to the unstructured coordinates. A :class:`RegularCoords` gives even more structure to the coordinates. This flavour requires that each point has equal spacing to the next point. This yields even more efficient algorithms and must be prefered above all other flavours.

.. note::
   Regularly-spaced coordinates are by definition also separated. This means that an instance of :class:`RegularCoords` can be treated as an instance of :class:`SeparatedCoords` in every way.

Grids
-----

Grids combine the coordinates described above with the N-dimensional area (or *weight*) of each of those points. Additionally it adds the context of a coordinate system. Again, coordinate systems are implemented as derived classes from the base class :class:`Grid`.

.. autosummary::
   :toctree: _autosummary

   Grid

These weights can also be let undefined. If they are needed by some other piece of code, they are generated on the fly, if possible.

.. autosummary::
   :toctree: _autosummary

   Grid.weights

Checking for a specific coordinate system and switching between them can be accomplished using the following member methods:

.. autosummary::
   :toctree: _autosummary

   Grid.is_
   Grid.as_

.. caution::
   The flavour of the corresponding coordinates can, and often are, changed when changing coordinate systems. This is unavoidable as, for example, regularly-spaced coordinates in a Cartesian coordinate system have no particular structure in a Polar coordinate system. When changing the new Polar grid back to a Cartesian coordinate system, the apparent structure is lost, even though it is still clearly visible in the location of the points themselves.

:class:`Grid` implements a few convenience methods to get information about the underlying coordinates:

.. autosummary::
   :toctree: _autosummary

   Grid.ndim
   Grid.size
   Grid.dims
   Grid.shape
   Grid.delta
   Grid.zero
   Grid.is_separated
   Grid.is_regular

Actual coordinates can be accessed by:

.. autosummary::
   :toctree: _autosummary

   Grid.points
   Grid.separated_coords
   Grid.regular_coords

Grids can be subject to simple transformations:

.. autosummary::
   :toctree: _autosummary

   Grid.scale
   Grid.scaled
   Grid.shift
   Grid.shifted
   Grid.reverse
   Grid.reversed

Other function exist to simplify working with Grids:

.. autosummary::
   :toctree: _autosummary

   Grid.copy
   Grid.subset
   Grid.closest_to

Currently only a Cartesian and Polar coordinate system are implemented:

.. autosummary::
   :toctree: _autosummary

   CartesianGrid
   PolarGrid

These provide attributes specific to their coordinate system:

.. autosummary::
   :toctree: _autosummary

   CartesianGrid.x
   CartesianGrid.y
   CartesianGrid.z
   CartesianGrid.w

   PolarGrid.r
   PolarGrid.theta



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
