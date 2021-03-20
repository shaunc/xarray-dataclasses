xarray-dataclasses
==================

|PyPI| |Python| |Test| |License|

xarray extension for DataArray and Dataset classes

TL;DR
-----

xarray-dataclasses is a Python package for creating DataArray and Dataset classes in the same manner as `the Python’s native dataclass <https://docs.python.org/3/library/dataclasses.html>`__. Here is an example code of what the package provides:

.. code:: python

   from xarray_dataclasses import Coord, Data, dataarrayclass


   @dataarrayclass
   class Image:
       """DataArray that represents an image."""

       data: Data[tuple['x', 'y'], float]
       x: Coord['x', int] = 0
       y: Coord['y', int] = 0


   # create a DataArray instance
   image = Image.new([[0, 1], [2, 3]], x=[0, 1], y=[0, 1])


   # create a DataArray instance filled with ones
   ones = Image.ones((2, 2), x=[0, 1], y=[0, 1])

Features
~~~~~~~~

-  DataArray or Dataset instances with fixed dimensions, data type, and coordinates can easily be created.
-  NumPy-like special functions such as ``ones()`` are provided as class methods.
-  100% compatible with `the Python’s native dataclass <https://docs.python.org/3/library/dataclasses.html>`__.
-  100% compatible with static type check by `Pyright <https://github.com/microsoft/pyright>`__.

Installation
~~~~~~~~~~~~

.. code:: shell

   $ pip install xarray-dataclasses

Introduction
------------

`xarray <https://xarray.pydata.org/en/stable/index.html>`__ is useful for handling labeled multi-dimensional data, but it is a bit troublesome to create a DataArray or Dataset instance with fixed dimensions, data type, or coordinates. For example, let us think about the following specifications of DataArray instances:

-  Dimensions of data must be ``('x', 'y')``.
-  Data type of data must be ``float``.
-  Data type of dimensions must be ``int``.
-  Default value of dimensions must be ``0``.

Then a function to create a spec-compliant DataArray instance is something like this:

.. code:: python

   import numpy as np
   import xarray as xr


   def spec_dataarray(data, x=None, y=None):
       """Create a spec-comliant DataArray instance."""
       data = np.array(data)

       if x is None:
           x = np.zeros(data.shape[0])
       else:
           x = np.array(x)

       if y is None:
           y = np.zeros(data.shape[1])
       else:
           y = np.array(y)

       return xr.DataArray(
           data=data.astype(float),
           dims=('x', 'y'),
           coords={
               'x': ('x', x.astype(int)),
               'y': ('y', y.astype(int)),
           },
       )


   dataarray = spec_dataarray([[0, 1], [2, 3]])

The issues are (1) it is hard to figure out the specs from the code and (2) it is hard to reuse the code, for example, to add a new coordinate to the original specs.

`xarray-dataclasses <#xarray-dataclasses>`__ resolves them by defining the specs as a dataclass with dedicated type hints:

.. code:: python

   from xarray_dataclasses import Coord, Data, dataarrayclass


   @dataarrayclass
   class Specs:
       data: Data[tuple['x', 'y'], float]
       x: Coord['x', int] = 0
       y: Coord['y', int] = 0


   dataarray = Specs.new([[0, 1], [2, 3]])

The specs are now much easier to read: The type hints, ``Data[<dims>, <dtype>]`` and ``Coord[<dims>, <dtype>]``, have complete information of DataArray creation. The default values are given as class variables.

The class decorator, ``@dataarrayclass``, converts a class to `the Python’s native dataclass <https://docs.python.org/3/library/dataclasses.html>`__ and add class methods such as ``new()`` to it. The extension of the specs is then easy by class inheritance.

Basic usage
-----------

xarray-dataclasses uses `the Python’s native dataclass <https://docs.python.org/3/library/dataclasses.html>`__ (please learn how to use it before proceeding). Data (or data variables), coordinates, attribute members, and name of a DataArray or Dataset instance are defined as dataclass fields with the following dedicated type hints.

Data
~~~~

``Data[<dims>, <dtype>]`` specifies the field whose value will become the data of a DataArray instance or a member of the data variables of a Dataset instance. It accepts two type variables, ``<dims>`` and ``<dtype>``, for fixing dimensions and data type, respectively. For example:

================================ ============== ==========================
Type hint                        Inferred dims  Inferred dtype
================================ ============== ==========================
``Data['x', typing.Any]``        ``('x',)``     ``None``
``Data['x', int]``               ``('x',)``     ``numpy.dtype('int64')``
``Data['x', float]``             ``('x',)``     ``numpy.dtype('float64')``
``Data[tuple['x', 'y'], float]`` ``('x', 'y')`` ``numpy.dtype('float64')``
================================ ============== ==========================

Note: for Python 3.7 and 3.8, use ``typing.Tuple[...]`` instead of ``tuple[...]``.

Coord
~~~~~

``Coord[<dims>, <dtypes>]`` specifies the field whose value will become a coordinate of a DataArray or Dataset instance. Similar to ``Data``, it accepts two type variables, ``<dims>`` and ``<dtype>``, for fixing dimensions and data type, respectively.

Attr
~~~~

``Attr[<type>]`` specifies the field whose value will become a member of the attributes (attrs) of a DataArray or Dataset instance. It accepts a type variable, ``<type>``, for specifying the type of the value. For example:

.. code:: python

   @dataarrayclass
   class Specs:
       units: Attr[str] = 'm/s'  # equivalent to str

Name
~~~~

``Name[<type>]`` specifies the field whose value will become the name of a DataArray. It accepts a type variable, ``<type>``, for specifying the type of the value. For example:

.. code:: python

   @dataarrayclass
   class Specs:
       name: Name[str] = 'default'  # equivalent to str

DataArray class
~~~~~~~~~~~~~~~

DataArray class is a dataclass that defines DataArray creation. For example:

.. code:: python

   from xarray_dataclasses import Attr, Coord, Data, Name, dataarrayclass


   @dataarrayclass
   class Image:
       """DataArray that represents an image."""

       data: Data[tuple['x', 'y'], float]
       x: Coord['x', int] = 0
       y: Coord['y', int] = 0
       dpi: Attr[int] = 300
       name: Name[str] = 'default'

where exactly one ``Data``-typed field is allowed. ``ValueError`` is raised if more than two ``Data``-type fields exist. A spec-compliant DataArray instance is created by a shorthand method, ``new()``:

.. code:: python

   Image.new([[0, 1], [2, 3]], x=[0, 1], y=[0, 1])

   <xarray.DataArray 'default' (x: 2, y: 2)>
   array([[0., 1.],
          [2., 3.]])
   Coordinates:
     * x        (x) int64 0 1
     * y        (y) int64 0 1
   Attributes:
       dpi:      300

DataArray class has NumPy-like ``empty()``, ``zeros()``, ``ones()``, ``full()`` methods:

.. code:: python

   Image.ones((3, 3), dpi=200, name='flat')

   <xarray.DataArray 'flat' (x: 3, y: 3)>
   array([[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]])
   Coordinates:
     * x        (x) int64 0 0 0
     * y        (y) int64 0 0 0
   Attributes:
       dpi:      200

Dataset class
~~~~~~~~~~~~~

Dataset class is a dataclass that defines Dataset creation. For example:

.. code:: python

   from xarray_dataclasses import Attr, Coord, Data, datasetclass


   @datasetclass
   class RGBImage:
       """Dataset that represents a three-color image."""

       red: Data[tuple['x', 'y'], float]
       green: Data[tuple['x', 'y'], float]
       blue: Data[tuple['x', 'y'], float]
       x: Coord['x', int] = 0
       y: Coord['y', int] = 0
       dpi: Attr[int] = 300

where multiple ``Data``-typed fields are allowed. A spec-compliant Dataset instance is created by a shorthand method, ``new()``:

.. code:: python

   RGBImage.new(
       [[0, 0], [0, 0]],  # red
       [[1, 1], [1, 1]],  # green
       [[2, 2], [2, 2]],  # blue
   )

   <xarray.Dataset>
   Dimensions:  (x: 2, y: 2)
   Coordinates:
     * x        (x) int64 0 0
     * y        (y) int64 0 0
   Data variables:
       red      (x, y) float64 0.0 0.0 0.0 0.0
       green    (x, y) float64 1.0 1.0 1.0 1.0
       blue     (x, y) float64 2.0 2.0 2.0 2.0
   Attributes:
       dpi:      300

.. raw:: html

   <!-- References -->

.. |PyPI| image:: https://img.shields.io/pypi/v/xarray-dataclasses.svg?label=PyPI&style=flat-square
   :target: https://pypi.org/project/xarray-dataclasses/
.. |Python| image:: https://img.shields.io/pypi/pyversions/xarray-dataclasses.svg?label=Python&color=yellow&style=flat-square
   :target: https://pypi.org/project/xarray-dataclasses/
.. |Test| image:: https://img.shields.io/github/workflow/status/astropenguin/xarray-dataclasses/Test?logo=github&label=Test&style=flat-square
   :target: https://github.com/astropenguin/xarray-dataclasses/actions
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg?label=License&style=flat-square
   :target: LICENSE
