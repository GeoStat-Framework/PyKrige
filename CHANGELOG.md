PyKrige Changelog
=================

**Version 0.1.2**
October 27, 2014

* First complete release.

**Version 0.2.0**
November 23, 2014

* Consolidated backbone functions into a single module in order to reduce redundancy in the code. `OrdinaryKriging` and `UniversalKriging` classes now import and call the `core` module for the standard functions.
* Fixed a few glaring mistakes in the code.
* Added more documentation.

**Version 1.0**
January 25, 2015

* Changed license to New BSD.
* Added support for point-specific and masked-grid kriging. Note that the arguments for the `OrdinaryKriging.execute()` and `UniversalKriging.execute()` methods have changed.
* Changed semivariogram binning procedure.
* Boosted execution speed by almost an order of magnitude.
* Fixed some problems with the external drift capabilities.
* Added more comprehensive testing script.
* Fixed slight problem with `read_asc_grid()` function in `kriging_tools`. Also made some code improvements to both the `write_asc_grid()` and `read_asc_grid()` functions in `kriging_tools`.

**Version 1.0.3**
February 15, 2015

* Fixed a problem with the tests that are performed to see if the kriging system is to be solved at a data point. (Tests are completed in order to determine whether to force the kriging solution to converge to the true data value.)
* Changed setup script.

**Version 1.1.0**
May 25, 2015

* Added support for two different approaches to solving the entire kriging problem. One approach solves for the specified grid or set of points in a single vectorized operation; this method is default. The other approach loops through the specified points and solves the kriging system at each point. In both of these techniques, the kriging matrix is set up and inverted only once. In the vectorized approach, the rest of the kriging system (i.e., the RHS matrix) is set up as a single large array, and the whole system is solved with a single call to `numpy.dot()`. This approach is faster, but it can consume a lot of RAM for large datasets and/or large grids. In the looping approach, the rest of the kriging system (the RHS matrix) is set up at each point, and the kriging system at that point is solved with a call to `numpy.dot()`. This approach is slower, but it does not take as much memory. The approach can be specified by using the `backend` kwarg in the `execute()` method: `'vectorized'` (default) for the vectorized approach, `'loop'` for the looping approach. Thanks to Roman Yurchak for these changes and optimizations.
* Added support for implementing custom variogram models. To do so, set `variogram_model` to `'custom'`. You must then also specify `variogram_parameters` as well as `variogram_function`, which must be a callable object that takes only two arguments, first a list of function parameters and then the distances at which to evaluate the variogram model. Note that currently the code will not automatically fit the custom variogram model to the data. You must provide the `variogram_parameters`, which will be passed to the callable `variogram_function` as the first argument.
* Modified anisotropy rotation so that coordinate system is rotated CCW by specified angle. The sense of rotation for 2D kriging is now the opposite of what it was before.
* Added support for 3D kriging. This is now available as class `Krige3D` in `pykrige.k3d`. The usage is essentially the same as with the two-dimensional kriging classes, except for a few extra arguments that must be passed during instantiation and when calling `Krige3D.execute()`. See `Krige3D.__doc__` for more information.