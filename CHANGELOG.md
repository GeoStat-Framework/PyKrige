Changelog
=========


Version 1.5.1
-------------
*August 20, 2020*

**New features**

* update Regression Kriging class to be compatible with all kriging features (#158)
* added option to enable/disable "exact values" to all kriging routines (#153)
* added option to use the pseudo-inverse in all kriging routines (#151)

**Changes**

* removed compat-layer for sklearn (#157)
* updated examples in documentation


Version 1.5.0
-------------
*April 04, 2020*

**New features**

* support for GSTools covariance models (#125)
* pre-build wheels for py35-py38 on Linux, Windows and MacOS (#142)
* GridSerachCV from the compat module sets iid=False by default (if present in sklearn)
  to be future prove (iid will be deprecated) (#144)

**Changes**

* dropped py2* and py<3.5 support (#142)
* installation now requires cython (#142)
* codebase was formatted with black (#144)
* internally use of scipys lapack/blas bindings (#142)
* PyKrige is now part of the GeoStat-Framework


Version 1.4.1
-------------
*January 13, 2019*

**New features**

* Added method to obtain variogram model points. PR[#94](https://github.com/GeoStat-Framework/PyKrige/pull/94) by [Daniel Mejía Raigosa](https://github.com/Daniel-M)

**Bug fixes**

* Fixed OrdinaryKriging readme example. PR[#107](https://github.com/GeoStat-Framework/PyKrige/pull/107) by [Harry Matchette-Downes](https://github.com/harrymd)
* Fixed kriging matrix not being calculated correctly for geographic coordinates. PR[99](https://github.com/GeoStat-Framework/PyKrige/pull/99) by [Mike Rilee](https://github.com/michaelleerilee)


Version 1.4.0
-------------
*April 24, 2018*

**New features**

* Regression kriging algotithm. PR [#27](https://github.com/GeoStat-Framework/PyKrige/pull/27) by [Sudipta Basaks](https://github.com/basaks).
* Support for spherical coordinates. PR [#23](https://github.com/GeoStat-Framework/PyKrige/pull/23) by [Malte Ziebarth](https://github.com/mjziebarth)
* Kriging parameter tuning with scikit-learn. PR [#24](https://github.com/GeoStat-Framework/PyKrige/pull/24) by [Sudipta Basaks](https://github.com/basaks).
* Variogram model parameters can be specified using a list or a dict. Allows for directly feeding in the partial sill rather than the full sill. PR [#47](https://github.com/GeoStat-Framework/PyKrige/pull/47) by [Benjamin Murphy](https://github.com/bsmurphy).

**Enhancements**

* Improved memory usage in variogram calculations. PR [#42](https://github.com/GeoStat-Framework/PyKrige/pull/42) by [Sudipta Basaks](https://github.com/basaks).
* Added benchmark scripts. PR [#36](https://github.com/GeoStat-Framework/PyKrige/pull/36) by [Roman Yurchak](https://github.com/rth)
* Added an extensive example using the meusegrids dataset. PR [#28](https://github.com/GeoStat-Framework/PyKrige/pull/28) by [kvanlombeek](https://github.com/kvanlombeek).

**Bug fixes**

* Statistics calculations in 3D kriging. PR [#45](https://github.com/GeoStat-Framework/PyKrige/pull/45) by [Will Chang](https://github.com/whdc).
* Automatic variogram estimation robustified. PR [#47](https://github.com/GeoStat-Framework/PyKrige/pull/47) by [Benjamin Murphy](https://github.com/bsmurphy).


Version 1.3.1
-------------
*December 10, 2016*

* More robust setup for building Cython extensions


Version 1.3.0
-------------
*October 23, 2015*

* Added support for Python 3.
* Updated the setup script to handle problems with trying to build the Cython extensions. If the appropriate compiler hasn't been installed on Windows, then the extensions won't work (see [this discussion of using Cython extensions on Windows] for how to deal with this problem). The setup script now attempts to build the Cython extensions and automatically falls back to pure Python if the build fails. **NOTE that the Cython extensions currently are not set up to work in Python 3** (see [discussion in issue #10]), so they are not built when installing with Python 3. This will be changed in the future.

* [closed issue #2]: https://github.com/GeoStat-Framework/PyKrige/issues/2
* [this discussion of using Cython extensions on Windows]: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
* [discussion in issue #10]: https://github.com/GeoStat-Framework/PyKrige/issues/10


Version 1.2.0
-------------
*August 1, 2015*

* Updated the execution portion of each class to streamline processing and reduce redundancy in the code.
* Integrated kriging with a moving window for two-dimensional ordinary kriging. Thanks to Roman Yurchak for this addition. This can be very useful for working with very large datasets, as it limits the size of the kriging matrix system. However, note that this approach can also produce unexpected oddities if the spatial covariance of the data does not decay quickly or if the window is too small. (See Kitanidis 1997 for a discussion of potential problems in kriging with a moving window; also see [closed issue #2] for a brief note about important considerations when kriging with a moving window.)
* Integrated a Cython backend for two-dimensional ordinary kriging. Again, thanks to Roman Yurchak for this addition. Note that currently the Cython backend is only implemented for two-dimensional ordinary kriging; it is not implemented in any of the other kriging classes. (I'll gladly accept any pull requests to extend the Cython backend to the other classes.)
* Implemented two new generic drift capabilities that should allow for use of arbitrary user-designed drifts. These generic drifts are referred to as 'specified' and 'functional' in the code. They are available for both two-dimensional and three-dimensional universal kriging (see below). With the 'specified' drift capability, the user specifies the values of the drift term at every data point and every point at which the kriging system is to be evaluated. With the 'functional' drift capability, the user provides callable function(s) of the two or three spatial coordinates that define the drift term(s). The functions must only take the spatial coordinates as arguments. An arbitrary number of 'specified' or 'functional' drift terms may be used. See `UniversalKriging.__doc__` or `UniversalKriging3D.__doc__` for more information.
* Made a few changes to how the drift terms are implemented when the problem is anisotropic. The regional linear drift is applied in the adjusted coordinate frame. For the point logarithmic drift, the point coordinates are transformed into the adjusted coordinate frame and the drift values are calculated in the transformed frame. The external scalar drift values are extracted using the original (i.e., unadjusted) coordinates. Any functions that are used with the 'functional' drift capability are evaluated in the adjusted coordinate frame. Specified drift values are not adjusted as they are taken to be for the exact points provided.
* Added support for three-dimensional universal kriging. The previous three-dimensional kriging class has been renamed OrdinaryKriging3D within module ok3d, and the new class is called UniversalKriging3D within module uk3d. See `UniversalKriging3D.__doc__` for usage information. A regional linear drift ('regional_linear') is the only code-internal drift that is currently supported, but the 'specified' and 'functional' generic drift capabilities are also implemented here (see above). The regional linear drift is applied in all three spatial dimensions.


Version 1.1.0
-------------
*May 25, 2015*

* Added support for two different approaches to solving the entire kriging problem. One approach solves for the specified grid or set of points in a single vectorized operation; this method is default. The other approach loops through the specified points and solves the kriging system at each point. In both of these techniques, the kriging matrix is set up and inverted only once. In the vectorized approach, the rest of the kriging system (i.e., the RHS matrix) is set up as a single large array, and the whole system is solved with a single call to `numpy.dot()`. This approach is faster, but it can consume a lot of RAM for large datasets and/or large grids. In the looping approach, the rest of the kriging system (the RHS matrix) is set up at each point, and the kriging system at that point is solved with a call to `numpy.dot()`. This approach is slower, but it does not take as much memory. The approach can be specified by using the `backend` kwarg in the `execute()` method: `'vectorized'` (default) for the vectorized approach, `'loop'` for the looping approach. Thanks to Roman Yurchak for these changes and optimizations.
* Added support for implementing custom variogram models. To do so, set `variogram_model` to `'custom'`. You must then also specify `variogram_parameters` as well as `variogram_function`, which must be a callable object that takes only two arguments, first a list of function parameters and then the distances at which to evaluate the variogram model. Note that currently the code will not automatically fit the custom variogram model to the data. You must provide the `variogram_parameters`, which will be passed to the callable `variogram_function` as the first argument.
* Modified anisotropy rotation so that coordinate system is rotated CCW by specified angle. The sense of rotation for 2D kriging is now the opposite of what it was before.
* Added support for 3D kriging. This is now available as class `Krige3D` in `pykrige.k3d`. The usage is essentially the same as with the two-dimensional kriging classes, except for a few extra arguments that must be passed during instantiation and when calling `Krige3D.execute()`. See `Krige3D.__doc__` for more information.


Version 1.0.3
-------------
*February 15, 2015*

* Fixed a problem with the tests that are performed to see if the kriging system is to be solved at a data point. (Tests are completed in order to determine whether to force the kriging solution to converge to the true data value.)
* Changed setup script.


Version 1.0
-----------
*January 25, 2015*

* Changed license to New BSD.
* Added support for point-specific and masked-grid kriging. Note that the arguments for the `OrdinaryKriging.execute()` and `UniversalKriging.execute()` methods have changed.
* Changed semivariogram binning procedure.
* Boosted execution speed by almost an order of magnitude.
* Fixed some problems with the external drift capabilities.
* Added more comprehensive testing script.
* Fixed slight problem with `read_asc_grid()` function in `kriging_tools`. Also made some code improvements to both the `write_asc_grid()` and `read_asc_grid()` functions in `kriging_tools`.


Version 0.2.0
-------------
*November 23, 2014*

* Consolidated backbone functions into a single module in order to reduce redundancy in the code. `OrdinaryKriging` and `UniversalKriging` classes now import and call the `core` module for the standard functions.
* Fixed a few glaring mistakes in the code.
* Added more documentation.


Version 0.1.2
-------------
*October 27, 2014*

* First complete release.
