PyKrige Changelog
=================

**Version 0.1.2**
October 27, 2014

* First complete release.

**Version 0.2.0**
November 23, 2014

* Consolidated backbone functions into a single module in order to reduce redundancy in the code. OrdinaryKriging and UniversalKriging classes now import and call the core module for the standard functions.
* Fixed a few glaring mistakes in the code.
* Added more documentation.

**Version 1.0**
January 25, 2015

* Changed license to New BSD.
* Added support for point-specific and masked-grid kriging. Note that the arguments for the OrdinaryKriging.execute() and UniversalKriging.execute() methods have changed.
* Changed semivariogram binning procedure.
* Boosted execution speed by almost an order of magnitude.
* Fixed some problems with the external drift capabilities.
* Added more comprehensive testing script.
* Fixed slight problem with read_asc_grid() function in kriging_tools. Also made some code improvements to both the write_asc_grid() and read_asc_grid() functions in kriging_tools.

**Version 1.0.3**
February 15, 2015

* Fixed a problem with the tests that are performed to see if the kriging system is to be solved at a data point. (Tests are completed in order to determine whether to force the kriging solution to converge to the true data value.)
* Changed setup script.