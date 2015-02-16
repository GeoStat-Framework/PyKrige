__author__ = 'Benjamin S. Murphy'
__version__ = '1.0.3'
__doc__ = """Code by Benjamin S. Murphy
bscott.murphy@gmail.com

Dependencies:
    numpy
    scipy
    matplotlib

Modules:
    ok: Contains class OrdinaryKriging, which is a convenience class
        for easy access to 2D Ordinary Kriging.
    uk: Contains class UniversalKriging, which  provides more control over
        2D kriging by utilizing drift terms. Supported drift terms
        currently include point-logarithmic, regional linear, and external
        z-scalar.
    kriging_tools: Contains a set of functions to work with *.asc files.
    variogram_models: Contains the definitions for the implemented variogram
        models. Note that the utilized formulas are as presented in Kitanidis,
        so the exact definition of the range (specifically, the associated
        scaling of that value) may differ slightly from other sources.
    core: Contains the backbone functions of the package that are called by
        both the OrdinaryKriging class and the UniversalKriging class.
        The functions were consolidated here in order to reduce redundancy
        in the code.
    test: Contains the test script.

References:
    P.K. Kitanidis, Introduction to Geostatistics: Applications in Hydrogeology,
    (Cambridge University Press, 1997) 272 p.
    
Copyright (c) 2015 Benjamin S. Murphy
"""