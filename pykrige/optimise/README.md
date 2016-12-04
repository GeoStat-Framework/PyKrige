# Optimised Kriging

The optimise module uses scikit-learn's `GridSearchCv` and scans across the parameters. One could simply replace `GridSearchCV` with `RandomSearchCV`. 


## How to use the optimise module

An example to use the optimise module is provided in a simple config file object inside `pykrige/optimise/pipeline.py`

To run the example optimiser with random data, use the following command from
    
    python part/to/pykrige/optimise/pipeline.py

This script will produce the optimisation result in the file `optimisation.csv`.
 