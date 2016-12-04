# Optimise Kriging Parameters

The optimise module uses scikit-learn's `GridSearchCv` and scans across the parameters. One could simply replace `GridSearchCV` with `RandomSearchCV`. 


## How to use the optimise module

An example to use the optimise module is provided in a simple config file object inside `pykrige/optimise/pipeline.py`

To run the example optimiser with random data, use the following command from
    
    python part/to/pykrige/optimise/pipeline.py

This script will produce the optimisation result in the file `optimisation.csv`.
 
A typical optimisation output will look like the following:

|mean_test_score|mean_train_score|rank_test_score|param_krige__variogram_model|param_krige__method|
|---------------|----------------|---------------|----------------------------|-------------------|
|-0.17|1.0|1|linear|ordinary|
|-0.17|1.0|3|power|ordinary|
|-1.41|1.0|5|gaussian|ordinary|
|-0.17|1.0|1|linear|universal|
|-0.17|1.0|3|power|universal|
|-1.41|1.0|5|gaussian|universal|
