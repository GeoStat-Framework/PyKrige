# coding: utf-8
# pylint: disable= invalid-name,  unused-import
"""For compatibility"""
import inspect
from functools import partial


# sklearn
try:
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split

    SKLEARN_INSTALLED = True
    arg_spec = inspect.getfullargspec(GridSearchCV)[0]
    # https://stackoverflow.com/a/56618067/6696397
    if "return_train_score" in arg_spec:
        GridSearchCV = partial(GridSearchCV, return_train_score=True)
    if "iid" in arg_spec:
        GridSearchCV = partial(GridSearchCV, iid=False)

except ImportError:
    SKLEARN_INSTALLED = False


class SklearnException(Exception):
    pass


def validate_sklearn():
    if not SKLEARN_INSTALLED:
        raise SklearnException(
            "sklearn needs to be installed in order to use this module"
        )
