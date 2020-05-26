# coding: utf-8
# pylint: disable= invalid-name,  unused-import
"""For compatibility"""
from functools import partial


# sklearn
try:
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split

    SKLEARN_INSTALLED = True

except ImportError:
    SKLEARN_INSTALLED = False


class SklearnException(Exception):
    pass


def validate_sklearn():
    if not SKLEARN_INSTALLED:
        raise SklearnException(
            "sklearn needs to be installed in order to use this module"
        )
