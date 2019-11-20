# coding: utf-8
# pylint: disable= invalid-name,  unused-import
"""For compatibility"""

from __future__ import absolute_import
import sys
import inspect
from functools import partial


PY3 = (sys.version_info[0] == 3)


# sklearn
try:
    try:  # scikit-learn 1.18.+
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import train_test_split
    except ImportError:  # older scikit-learn versions
        from sklearn.grid_search import GridSearchCV
        from sklearn.cross_validation import train_test_split

    SKLEARN_INSTALLED = True
    if SKLEARN_INSTALLED:
        if PY3:
            arg_spec = inspect.getfullargspec(GridSearchCV)[0]
            # https://stackoverflow.com/a/56618067/6696397
            if "return_train_score" in arg_spec:
                GridSearchCV = partial(GridSearchCV, return_train_score=True)

except ImportError:
    SKLEARN_INSTALLED = False


class SklearnException(Exception):
    pass


def validate_sklearn():
    if not SKLEARN_INSTALLED:
        raise SklearnException('sklearn needs to be installed in order '
                               'to use this module')
