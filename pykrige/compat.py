# coding: utf-8
# pylint: disable= invalid-name,  unused-import
"""For compatibility"""

from __future__ import absolute_import
import sys


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

except ImportError:
    SKLEARN_INSTALLED = False


class SklearnException(Exception):
    pass


def validate_sklearn():
    if not SKLEARN_INSTALLED:
        raise SklearnException('sklearn needs to be installed in order '
                               'to use this module')
