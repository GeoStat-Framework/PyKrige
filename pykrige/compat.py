# coding: utf-8
# pylint: disable= invalid-name,  unused-import
"""For compatibility"""

from __future__ import absolute_import
import sys
from packaging import version


PY3 = (sys.version_info[0] == 3)


# sklearn
try:
    from sklearn import __version__ as skl_ver
    try:  # scikit-learn 1.18.+
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import train_test_split
    except ImportError:  # older scikit-learn versions
        from sklearn.grid_search import GridSearchCV
        from sklearn.cross_validation import train_test_split

    SKLEARN_INSTALLED = True
    # state if train_score is returned (false from v0.21 on)
    TRAIN_SCORE_ON = version.parse(skl_ver) < version.parse("0.21")

except ImportError:
    SKLEARN_INSTALLED = False
    TRAIN_SCORE_ON = False


class SklearnException(Exception):
    pass


def validate_sklearn():
    if not SKLEARN_INSTALLED:
        raise SklearnException('sklearn needs to be installed in order '
                               'to use this module')
