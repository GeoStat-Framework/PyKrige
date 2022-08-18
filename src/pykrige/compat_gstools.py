# coding: utf-8
# pylint: disable= invalid-name,  unused-import
"""For GSTools compatibility."""

# gstools
try:
    import gstools as gs

    GSTOOLS_INSTALLED = True
    GSTOOLS_VERSION = list(map(int, gs.__version__.split(".")[:2]))
except ImportError:
    gs = None
    GSTOOLS_INSTALLED = False
    GSTOOLS_VERSION = None


class GSToolsException(Exception):
    """Exception for GSTools."""


def validate_gstools(model):
    """Validate presence of GSTools."""
    if not GSTOOLS_INSTALLED:
        raise GSToolsException(
            "GSTools needs to be installed in order to use their CovModel class."
        )
    if not isinstance(model, gs.CovModel):
        raise GSToolsException(
            "GSTools: given variogram model is not a CovModel instance."
        )
    if GSTOOLS_VERSION < [1, 3]:
        raise GSToolsException("GSTools: need at least GSTools v1.3.")
    if model.latlon and GSTOOLS_VERSION < [1, 4]:
        raise GSToolsException(
            "GSTools: latlon models in PyKrige are only supported from GSTools v1.4."
        )
