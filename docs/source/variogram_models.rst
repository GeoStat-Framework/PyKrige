Variogram Models
================

PyKrige internally supports the six variogram models listed below.
Additionally, the code supports user-defined variogram models via the 'custom'
variogram model keyword argument.

* Gaussian Model

.. math::
    p \cdot (1 - e^{ - \frac{d^2}{(\frac{4}{7} r)^2}}) + n

* Exponential Model

.. math::
    p \cdot (1 - e^{ - \frac{d}{r/3}}) + n

* Spherical Model

.. math::
    \begin{cases}
        p \cdot (\frac{3d}{2r} - \frac{d^3}{2r^3}) + n & d \leq r \\
        p + n & d > r
    \end{cases}

* Linear Model

.. math::

    s \cdot d + n

Where `s` is the slope and `n` is the nugget.

* Power Model

.. math::

    s \cdot d^e + n

Where `s` is the scaling factor, `e` is the exponent (between 0 and 2), and `n`
is the nugget term.

* Hole-Effect Model

.. math::
    p \cdot (1 - (1 - \frac{d}{r / 3}) * e^{ - \frac{d}{r / 3}}) + n

Variables are defined as:

:math:`d` = distance values at which to calculate the variogram

:math:`p` = partial sill (psill = sill - nugget)

:math:`r` = range

:math:`n` = nugget

:math:`s` = scaling factor or slope

:math:`e` = exponent for power model

For stationary variogram models (gaussian, exponential, spherical, and
hole-effect models), the partial sill is defined as the difference between
the full sill and the nugget term. The sill represents the asymptotic
maximum spatial variance at longest lags (distances). The range represents
the distance at which the spatial variance has reached ~95% of the
sill variance. The nugget effectively takes up 'noise' in measurements.
It represents the random deviations from an overall smooth spatial data trend.
(The name *nugget* is an allusion to kriging's mathematical origin in
gold exploration; the nugget effect is intended to take into account the
possibility that when sampling you randomly hit a pocket gold that is
anomalously richer than the surrounding area.)

For nonstationary models (linear and power models, with unbounded spatial
variances), the nugget has the same meaning. The exponent  for the power-law
model should be between 0 and 2 [1]_.

**A few important notes:**

The PyKrige user interface by default takes the full sill. This default behavior
can be changed with a keyword flag, so that the user can supply the partial sill
instead. The code internally uses the partial sill (psill = sill - nugget)
rather than the full sill, as it's safer to perform automatic variogram
estimation using the partial sill.

The exact definitions of the variogram models here may differ from those used
elsewhere. Keep that in mind when switching from another kriging code over to
PyKrige.

According to [1]_, the hole-effect variogram model is only correct for the
1D case. It's implemented here for completeness and should be used cautiously.

References
----------
.. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
    Hydrogeology, (Cambridge University Press, 1997) 272 p.
