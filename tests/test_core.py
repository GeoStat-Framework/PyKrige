"""
Testing code.
Updated BSM February 2017
"""
import sys
import os

import numpy as np
import pytest
from pytest import approx
from numpy.testing import assert_allclose
from scipy.spatial.distance import cdist

from pykrige import kriging_tools as kt
from pykrige import core
from pykrige import variogram_models
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

allclose_pars = {"rtol": 1e-05, "atol": 1e-08}


@pytest.fixture
def validation_ref():

    data = np.genfromtxt(os.path.join(BASE_DIR, "test_data/test_data.txt"))
    ok_test_answer, ok_test_gridx, ok_test_gridy, cellsize, no_data = kt.read_asc_grid(
        os.path.join(BASE_DIR, "test_data/test1_answer.asc"), footer=2
    )
    uk_test_answer, uk_test_gridx, uk_test_gridy, cellsize, no_data = kt.read_asc_grid(
        os.path.join(BASE_DIR, "test_data/test2_answer.asc"), footer=2
    )

    return (
        data,
        (ok_test_answer, ok_test_gridx, ok_test_gridy),
        (uk_test_answer, uk_test_gridx, uk_test_gridy),
    )


@pytest.fixture
def sample_data_2d():

    data = np.array(
        [
            [0.3, 1.2, 0.47],
            [1.9, 0.6, 0.56],
            [1.1, 3.2, 0.74],
            [3.3, 4.4, 1.47],
            [4.7, 3.8, 1.74],
        ]
    )
    gridx = np.arange(0.0, 6.0, 1.0)
    gridx_2 = np.arange(0.0, 5.5, 0.5)
    gridy = np.arange(0.0, 5.5, 0.5)
    xi, yi = np.meshgrid(gridx, gridy)
    mask = np.array(xi == yi)
    return data, (gridx, gridy, gridx_2), mask


@pytest.fixture
def sample_data_3d():
    data = np.array(
        [
            [0.1, 0.1, 0.3, 0.9],
            [0.2, 0.1, 0.4, 0.8],
            [0.1, 0.3, 0.1, 0.9],
            [0.5, 0.4, 0.4, 0.5],
            [0.3, 0.3, 0.2, 0.7],
        ]
    )
    gridx = np.arange(0.0, 0.6, 0.05)
    gridy = np.arange(0.0, 0.6, 0.01)
    gridz = np.arange(0.0, 0.6, 0.1)
    zi, yi, xi = np.meshgrid(gridz, gridy, gridx, indexing="ij")
    mask = np.array((xi == yi) & (yi == zi))
    return data, (gridx, gridy, gridz), mask


def test_core_adjust_for_anisotropy():

    X = np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]).T
    X_adj = core._adjust_for_anisotropy(X, [0.0, 0.0], [2.0], [90.0])
    assert_allclose(X_adj[:, 0], np.array([0.0, 1.0, 0.0, -1.0]), **allclose_pars)
    assert_allclose(X_adj[:, 1], np.array([-2.0, 0.0, 2.0, 0.0]), **allclose_pars)


def test_core_adjust_for_anisotropy_3d():

    # this is a bad examples, as the X matrix is symmetric
    # and insensitive to transpositions
    X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).T
    X_adj = core._adjust_for_anisotropy(
        X, [0.0, 0.0, 0.0], [2.0, 2.0], [90.0, 0.0, 0.0]
    )
    assert_allclose(X_adj[:, 0], np.array([1.0, 0.0, 0.0]), **allclose_pars)
    assert_allclose(X_adj[:, 1], np.array([0.0, 0.0, 2.0]), **allclose_pars)
    assert_allclose(X_adj[:, 2], np.array([0.0, -2.0, 0.0]), **allclose_pars)
    X_adj = core._adjust_for_anisotropy(
        X, [0.0, 0.0, 0.0], [2.0, 2.0], [0.0, 90.0, 0.0]
    )
    assert_allclose(X_adj[:, 0], np.array([0.0, 0.0, -1.0]), **allclose_pars)
    assert_allclose(X_adj[:, 1], np.array([0.0, 2.0, 0.0]), **allclose_pars)
    assert_allclose(X_adj[:, 2], np.array([2.0, 0.0, 0.0]), **allclose_pars)
    X_adj = core._adjust_for_anisotropy(
        X, [0.0, 0.0, 0.0], [2.0, 2.0], [0.0, 0.0, 90.0]
    )
    assert_allclose(X_adj[:, 0], np.array([0.0, 1.0, 0.0]), **allclose_pars)
    assert_allclose(X_adj[:, 1], np.array([-2.0, 0.0, 0.0]), **allclose_pars)
    assert_allclose(X_adj[:, 2], np.array([0.0, 0.0, 2.0]), **allclose_pars)


def test_core_make_variogram_parameter_list():

    # test of first case - variogram_model_parameters is None
    # function should return None unaffected
    result = core._make_variogram_parameter_list("linear", None)
    assert result is None

    # tests for second case - variogram_model_parameters is dict
    with pytest.raises(KeyError):
        core._make_variogram_parameter_list("linear", {"tacos": 1.0, "burritos": 2.0})
    result = core._make_variogram_parameter_list(
        "linear", {"slope": 1.0, "nugget": 0.0}
    )
    assert result == [1.0, 0.0]

    with pytest.raises(KeyError):
        core._make_variogram_parameter_list("power", {"frijoles": 1.0})
    result = core._make_variogram_parameter_list(
        "power", {"scale": 2.0, "exponent": 1.0, "nugget": 0.0}
    )
    assert result == [2.0, 1.0, 0.0]

    with pytest.raises(KeyError):
        core._make_variogram_parameter_list("exponential", {"tacos": 1.0})
    with pytest.raises(KeyError):
        core._make_variogram_parameter_list(
            "exponential", {"range": 1.0, "nugget": 1.0}
        )
    result = core._make_variogram_parameter_list(
        "exponential", {"sill": 5.0, "range": 2.0, "nugget": 1.0}
    )
    assert result == [4.0, 2.0, 1.0]
    result = core._make_variogram_parameter_list(
        "exponential", {"psill": 4.0, "range": 2.0, "nugget": 1.0}
    )
    assert result == [4.0, 2.0, 1.0]

    with pytest.raises(TypeError):
        core._make_variogram_parameter_list("custom", {"junk": 1.0})
    with pytest.raises(ValueError):
        core._make_variogram_parameter_list("blarg", {"junk": 1.0})

    # tests for third case - variogram_model_parameters is list
    with pytest.raises(ValueError):
        core._make_variogram_parameter_list("linear", [1.0, 2.0, 3.0])
    result = core._make_variogram_parameter_list("linear", [1.0, 2.0])
    assert result == [1.0, 2.0]

    with pytest.raises(ValueError):
        core._make_variogram_parameter_list("power", [1.0, 2.0])

    result = core._make_variogram_parameter_list("power", [1.0, 2.0, 3.0])
    assert result == [1.0, 2.0, 3.0]

    with pytest.raises(ValueError):
        core._make_variogram_parameter_list("exponential", [1.0, 2.0, 3.0, 4.0])
    result = core._make_variogram_parameter_list("exponential", [5.0, 2.0, 1.0])
    assert result == [4.0, 2.0, 1.0]

    result = core._make_variogram_parameter_list("custom", [1.0, 2.0, 3.0])
    assert result == [1.0, 2.0, 3]

    with pytest.raises(ValueError):
        core._make_variogram_parameter_list("junk", [1.0, 1.0, 1.0])

    # test for last case - make sure function handles incorrect
    # variogram_model_parameters type appropriately
    with pytest.raises(TypeError):
        core._make_variogram_parameter_list("linear", "tacos")


def test_core_initialize_variogram_model(validation_ref):

    data, _, _ = validation_ref

    # Note the variogram_function argument is not a string in real life...
    # core._initialize_variogram_model also checks the length of input
    # lists, which is redundant now because the same tests are done in
    # core._make_variogram_parameter_list
    with pytest.raises(ValueError):
        core._initialize_variogram_model(
            np.vstack((data[:, 0], data[:, 1])).T,
            data[:, 2],
            "linear",
            [0.0],
            "linear",
            6,
            False,
            "euclidean",
        )
    with pytest.raises(ValueError):
        core._initialize_variogram_model(
            np.vstack((data[:, 0], data[:, 1])).T,
            data[:, 2],
            "spherical",
            [0.0],
            "spherical",
            6,
            False,
            "euclidean",
        )

    # core._initialize_variogram_model does also check coordinate type,
    # this is NOT redundant
    with pytest.raises(ValueError):
        core._initialize_variogram_model(
            np.vstack((data[:, 0], data[:, 1])).T,
            data[:, 2],
            "spherical",
            [0.0, 0.0, 0.0],
            "spherical",
            6,
            False,
            "tacos",
        )

    x = np.array([1.0 + n / np.sqrt(2) for n in range(4)])
    y = np.array([1.0 + n / np.sqrt(2) for n in range(4)])
    z = np.arange(1.0, 5.0, 1.0)
    lags, semivariance, variogram_model_parameters = core._initialize_variogram_model(
        np.vstack((x, y)).T, z, "linear", [0.0, 0.0], "linear", 6, False, "euclidean"
    )

    assert_allclose(lags, np.array([1.0, 2.0, 3.0]))
    assert_allclose(semivariance, np.array([0.5, 2.0, 4.5]))


def test_core_initialize_variogram_model_3d(sample_data_3d):

    data, _, _ = sample_data_3d

    # Note the variogram_function argument is not a string in real life...
    # again, these checks in core._initialize_variogram_model are redundant
    # now because the same tests are done in
    # core._make_variogram_parameter_list
    with pytest.raises(ValueError):
        core._initialize_variogram_model(
            np.vstack((data[:, 0], data[:, 1], data[:, 2])).T,
            data[:, 3],
            "linear",
            [0.0],
            "linear",
            6,
            False,
            "euclidean",
        )

    with pytest.raises(ValueError):
        core._initialize_variogram_model(
            np.vstack((data[:, 0], data[:, 1], data[:, 2])).T,
            data[:, 3],
            "spherical",
            [0.0],
            "spherical",
            6,
            False,
            "euclidean",
        )

    with pytest.raises(ValueError):
        core._initialize_variogram_model(
            np.vstack((data[:, 0], data[:, 1], data[:, 2])).T,
            data[:, 3],
            "linear",
            [0.0, 0.0],
            "linear",
            6,
            False,
            "geographic",
        )

    lags, semivariance, variogram_model_parameters = core._initialize_variogram_model(
        np.vstack(
            (
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([1.0, 2.0, 3.0, 4.0]),
            )
        ).T,
        np.array([1.0, 2.0, 3.0, 4.0]),
        "linear",
        [0.0, 0.0],
        "linear",
        3,
        False,
        "euclidean",
    )
    assert_allclose(
        lags, np.array([np.sqrt(3.0), 2.0 * np.sqrt(3.0), 3.0 * np.sqrt(3.0)])
    )
    assert_allclose(semivariance, np.array([0.5, 2.0, 4.5]))


def test_core_calculate_variogram_model():

    res = core._calculate_variogram_model(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([2.05, 2.95, 4.05, 4.95]),
        "linear",
        variogram_models.linear_variogram_model,
        False,
    )
    assert_allclose(res, np.array([0.98, 1.05]), 0.01, 0.01)

    res = core._calculate_variogram_model(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([2.05, 2.95, 4.05, 4.95]),
        "linear",
        variogram_models.linear_variogram_model,
        True,
    )
    assert_allclose(res, np.array([0.98, 1.05]), 0.01, 0.01)

    res = core._calculate_variogram_model(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 2.8284271, 5.1961524, 8.0]),
        "power",
        variogram_models.power_variogram_model,
        False,
    )
    assert_allclose(res, np.array([1.0, 1.5, 0.0]), 0.001, 0.001)

    res = core._calculate_variogram_model(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 1.4142, 1.7321, 2.0]),
        "power",
        variogram_models.power_variogram_model,
        False,
    )
    assert_allclose(res, np.array([1.0, 0.5, 0.0]), 0.001, 0.001)

    res = core._calculate_variogram_model(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.2642, 1.7293, 1.9004, 1.9634]),
        "exponential",
        variogram_models.exponential_variogram_model,
        False,
    )
    assert_allclose(res, np.array([2.0, 3.0, 0.0]), 0.001, 0.001)

    res = core._calculate_variogram_model(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.5769, 1.4872, 1.9065, 1.9914]),
        "gaussian",
        variogram_models.gaussian_variogram_model,
        False,
    )
    assert_allclose(res, np.array([2.0, 3.0, 0.0]), 0.001, 0.001)

    res = core._calculate_variogram_model(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([3.33060952, 3.85063879, 3.96667301, 3.99256374]),
        "exponential",
        variogram_models.exponential_variogram_model,
        False,
    )
    assert_allclose(res, np.array([3.0, 2.0, 1.0]), 0.001, 0.001)

    res = core._calculate_variogram_model(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([2.60487044, 3.85968813, 3.99694817, 3.99998564]),
        "gaussian",
        variogram_models.gaussian_variogram_model,
        False,
    )
    assert_allclose(res, np.array([3.0, 2.0, 1.0]), 0.001, 0.001)


def test_core_krige():

    # Example 3.2 from Kitanidis
    data = np.array([[9.7, 47.6, 1.22], [43.8, 24.6, 2.822]])
    z, ss = core._krige(
        np.vstack((data[:, 0], data[:, 1])).T,
        data[:, 2],
        np.array([18.8, 67.9]),
        variogram_models.linear_variogram_model,
        [0.006, 0.1],
        "euclidean",
    )
    assert z == approx(1.6364, rel=1e-4)
    assert ss == approx(0.4201, rel=1e-4)

    z, ss = core._krige(
        np.vstack((data[:, 0], data[:, 1])).T,
        data[:, 2],
        np.array([43.8, 24.6]),
        variogram_models.linear_variogram_model,
        [0.006, 0.1],
        "euclidean",
    )
    assert z == approx(2.822, rel=1e-3)
    assert ss == approx(0.0, rel=1e-3)


def test_core_krige_3d():

    # Adapted from example 3.2 from Kitanidis
    data = np.array([[9.7, 47.6, 1.0, 1.22], [43.8, 24.6, 1.0, 2.822]])
    z, ss = core._krige(
        np.vstack((data[:, 0], data[:, 1], data[:, 2])).T,
        data[:, 3],
        np.array([18.8, 67.9, 1.0]),
        variogram_models.linear_variogram_model,
        [0.006, 0.1],
        "euclidean",
    )
    assert z == approx(1.6364, rel=1e-4)
    assert ss == approx(0.4201, rel=1e-4)

    z, ss = core._krige(
        np.vstack((data[:, 0], data[:, 1], data[:, 2])).T,
        data[:, 3],
        np.array([43.8, 24.6, 1.0]),
        variogram_models.linear_variogram_model,
        [0.006, 0.1],
        "euclidean",
    )
    assert z == approx(2.822, rel=1e-3)
    assert ss == approx(0.0, rel=1e-3)


def test_non_exact():
    # custom data for this test
    data = np.array(
        [[0.0, 0.0, 0.47], [1.5, 1.5, 0.56], [3, 3, 0.74], [4.5, 4.5, 1.47],]
    )

    # construct grid points so diagonal
    # is identical to input points
    gridx = np.arange(0.0, 4.51, 1.5)
    gridy = np.arange(0.0, 4.51, 1.5)

    ok = OrdinaryKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=[500.0, 3000.0, 5.0],
    )
    z, ss = ok.execute("grid", gridx, gridy, backend="vectorized")

    ok_non_exact = OrdinaryKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=[500.0, 3000.0, 5.0],
        exact_values=False,
    )
    z_non_exact, ss_non_exact = ok_non_exact.execute(
        "grid", gridx, gridy, backend="vectorized"
    )

    in_values = np.diag(z)

    # test that krig field
    # at input location are identical
    # to the inputs themselves  with
    # exact_values == True
    assert_allclose(in_values, data[:, 2])

    # test that krig field
    # at input location are different
    # than the inputs themselves
    # with exact_values == False
    assert ~np.allclose(in_values, data[:, 2])

    # test that off diagonal values are the same
    # by filling with dummy value and comparing
    # each entry in array
    np.fill_diagonal(z, 0.0)
    np.fill_diagonal(z_non_exact, 0.0)

    assert_allclose(z, z_non_exact)


def test_ok(validation_ref):

    # Test to compare OK results to those obtained using KT3D_H2O.
    # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater,
    # vol. 47, no. 4, 580-586.)

    data, (ok_test_answer, gridx, gridy), _ = validation_ref

    ok = OrdinaryKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=[500.0, 3000.0, 0.0],
    )
    z, ss = ok.execute("grid", gridx, gridy, backend="vectorized")
    assert_allclose(z, ok_test_answer)
    z, ss = ok.execute("grid", gridx, gridy, backend="loop")
    assert_allclose(z, ok_test_answer)


def test_ok_update_variogram_model(validation_ref):

    data, (ok_test_answer, gridx, gridy), _ = validation_ref

    with pytest.raises(ValueError):
        OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model="blurg")

    ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2])
    variogram_model = ok.variogram_model
    variogram_parameters = ok.variogram_model_parameters
    anisotropy_scaling = ok.anisotropy_scaling
    anisotropy_angle = ok.anisotropy_angle

    with pytest.raises(ValueError):
        ok.update_variogram_model("blurg")

    ok.update_variogram_model("power", anisotropy_scaling=3.0, anisotropy_angle=45.0)

    # TODO: check that new parameters equal to the set parameters
    assert variogram_model != ok.variogram_model
    assert not np.array_equal(variogram_parameters, ok.variogram_model_parameters)
    assert anisotropy_scaling != ok.anisotropy_scaling
    assert anisotropy_angle != ok.anisotropy_angle


def test_ok_get_variogram_points(validation_ref):
    # Test to compare the variogram of OK results to those obtained using
    # KT3D_H2O.
    # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater,
    # vol. 47, no. 4, 580-586.)

    # Variogram parameters
    _variogram_parameters = [500.0, 3000.0, 0.0]

    data, _, (ok_test_answer, gridx, gridy) = validation_ref

    ok = OrdinaryKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=_variogram_parameters,
    )

    # Get the variogram points from the UniversalKriging instance
    lags, calculated_variogram = ok.get_variogram_points()

    # Generate the expected variogram points according to the
    # exponential variogram model
    expected_variogram = variogram_models.exponential_variogram_model(
        _variogram_parameters, lags
    )

    assert_allclose(calculated_variogram, expected_variogram)


def test_ok_execute(sample_data_2d):

    data, (gridx, gridy, _), mask_ref = sample_data_2d

    ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2])

    with pytest.raises(ValueError):
        OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], exact_values="blurg")

    ok_non_exact = OrdinaryKriging(
        data[:, 0], data[:, 1], data[:, 2], exact_values=False
    )

    with pytest.raises(ValueError):
        ok.execute("blurg", gridx, gridy)

    z, ss = ok.execute("grid", gridx, gridy, backend="vectorized")
    shape = (gridy.size, gridx.size)
    assert z.shape == shape
    assert ss.shape == shape
    assert np.amax(z) != np.amin(z)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(z)

    z, ss = ok.execute("grid", gridx, gridy, backend="loop")
    shape = (gridy.size, gridx.size)
    assert z.shape == shape
    assert ss.shape == shape
    assert np.amax(z) != np.amin(z)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(z)

    z1, ss1 = ok_non_exact.execute("grid", gridx, gridy, backend="loop")
    assert_allclose(z1, z)
    assert_allclose(ss1, ss)

    z, ss = ok_non_exact.execute("grid", gridx, gridy, backend="loop")
    shape = (gridy.size, gridx.size)
    assert z.shape == shape
    assert ss.shape == shape
    assert np.amax(z) != np.amin(z)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(z)

    with pytest.raises(IOError):
        ok.execute("masked", gridx, gridy, backend="vectorized")
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        ok.execute("masked", gridx, gridy, mask=mask, backend="vectorized")
    z, ss = ok.execute("masked", gridx, gridy, mask=mask_ref, backend="vectorized")
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked
    z, ss = ok.execute("masked", gridx, gridy, mask=mask_ref.T, backend="vectorized")
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked

    with pytest.raises(IOError):
        ok.execute("masked", gridx, gridy, backend="loop")
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        ok.execute("masked", gridx, gridy, mask=mask, backend="loop")
    z, ss = ok.execute("masked", gridx, gridy, mask=mask_ref, backend="loop")
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked
    z, ss = ok.execute("masked", gridx, gridy, mask=mask_ref.T, backend="loop")
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked

    z, ss = ok_non_exact.execute(
        "masked", gridx, gridy, mask=mask_ref.T, backend="loop"
    )
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked

    with pytest.raises(ValueError):
        ok.execute(
            "points",
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
            backend="vectorized",
        )
    z, ss = ok.execute("points", gridx[0], gridy[0], backend="vectorized")
    assert z.shape == (1,)
    assert ss.shape == (1,)

    with pytest.raises(ValueError):
        ok.execute(
            "points", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]), backend="loop"
        )
    z, ss = ok.execute("points", gridx[0], gridy[0], backend="loop")
    assert z.shape == (1,)
    assert ss.shape == (1,)


def test_cython_ok(sample_data_2d):
    data, (gridx, gridy, _), mask_ref = sample_data_2d

    ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2])
    ok_non_exact = OrdinaryKriging(
        data[:, 0], data[:, 1], data[:, 2], exact_values=False
    )

    z1, ss1 = ok.execute("grid", gridx, gridy, backend="loop")
    z2, ss2 = ok.execute("grid", gridx, gridy, backend="C")
    assert_allclose(z1, z2)
    assert_allclose(ss1, ss2)

    z1, ss1 = ok_non_exact.execute("grid", gridx, gridy, backend="loop")
    z2, ss2 = ok_non_exact.execute("grid", gridx, gridy, backend="C")
    assert_allclose(z1, z2)
    assert_allclose(ss1, ss2)

    closest_points = 4

    z1, ss1 = ok.execute(
        "grid", gridx, gridy, backend="loop", n_closest_points=closest_points
    )
    z2, ss2 = ok.execute(
        "grid", gridx, gridy, backend="C", n_closest_points=closest_points
    )
    assert_allclose(z1, z2)
    assert_allclose(ss1, ss2)

    z1, ss1 = ok_non_exact.execute(
        "grid", gridx, gridy, backend="loop", n_closest_points=closest_points
    )
    z2, ss2 = ok_non_exact.execute(
        "grid", gridx, gridy, backend="C", n_closest_points=closest_points
    )
    assert_allclose(z1, z2)
    assert_allclose(ss1, ss2)


def test_uk(validation_ref):

    # Test to compare UK with linear drift to results from KT3D_H2O.
    # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater,
    # vol. 47, no. 4, 580-586.)

    data, _, (uk_test_answer, gridx, gridy) = validation_ref

    uk = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=[500.0, 3000.0, 0.0],
        drift_terms=["regional_linear"],
    )
    z, ss = uk.execute("grid", gridx, gridy, backend="vectorized")
    assert_allclose(z, uk_test_answer)
    z, ss = uk.execute("grid", gridx, gridy, backend="loop")
    assert_allclose(z, uk_test_answer)


def test_uk_update_variogram_model(sample_data_2d):

    data, (gridx, gridy, _), mask_ref = sample_data_2d

    with pytest.raises(ValueError):
        UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model="blurg")
    with pytest.raises(ValueError):
        UniversalKriging(data[:, 0], data[:, 1], data[:, 2], drift_terms=["external_Z"])
    with pytest.raises(ValueError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            drift_terms=["external_Z"],
            external_drift=np.array([0]),
        )
    with pytest.raises(ValueError):
        UniversalKriging(data[:, 0], data[:, 1], data[:, 2], drift_terms=["point_log"])

    uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2])
    variogram_model = uk.variogram_model
    variogram_parameters = uk.variogram_model_parameters
    anisotropy_scaling = uk.anisotropy_scaling
    anisotropy_angle = uk.anisotropy_angle

    with pytest.raises(ValueError):
        uk.update_variogram_model("blurg")
    uk.update_variogram_model("power", anisotropy_scaling=3.0, anisotropy_angle=45.0)
    # TODO: check that the new parameters are equal to the expected ones
    assert variogram_model != uk.variogram_model
    assert not np.array_equal(variogram_parameters, uk.variogram_model_parameters)
    assert anisotropy_scaling != uk.anisotropy_scaling
    assert anisotropy_angle != uk.anisotropy_angle


def test_uk_get_variogram_points(validation_ref):
    # Test to compare the variogram of UK with linear drift to results from
    # KT3D_H2O.
    # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater,
    # vol. 47, no. 4, 580-586.)

    # Variogram parameters
    _variogram_parameters = [500.0, 3000.0, 0.0]

    data, _, (uk_test_answer, gridx, gridy) = validation_ref

    uk = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=_variogram_parameters,
        drift_terms=["regional_linear"],
    )

    # Get the variogram points from the UniversalKriging instance
    lags, calculated_variogram = uk.get_variogram_points()

    # Generate the expected variogram points according to the
    # exponential variogram model
    expected_variogram = variogram_models.exponential_variogram_model(
        _variogram_parameters, lags
    )

    assert_allclose(calculated_variogram, expected_variogram)


def test_uk_calculate_data_point_zscalars(sample_data_2d):

    data, (gridx, gridy, _), mask_ref = sample_data_2d

    dem = np.arange(0.0, 5.1, 0.1)
    dem = np.repeat(dem[np.newaxis, :], 6, axis=0)
    dem_x = np.arange(0.0, 5.1, 0.1)
    dem_y = np.arange(0.0, 6.0, 1.0)

    with pytest.raises(ValueError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="linear",
            variogram_parameters=[1.0, 0.0],
            drift_terms=["external_Z"],
        )
    with pytest.raises(ValueError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="linear",
            variogram_parameters=[1.0, 0.0],
            drift_terms=["external_Z"],
            external_drift=dem,
        )
    with pytest.raises(ValueError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="linear",
            variogram_parameters=[1.0, 0.0],
            drift_terms=["external_Z"],
            external_drift=dem,
            external_drift_x=dem_x,
            external_drift_y=np.arange(0.0, 5.0, 1.0),
        )

    uk = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        variogram_parameters=[1.0, 0.0],
        drift_terms=["external_Z"],
        external_drift=dem,
        external_drift_x=dem_x,
        external_drift_y=dem_y,
    )
    assert_allclose(uk.z_scalars, data[:, 0])

    xi, yi = np.meshgrid(np.arange(0.0, 5.3, 0.1), gridy)
    with pytest.raises(ValueError):
        uk._calculate_data_point_zscalars(xi, yi)

    xi, yi = np.meshgrid(np.arange(0.0, 5.0, 0.1), gridy)
    z_scalars = uk._calculate_data_point_zscalars(xi, yi)
    assert_allclose(z_scalars[0, :], np.arange(0.0, 5.0, 0.1))


def test_uk_execute_single_point():

    # Test data and answer from lecture notes by Nicolas Christou, UCLA Stats
    data = np.array(
        [
            [61.0, 139.0, 477.0],
            [63.0, 140.0, 696.0],
            [64.0, 129.0, 227.0],
            [68.0, 128.0, 646.0],
            [71.0, 140.0, 606.0],
            [73.0, 141.0, 791.0],
            [75.0, 128.0, 783.0],
        ]
    )
    point = (65.0, 137.0)
    z_answer = 567.54
    ss_answer = 9.044

    uk = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=[10.0, 9.99, 0.0],
        drift_terms=["regional_linear"],
    )
    z, ss = uk.execute(
        "points", np.array([point[0]]), np.array([point[1]]), backend="vectorized"
    )
    assert z_answer == approx(z[0], rel=0.1)
    assert ss_answer == approx(ss[0], rel=0.1)

    z, ss = uk.execute(
        "points", np.array([61.0]), np.array([139.0]), backend="vectorized"
    )
    assert z[0] == approx(477.0, rel=1e-3)
    assert ss[0] == approx(0.0, rel=1e-3)

    z, ss = uk.execute("points", np.array([61.0]), np.array([139.0]), backend="loop")
    assert z[0] == approx(477.0, rel=1e-3)
    assert ss[0] == approx(0.0, rel=1e-3)


def test_uk_execute(sample_data_2d):

    data, (gridx, gridy, _), mask_ref = sample_data_2d

    uk = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["regional_linear"],
    )

    with pytest.raises(ValueError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="linear",
            drift_terms=["regional_linear"],
            exact_values="blurg",
        )

    uk_non_exact = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["regional_linear"],
    )

    with pytest.raises(ValueError):
        uk.execute("blurg", gridx, gridy)
    with pytest.raises(ValueError):
        uk.execute("grid", gridx, gridy, backend="mrow")

    z, ss = uk.execute("grid", gridx, gridy, backend="vectorized")
    shape = (gridy.size, gridx.size)
    assert z.shape == shape
    assert ss.shape == shape
    assert np.amax(z) != np.amin(z)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(z)

    z1, ss1 = uk_non_exact.execute("grid", gridx, gridy, backend="vectorized")
    assert_allclose(z1, z)
    assert_allclose(ss1, ss)

    z, ss = uk_non_exact.execute("grid", gridx, gridy, backend="vectorized")
    shape = (gridy.size, gridx.size)
    assert z.shape == shape
    assert ss.shape == shape
    assert np.amax(z) != np.amin(z)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(z)

    z, ss = uk.execute("grid", gridx, gridy, backend="loop")
    shape = (gridy.size, gridx.size)
    assert z.shape == shape
    assert ss.shape == shape
    assert np.amax(z) != np.amin(z)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(z)

    with pytest.raises(IOError):
        uk.execute("masked", gridx, gridy, backend="vectorized")
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        uk.execute("masked", gridx, gridy, mask=mask, backend="vectorized")
    z, ss = uk.execute("masked", gridx, gridy, mask=mask_ref, backend="vectorized")
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked
    z, ss = uk.execute("masked", gridx, gridy, mask=mask_ref.T, backend="vectorized")
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked

    with pytest.raises(IOError):
        uk.execute("masked", gridx, gridy, backend="loop")
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        uk.execute("masked", gridx, gridy, mask=mask, backend="loop")
    z, ss = uk.execute("masked", gridx, gridy, mask=mask_ref, backend="loop")
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked
    z, ss = uk.execute("masked", gridx, gridy, mask=mask_ref.T, backend="loop")
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked

    z, ss = uk_non_exact.execute(
        "masked", gridx, gridy, mask=mask_ref.T, backend="loop"
    )
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked

    with pytest.raises(ValueError):
        uk.execute(
            "points",
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
            backend="vectorized",
        )
    z, ss = uk.execute("points", gridx[0], gridy[0], backend="vectorized")
    assert z.shape == (1,)
    assert ss.shape == (1,)

    with pytest.raises(ValueError):
        uk.execute(
            "points", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]), backend="loop"
        )
    z, ss = uk.execute("points", gridx[0], gridy[0], backend="loop")
    assert z.shape == (1,)
    assert ss.shape == (1,)


def test_ok_uk_produce_same_result(validation_ref):

    data, _, (uk_test_answer, gridx_ref, gridy_ref) = validation_ref

    gridx = np.linspace(1067000.0, 1072000.0, 100)
    gridy = np.linspace(241500.0, 244000.0, 100)
    ok = OrdinaryKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )
    z_ok, ss_ok = ok.execute("grid", gridx, gridy, backend="vectorized")
    uk = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )
    uk_non_exact = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        exact_values=False,
    )
    z_uk, ss_uk = uk.execute("grid", gridx, gridy, backend="vectorized")
    assert_allclose(z_ok, z_uk)
    assert_allclose(ss_ok, ss_uk)

    z_uk, ss_uk = uk_non_exact.execute("grid", gridx, gridy, backend="vectorized")
    assert_allclose(z_ok, z_uk)
    assert_allclose(ss_ok, ss_uk)

    z_ok, ss_ok = ok.execute("grid", gridx, gridy, backend="loop")
    z_uk, ss_uk = uk.execute("grid", gridx, gridy, backend="loop")
    assert_allclose(z_ok, z_uk)
    assert_allclose(ss_ok, ss_uk)

    z_uk, ss_uk = uk_non_exact.execute("grid", gridx, gridy, backend="loop")
    assert_allclose(z_ok, z_uk)
    assert_allclose(ss_ok, ss_uk)


def test_ok_backends_produce_same_result(validation_ref):

    data, _, (uk_test_answer, gridx_ref, gridy_ref) = validation_ref

    gridx = np.linspace(1067000.0, 1072000.0, 100)
    gridy = np.linspace(241500.0, 244000.0, 100)
    ok = OrdinaryKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )
    z_ok_v, ss_ok_v = ok.execute("grid", gridx, gridy, backend="vectorized")
    z_ok_l, ss_ok_l = ok.execute("grid", gridx, gridy, backend="loop")
    assert_allclose(z_ok_v, z_ok_l)
    assert_allclose(ss_ok_v, ss_ok_l)


def test_uk_backends_produce_same_result(validation_ref):

    data, _, (uk_test_answer, gridx_ref, gridy_ref) = validation_ref

    gridx = np.linspace(1067000.0, 1072000.0, 100)
    gridy = np.linspace(241500.0, 244000.0, 100)
    uk = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )
    z_uk_v, ss_uk_v = uk.execute("grid", gridx, gridy, backend="vectorized")
    z_uk_l, ss_uk_l = uk.execute("grid", gridx, gridy, backend="loop")
    assert_allclose(z_uk_v, z_uk_l)
    assert_allclose(ss_uk_v, ss_uk_l)


def test_kriging_tools(sample_data_2d):

    data, (gridx, gridy, gridx_2), mask_ref = sample_data_2d

    ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2])
    z_write, ss_write = ok.execute("grid", gridx, gridy)

    kt.write_asc_grid(
        gridx,
        gridy,
        z_write,
        filename=os.path.join(BASE_DIR, "test_data/temp.asc"),
        style=1,
    )
    z_read, x_read, y_read, cellsize, no_data = kt.read_asc_grid(
        os.path.join(BASE_DIR, "test_data/temp.asc")
    )
    assert_allclose(z_write, z_read, 0.01, 0.01)
    assert_allclose(gridx, x_read)
    assert_allclose(gridy, y_read)

    z_write, ss_write = ok.execute("masked", gridx, gridy, mask=mask_ref)
    kt.write_asc_grid(
        gridx,
        gridy,
        z_write,
        filename=os.path.join(BASE_DIR, "test_data/temp.asc"),
        style=1,
    )
    z_read, x_read, y_read, cellsize, no_data = kt.read_asc_grid(
        os.path.join(BASE_DIR, "test_data/temp.asc")
    )
    assert np.ma.allclose(
        z_write,
        np.ma.masked_where(z_read == no_data, z_read),
        masked_equal=True,
        rtol=0.01,
        atol=0.01,
    )
    assert_allclose(gridx, x_read)
    assert_allclose(gridy, y_read)

    ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2])
    z_write, ss_write = ok.execute("grid", gridx_2, gridy)

    kt.write_asc_grid(
        gridx_2,
        gridy,
        z_write,
        filename=os.path.join(BASE_DIR, "test_data/temp.asc"),
        style=2,
    )
    z_read, x_read, y_read, cellsize, no_data = kt.read_asc_grid(
        os.path.join(BASE_DIR, "test_data/temp.asc")
    )
    assert_allclose(z_write, z_read, 0.01, 0.01)
    assert_allclose(gridx_2, x_read)
    assert_allclose(gridy, y_read)

    os.remove(os.path.join(BASE_DIR, "test_data/temp.asc"))


# http://doc.pytest.org/en/latest/skipping.html#id1
@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_uk_three_primary_drifts(sample_data_2d):

    data, (gridx, gridy, gridx_2), mask_ref = sample_data_2d

    well = np.array([[1.1, 1.1, -1.0]])
    dem = np.arange(0.0, 5.1, 0.1)
    dem = np.repeat(dem[np.newaxis, :], 6, axis=0)
    dem_x = np.arange(0.0, 5.1, 0.1)
    dem_y = np.arange(0.0, 6.0, 1.0)

    uk = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["regional_linear", "external_Z", "point_log"],
        point_drift=well,
        external_drift=dem,
        external_drift_x=dem_x,
        external_drift_y=dem_y,
    )

    z, ss = uk.execute("grid", gridx, gridy, backend="vectorized")
    assert z.shape == (gridy.shape[0], gridx.shape[0])
    assert ss.shape == (gridy.shape[0], gridx.shape[0])
    assert np.all(np.isfinite(z))
    assert not np.all(np.isnan(z))
    assert np.all(np.isfinite(ss))
    assert not np.all(np.isnan(ss))

    z, ss = uk.execute("grid", gridx, gridy, backend="loop")
    assert z.shape == (gridy.shape[0], gridx.shape[0])
    assert ss.shape == (gridy.shape[0], gridx.shape[0])
    assert np.all(np.isfinite(z))
    assert not np.all(np.isnan(z))
    assert np.all(np.isfinite(ss))
    assert not np.all(np.isnan(ss))


def test_uk_specified_drift(sample_data_2d):

    data, (gridx, gridy, gridx_2), mask_ref = sample_data_2d

    xg, yg = np.meshgrid(gridx, gridy)
    well = np.array([[1.1, 1.1, -1.0]])
    point_log = (
        well[0, 2]
        * np.log(np.sqrt((xg - well[0, 0]) ** 2.0 + (yg - well[0, 1]) ** 2.0))
        * -1.0
    )
    if np.any(np.isinf(point_log)):
        point_log[np.isinf(point_log)] = -100.0 * well[0, 2] * -1.0
    point_log_data = (
        well[0, 2]
        * np.log(
            np.sqrt((data[:, 0] - well[0, 0]) ** 2.0 + (data[:, 1] - well[0, 1]) ** 2.0)
        )
        * -1.0
    )
    if np.any(np.isinf(point_log_data)):
        point_log_data[np.isinf(point_log_data)] = -100.0 * well[0, 2] * -1.0

    with pytest.raises(ValueError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="linear",
            drift_terms=["specified"],
        )
    with pytest.raises(TypeError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="linear",
            drift_terms=["specified"],
            specified_drift=data[:, 0],
        )
    with pytest.raises(ValueError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="linear",
            drift_terms=["specified"],
            specified_drift=[data[:2, 0]],
        )

    uk_spec = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["specified"],
        specified_drift=[data[:, 0], data[:, 1]],
    )
    with pytest.raises(ValueError):
        uk_spec.execute("grid", gridx, gridy, specified_drift_arrays=[gridx, gridy])
    with pytest.raises(TypeError):
        uk_spec.execute("grid", gridx, gridy, specified_drift_arrays=gridx)
    with pytest.raises(ValueError):
        uk_spec.execute("grid", gridx, gridy, specified_drift_arrays=[xg])
    z_spec, ss_spec = uk_spec.execute(
        "grid", gridx, gridy, specified_drift_arrays=[xg, yg]
    )

    uk_lin = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["regional_linear"],
    )
    z_lin, ss_lin = uk_lin.execute("grid", gridx, gridy)

    assert_allclose(z_spec, z_lin)
    assert_allclose(ss_spec, ss_lin)

    uk_spec = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["specified"],
        specified_drift=[point_log_data],
    )
    z_spec, ss_spec = uk_spec.execute(
        "grid", gridx, gridy, specified_drift_arrays=[point_log]
    )

    uk_lin = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["point_log"],
        point_drift=well,
    )
    z_lin, ss_lin = uk_lin.execute("grid", gridx, gridy)

    assert_allclose(z_spec, z_lin)
    assert_allclose(ss_spec, ss_lin)

    uk_spec = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["specified"],
        specified_drift=[data[:, 0], data[:, 1], point_log_data],
    )
    z_spec, ss_spec = uk_spec.execute(
        "grid", gridx, gridy, specified_drift_arrays=[xg, yg, point_log]
    )
    uk_lin = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["regional_linear", "point_log"],
        point_drift=well,
    )
    z_lin, ss_lin = uk_lin.execute("grid", gridx, gridy)

    assert_allclose(z_spec, z_lin)
    assert_allclose(ss_spec, ss_lin)


def test_uk_functional_drift(sample_data_2d):

    data, (gridx, gridy, gridx_2), mask_ref = sample_data_2d

    well = np.array([[1.1, 1.1, -1.0]])
    func_x = lambda x, y: x  # noqa
    func_y = lambda x, y: y  # noqa

    def func_well(x, y):
        return -well[0, 2] * np.log(
            np.sqrt((x - well[0, 0]) ** 2.0 + (y - well[0, 1]) ** 2.0)
        )

    with pytest.raises(ValueError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="linear",
            drift_terms=["functional"],
        )
    with pytest.raises(TypeError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="linear",
            drift_terms=["functional"],
            functional_drift=func_x,
        )

    uk_func = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["functional"],
        functional_drift=[func_x, func_y],
    )
    z_func, ss_func = uk_func.execute("grid", gridx, gridy)
    uk_lin = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["regional_linear"],
    )
    z_lin, ss_lin = uk_lin.execute("grid", gridx, gridy)
    assert_allclose(z_func, z_lin)
    assert_allclose(ss_func, ss_lin)

    uk_func = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["functional"],
        functional_drift=[func_well],
    )
    z_func, ss_func = uk_func.execute("grid", gridx, gridy)
    uk_lin = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["point_log"],
        point_drift=well,
    )
    z_lin, ss_lin = uk_lin.execute("grid", gridx, gridy)
    assert_allclose(z_func, z_lin)
    assert_allclose(ss_func, ss_lin)

    uk_func = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["functional"],
        functional_drift=[func_x, func_y, func_well],
    )
    z_func, ss_func = uk_func.execute("grid", gridx, gridy)
    uk_lin = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        drift_terms=["regional_linear", "point_log"],
        point_drift=well,
    )
    z_lin, ss_lin = uk_lin.execute("grid", gridx, gridy)
    assert_allclose(z_func, z_lin)
    assert_allclose(ss_func, ss_lin)


def test_uk_with_external_drift(validation_ref):

    data, _, (uk_test_answer, gridx_ref, gridy_ref) = validation_ref

    dem, demx, demy, cellsize, no_data = kt.read_asc_grid(
        os.path.join(BASE_DIR, "test_data/test3_dem.asc")
    )
    uk = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="spherical",
        variogram_parameters=[500.0, 3000.0, 0.0],
        anisotropy_scaling=1.0,
        anisotropy_angle=0.0,
        drift_terms=["external_Z"],
        external_drift=dem,
        external_drift_x=demx,
        external_drift_y=demy,
        verbose=False,
    )
    answer, gridx, gridy, cellsize, no_data = kt.read_asc_grid(
        os.path.join(BASE_DIR, "test_data/test3_answer.asc")
    )

    z, ss = uk.execute("grid", gridx, gridy, backend="vectorized")
    assert_allclose(z, answer, **allclose_pars)

    z, ss = uk.execute("grid", gridx, gridy, backend="loop")
    assert_allclose(z, answer, **allclose_pars)


def test_force_exact():
    data = np.array([[1.0, 1.0, 2.0], [2.0, 2.0, 1.5], [3.0, 3.0, 1.0]])
    ok = OrdinaryKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        variogram_parameters=[1.0, 1.0],
    )
    z, ss = ok.execute("grid", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], backend="vectorized")
    assert z[0, 0] == approx(2.0)
    assert ss[0, 0] == approx(0.0)
    assert z[1, 1] == approx(1.5)
    assert ss[1, 1] == approx(0.0)
    assert z[2, 2] == approx(1.0)
    assert ss[2, 2] == approx(0.0)
    assert ss[0, 2] != approx(0.0)
    assert ss[2, 0] != approx(0.0)
    z, ss = ok.execute(
        "points", [1.0, 2.0, 3.0, 3.0], [2.0, 1.0, 1.0, 3.0], backend="vectorized"
    )
    assert ss[0] != approx(0.0)
    assert ss[1] != approx(0.0)
    assert ss[2] != approx(0.0)
    assert z[3] == approx(1.0)
    assert ss[3] == approx(0.0)
    z, ss = ok.execute(
        "grid", np.arange(0.0, 4.0, 0.1), np.arange(0.0, 4.0, 0.1), backend="vectorized"
    )
    assert z[10, 10] == approx(2.0)
    assert ss[10, 10] == approx(0.0)
    assert z[20, 20] == approx(1.5)
    assert ss[20, 20] == approx(0.0)
    assert z[30, 30] == approx(1.0)
    assert ss[30, 30] == approx(0.0)
    assert ss[0, 0] != approx(0.0)
    assert ss[15, 15] != approx(0.0)
    assert ss[10, 0] != approx(0.0)
    assert ss[0, 10] != approx(0.0)
    assert ss[20, 10] != approx(0.0)
    assert ss[10, 20] != approx(0.0)
    assert ss[30, 20] != approx(0.0)
    assert ss[20, 30] != approx(0.0)
    z, ss = ok.execute(
        "grid", np.arange(0.0, 3.1, 0.1), np.arange(2.1, 3.1, 0.1), backend="vectorized"
    )
    assert np.any(np.isclose(ss, 0))
    assert not np.any(np.isclose(ss[:9, :30], 0))
    assert not np.allclose(z[:9, :30], 0.0)
    z, ss = ok.execute(
        "grid", np.arange(0.0, 1.9, 0.1), np.arange(2.1, 3.1, 0.1), backend="vectorized"
    )
    assert not np.any(np.isclose(ss, 0))
    z, ss = ok.execute(
        "masked",
        np.arange(2.5, 3.5, 0.1),
        np.arange(2.5, 3.5, 0.25),
        backend="vectorized",
        mask=np.asarray(
            np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.0
        ),
    )
    assert np.isclose(ss[2, 5], 0)
    assert not np.allclose(ss, 0.0)

    z, ss = ok.execute("grid", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], backend="loop")
    assert z[0, 0] == approx(2.0)
    assert ss[0, 0] == approx(0.0)
    assert z[1, 1] == approx(1.5)
    assert ss[1, 1] == approx(0.0)
    assert z[2, 2] == approx(1.0)
    assert ss[2, 2] == approx(0.0)
    assert ss[0, 2] != approx(0.0)
    assert ss[2, 0] != approx(0.0)
    z, ss = ok.execute(
        "points", [1.0, 2.0, 3.0, 3.0], [2.0, 1.0, 1.0, 3.0], backend="loop"
    )
    assert ss[0] != approx(0.0)
    assert ss[1] != approx(0.0)
    assert ss[2] != approx(0.0)
    assert z[3] == approx(1.0)
    assert ss[3] == approx(0.0)
    z, ss = ok.execute(
        "grid", np.arange(0.0, 4.0, 0.1), np.arange(0.0, 4.0, 0.1), backend="loop"
    )
    assert z[10, 10] == approx(2.0)
    assert ss[10, 10] == approx(0.0)
    assert z[20, 20] == approx(1.5)
    assert ss[20, 20] == approx(0.0)
    assert z[30, 30] == approx(1.0)
    assert ss[30, 30] == approx(0.0)
    assert ss[0, 0] != approx(0.0)
    assert ss[15, 15] != approx(0.0)
    assert ss[10, 0] != approx(0.0)
    assert ss[0, 10] != approx(0.0)
    assert ss[20, 10] != approx(0.0)
    assert ss[10, 20] != approx(0.0)
    assert ss[30, 20] != approx(0.0)
    assert ss[20, 30] != approx(0.0)
    z, ss = ok.execute(
        "grid", np.arange(0.0, 3.1, 0.1), np.arange(2.1, 3.1, 0.1), backend="loop"
    )
    assert np.any(np.isclose(ss, 0))
    assert not np.any(np.isclose(ss[:9, :30], 0))
    assert not np.allclose(z[:9, :30], 0.0)
    z, ss = ok.execute(
        "grid", np.arange(0.0, 1.9, 0.1), np.arange(2.1, 3.1, 0.1), backend="loop"
    )
    assert not np.any(np.isclose(ss, 0))
    z, ss = ok.execute(
        "masked",
        np.arange(2.5, 3.5, 0.1),
        np.arange(2.5, 3.5, 0.25),
        backend="loop",
        mask=np.asarray(
            np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.0
        ),
    )
    assert np.isclose(ss[2, 5], 0)
    assert not np.allclose(ss, 0.0)

    uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2])
    z, ss = uk.execute("grid", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], backend="vectorized")
    assert z[0, 0] == approx(2.0)
    assert ss[0, 0] == approx(0.0)
    assert z[1, 1] == approx(1.5)
    assert ss[1, 1] == approx(0.0)
    assert z[2, 2] == approx(1.0)
    assert ss[2, 2] == approx(0.0)
    assert ss[0, 2] != approx(0.0)
    assert ss[2, 0] != approx(0.0)
    z, ss = uk.execute(
        "points", [1.0, 2.0, 3.0, 3.0], [2.0, 1.0, 1.0, 3.0], backend="vectorized"
    )
    assert ss[0] != approx(0.0)
    assert ss[1] != approx(0.0)
    assert ss[2] != approx(0.0)
    assert z[3] == approx(1.0)
    assert ss[3] == approx(0.0)
    z, ss = uk.execute(
        "grid", np.arange(0.0, 4.0, 0.1), np.arange(0.0, 4.0, 0.1), backend="vectorized"
    )
    assert z[10, 10] == approx(2.0)
    assert ss[10, 10] == approx(0.0)
    assert z[20, 20] == approx(1.5)
    assert ss[20, 20] == approx(0.0)
    assert z[30, 30] == approx(1.0)
    assert ss[30, 30] == approx(0.0)
    assert ss[0, 0] != approx(0.0)
    assert ss[15, 15] != approx(0.0)
    assert ss[10, 0] != approx(0.0)
    assert ss[0, 10] != approx(0.0)
    assert ss[20, 10] != approx(0.0)
    assert ss[10, 20] != approx(0.0)
    assert ss[30, 20] != approx(0.0)
    assert ss[20, 30] != approx(0.0)
    z, ss = uk.execute(
        "grid", np.arange(0.0, 3.1, 0.1), np.arange(2.1, 3.1, 0.1), backend="vectorized"
    )
    assert np.any(np.isclose(ss, 0))
    assert not np.any(np.isclose(ss[:9, :30], 0))
    assert not np.allclose(z[:9, :30], 0.0)
    z, ss = uk.execute(
        "grid", np.arange(0.0, 1.9, 0.1), np.arange(2.1, 3.1, 0.1), backend="vectorized"
    )
    assert not (np.any(np.isclose(ss, 0)))
    z, ss = uk.execute(
        "masked",
        np.arange(2.5, 3.5, 0.1),
        np.arange(2.5, 3.5, 0.25),
        backend="vectorized",
        mask=np.asarray(
            np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.0
        ),
    )
    assert np.isclose(ss[2, 5], 0)
    assert not np.allclose(ss, 0.0)
    z, ss = uk.execute("grid", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], backend="loop")
    assert z[0, 0] == approx(2.0)
    assert ss[0, 0] == approx(0.0)
    assert z[1, 1] == approx(1.5)
    assert ss[1, 1] == approx(0.0)
    assert z[2, 2] == approx(1.0)
    assert ss[2, 2] == approx(0.0)
    assert ss[0, 2] != approx(0.0)
    assert ss[2, 0] != approx(0.0)
    z, ss = uk.execute(
        "points", [1.0, 2.0, 3.0, 3.0], [2.0, 1.0, 1.0, 3.0], backend="loop"
    )
    assert ss[0] != approx(0.0)
    assert ss[1] != approx(0.0)
    assert ss[2] != approx(0.0)
    assert z[3] == approx(1.0)
    assert ss[3] == approx(0.0)
    z, ss = uk.execute(
        "grid", np.arange(0.0, 4.0, 0.1), np.arange(0.0, 4.0, 0.1), backend="loop"
    )
    assert z[10, 10] == approx(2.0)
    assert ss[10, 10] == approx(0.0)
    assert z[20, 20] == approx(1.5)
    assert ss[20, 20] == approx(0.0)
    assert z[30, 30] == approx(1.0)
    assert ss[30, 30] == approx(0.0)
    assert ss[0, 0] != approx(0.0)
    assert ss[15, 15] != approx(0.0)
    assert ss[10, 0] != approx(0.0)
    assert ss[0, 10] != approx(0.0)
    assert ss[20, 10] != approx(0.0)
    assert ss[10, 20] != approx(0.0)
    assert ss[30, 20] != approx(0.0)
    assert ss[20, 30] != approx(0.0)
    z, ss = uk.execute(
        "grid", np.arange(0.0, 3.1, 0.1), np.arange(2.1, 3.1, 0.1), backend="loop"
    )
    assert np.any(np.isclose(ss, 0))
    assert not np.any(np.isclose(ss[:9, :30], 0))
    assert not np.allclose(z[:9, :30], 0.0)
    z, ss = uk.execute(
        "grid", np.arange(0.0, 1.9, 0.1), np.arange(2.1, 3.1, 0.1), backend="loop"
    )
    assert not np.any(np.isclose(ss, 0))
    z, ss = uk.execute(
        "masked",
        np.arange(2.5, 3.5, 0.1),
        np.arange(2.5, 3.5, 0.25),
        backend="loop",
        mask=np.asarray(
            np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.0
        ),
    )
    assert np.isclose(ss[2, 5], 0)
    assert not np.allclose(ss, 0.0)

    z, ss = core._krige(
        np.vstack((data[:, 0], data[:, 1])).T,
        data[:, 2],
        np.array([1.0, 1.0]),
        variogram_models.linear_variogram_model,
        [1.0, 1.0],
        "euclidean",
    )
    assert z == approx(2.0)
    assert ss == approx(0.0)
    z, ss = core._krige(
        np.vstack((data[:, 0], data[:, 1])).T,
        data[:, 2],
        np.array([1.0, 2.0]),
        variogram_models.linear_variogram_model,
        [1.0, 1.0],
        "euclidean",
    )
    assert ss != approx(0.0)

    data = np.zeros((50, 3))
    x, y = np.meshgrid(np.arange(0.0, 10.0, 1.0), np.arange(0.0, 10.0, 2.0))
    data[:, 0] = np.ravel(x)
    data[:, 1] = np.ravel(y)
    data[:, 2] = np.ravel(x) * np.ravel(y)
    ok = OrdinaryKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        variogram_parameters=[100.0, 1.0],
    )
    z, ss = ok.execute(
        "grid",
        np.arange(0.0, 10.0, 1.0),
        np.arange(0.0, 10.0, 2.0),
        backend="vectorized",
    )
    assert_allclose(np.ravel(z), data[:, 2], **allclose_pars)
    assert_allclose(ss, 0.0, **allclose_pars)
    z, ss = ok.execute(
        "grid",
        np.arange(0.5, 10.0, 1.0),
        np.arange(0.5, 10.0, 2.0),
        backend="vectorized",
    )
    assert not np.allclose(np.ravel(z), data[:, 2])
    assert not np.allclose(ss, 0.0)
    z, ss = ok.execute(
        "grid", np.arange(0.0, 10.0, 1.0), np.arange(0.0, 10.0, 2.0), backend="loop"
    )
    assert_allclose(np.ravel(z), data[:, 2], **allclose_pars)
    assert_allclose(ss, 0.0, **allclose_pars)
    z, ss = ok.execute(
        "grid", np.arange(0.5, 10.0, 1.0), np.arange(0.5, 10.0, 2.0), backend="loop"
    )
    assert not np.allclose(np.ravel(z), data[:, 2])
    assert not np.allclose(ss, 0.0)

    uk = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        variogram_parameters=[100.0, 1.0],
    )
    z, ss = uk.execute(
        "grid",
        np.arange(0.0, 10.0, 1.0),
        np.arange(0.0, 10.0, 2.0),
        backend="vectorized",
    )
    assert_allclose(np.ravel(z), data[:, 2], **allclose_pars)
    assert_allclose(ss, 0.0, **allclose_pars)
    z, ss = uk.execute(
        "grid",
        np.arange(0.5, 10.0, 1.0),
        np.arange(0.5, 10.0, 2.0),
        backend="vectorized",
    )
    assert not np.allclose(np.ravel(z), data[:, 2])
    assert not np.allclose(ss, 0.0)
    z, ss = uk.execute(
        "grid", np.arange(0.0, 10.0, 1.0), np.arange(0.0, 10.0, 2.0), backend="loop"
    )
    assert_allclose(np.ravel(z), data[:, 2], **allclose_pars)
    assert_allclose(ss, 0.0, **allclose_pars)
    z, ss = uk.execute(
        "grid", np.arange(0.5, 10.0, 1.0), np.arange(0.5, 10.0, 2.0), backend="loop"
    )
    assert not np.allclose(np.ravel(z), data[:, 2])
    assert not np.allclose(ss, 0.0)


def test_custom_variogram(sample_data_2d):
    data, (gridx, gridy, gridx_2), mask_ref = sample_data_2d

    def func(params, dist):
        return params[0] * np.log10(dist + params[1]) + params[2]

    with pytest.raises(ValueError):
        UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model="mrow")
    with pytest.raises(ValueError):
        UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model="custom")
    with pytest.raises(ValueError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="custom",
            variogram_function=0,
        )
    with pytest.raises(ValueError):
        UniversalKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="custom",
            variogram_function=func,
        )
    uk = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="custom",
        variogram_parameters=[1.0, 1.0, 1.0],
        variogram_function=func,
    )
    assert uk.variogram_function([1.0, 1.0, 1.0], 1.0) == approx(1.3010, rel=1e-4)
    uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model="linear")
    uk.update_variogram_model(
        "custom", variogram_parameters=[1.0, 1.0, 1.0], variogram_function=func
    )
    assert uk.variogram_function([1.0, 1.0, 1.0], 1.0) == approx(1.3010, rel=1e-4)

    with pytest.raises(ValueError):
        OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model="mrow")
    with pytest.raises(ValueError):
        OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model="custom")
    with pytest.raises(ValueError):
        OrdinaryKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="custom",
            variogram_function=0,
        )
    with pytest.raises(ValueError):
        OrdinaryKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model="custom",
            variogram_function=func,
        )
    ok = OrdinaryKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="custom",
        variogram_parameters=[1.0, 1.0, 1.0],
        variogram_function=func,
    )
    assert ok.variogram_function([1.0, 1.0, 1.0], 1.0) == approx(1.3010, rel=1e-4)
    ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model="linear")
    ok.update_variogram_model(
        "custom", variogram_parameters=[1.0, 1.0, 1.0], variogram_function=func
    )
    assert ok.variogram_function([1.0, 1.0, 1.0], 1.0) == approx(1.3010, rel=1e-4)


def test_ok3d(validation_ref):

    data, (ok_test_answer, gridx_ref, gridy_ref), _ = validation_ref

    # Test to compare K3D results to those obtained using KT3D_H2O.
    # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater, vol. 47,
    # no. 4, 580-586.)
    k3d = OrdinaryKriging3D(
        data[:, 0],
        data[:, 1],
        np.zeros(data[:, 1].shape),
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=[500.0, 3000.0, 0.0],
    )

    with pytest.raises(ValueError):
        OrdinaryKriging3D(
            data[:, 0],
            data[:, 1],
            np.zeros(data[:, 1].shape),
            data[:, 2],
            variogram_model="exponential",
            variogram_parameters=[500.0, 3000.0, 0.0],
            exact_values="blurg",
        )

    ok3d_non_exact = OrdinaryKriging3D(
        data[:, 0],
        data[:, 1],
        np.zeros(data[:, 1].shape),
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=[500.0, 3000.0, 0.0],
        exact_values=False,
    )

    k, ss = k3d.execute(
        "grid", gridx_ref, gridy_ref, np.array([0.0]), backend="vectorized"
    )
    assert_allclose(np.squeeze(k), ok_test_answer)
    k, ss = k3d.execute("grid", gridx_ref, gridy_ref, np.array([0.0]), backend="loop")
    assert_allclose(np.squeeze(k), ok_test_answer)

    # Test to compare K3D results to those obtained using KT3D.
    data = np.genfromtxt(
        os.path.join(BASE_DIR, "test_data", "test3d_data.txt"), skip_header=1
    )
    ans = np.genfromtxt(os.path.join(BASE_DIR, "test_data", "test3d_answer.txt"))
    ans_z = ans[:, 0].reshape((10, 10, 10))
    ans_ss = ans[:, 1].reshape((10, 10, 10))
    k3d = OrdinaryKriging3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        variogram_model="linear",
        variogram_parameters=[1.0, 0.1],
    )
    k, ss = k3d.execute(
        "grid", np.arange(10.0), np.arange(10.0), np.arange(10.0), backend="vectorized"
    )
    assert_allclose(k, ans_z, rtol=1e-3, atol=1e-8)
    assert_allclose(ss, ans_ss, rtol=1e-3, atol=1e-8)
    k3d = OrdinaryKriging3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        variogram_model="linear",
        variogram_parameters=[1.0, 0.1],
    )
    k, ss = k3d.execute(
        "grid", np.arange(10.0), np.arange(10.0), np.arange(10.0), backend="loop"
    )
    assert_allclose(k, ans_z, rtol=1e-3, atol=1e-8)
    assert_allclose(ss, ans_ss, rtol=1e-3, atol=1e-8)


def test_ok3d_moving_window():

    # Test to compare K3D results to those obtained using KT3D.
    data = np.genfromtxt(
        os.path.join(BASE_DIR, "test_data", "test3d_data.txt"), skip_header=1
    )
    ans = np.genfromtxt(os.path.join(BASE_DIR, "./test_data/test3d_answer.txt"))
    ans_z = ans[:, 0].reshape((10, 10, 10))
    ans_ss = ans[:, 1].reshape((10, 10, 10))
    k3d = OrdinaryKriging3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        variogram_model="linear",
        variogram_parameters=[1.0, 0.1],
    )
    k, ss = k3d.execute(
        "grid",
        np.arange(10.0),
        np.arange(10.0),
        np.arange(10.0),
        backend="loop",
        n_closest_points=10,
    )
    assert_allclose(k, ans_z, rtol=1e-3)
    assert_allclose(ss, ans_ss, rtol=1e-3)


def test_ok3d_uk3d_and_backends_produce_same_results(validation_ref):

    data, _, (uk_test_answer, gridx_ref, gridy_ref) = validation_ref

    ok3d = OrdinaryKriging3D(
        data[:, 0],
        data[:, 1],
        np.zeros(data[:, 1].shape),
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=[500.0, 3000.0, 0.0],
    )
    ok_v, oss_v = ok3d.execute(
        "grid", gridx_ref, gridy_ref, np.array([0.0]), backend="vectorized"
    )
    ok_l, oss_l = ok3d.execute(
        "grid", gridx_ref, gridy_ref, np.array([0.0]), backend="loop"
    )

    uk3d = UniversalKriging3D(
        data[:, 0],
        data[:, 1],
        np.zeros(data[:, 1].shape),
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=[500.0, 3000.0, 0.0],
    )
    uk_v, uss_v = uk3d.execute(
        "grid", gridx_ref, gridy_ref, np.array([0.0]), backend="vectorized"
    )
    assert_allclose(uk_v, ok_v)
    uk_l, uss_l = uk3d.execute(
        "grid", gridx_ref, gridy_ref, np.array([0.0]), backend="loop"
    )
    assert_allclose(uk_l, ok_l)
    assert_allclose(uk_l, uk_v)
    assert_allclose(uss_l, uss_v)

    data = np.genfromtxt(
        os.path.join(BASE_DIR, "test_data", "test3d_data.txt"), skip_header=1
    )
    ok3d = OrdinaryKriging3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        variogram_model="linear",
        variogram_parameters=[1.0, 0.1],
    )
    ok_v, oss_v = ok3d.execute(
        "grid", np.arange(10.0), np.arange(10.0), np.arange(10.0), backend="vectorized"
    )
    ok_l, oss_l = ok3d.execute(
        "grid", np.arange(10.0), np.arange(10.0), np.arange(10.0), backend="loop"
    )

    uk3d = UniversalKriging3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        variogram_model="linear",
        variogram_parameters=[1.0, 0.1],
    )
    uk_v, uss_v = uk3d.execute(
        "grid", np.arange(10.0), np.arange(10.0), np.arange(10.0), backend="vectorized"
    )
    assert_allclose(uk_v, ok_v)
    assert_allclose(uss_v, oss_v)
    uk_l, uss_l = uk3d.execute(
        "grid", np.arange(10.0), np.arange(10.0), np.arange(10.0), backend="loop"
    )
    assert_allclose(uk_l, ok_l)
    assert_allclose(uss_l, oss_l)
    assert_allclose(uk_l, uk_v)
    assert_allclose(uss_l, uss_v)


def test_ok3d_update_variogram_model(sample_data_3d):

    data, (gridx_ref, gridy_ref, gridz_ref), mask_ref = sample_data_3d

    with pytest.raises(ValueError):
        OrdinaryKriging3D(
            data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="blurg"
        )

    k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3])
    variogram_model = k3d.variogram_model
    variogram_parameters = k3d.variogram_model_parameters
    anisotropy_scaling_y = k3d.anisotropy_scaling_y
    anisotropy_scaling_z = k3d.anisotropy_scaling_z
    anisotropy_angle_x = k3d.anisotropy_angle_x
    anisotropy_angle_y = k3d.anisotropy_angle_y
    anisotropy_angle_z = k3d.anisotropy_angle_z

    with pytest.raises(ValueError):
        k3d.update_variogram_model("blurg")
    k3d.update_variogram_model(
        "power",
        anisotropy_scaling_y=3.0,
        anisotropy_scaling_z=3.0,
        anisotropy_angle_x=45.0,
        anisotropy_angle_y=45.0,
        anisotropy_angle_z=45.0,
    )
    assert variogram_model != k3d.variogram_model
    assert not np.array_equal(variogram_parameters, k3d.variogram_model_parameters)
    assert anisotropy_scaling_y != k3d.anisotropy_scaling_y
    assert anisotropy_scaling_z != k3d.anisotropy_scaling_z
    assert anisotropy_angle_x != k3d.anisotropy_angle_x
    assert anisotropy_angle_y != k3d.anisotropy_angle_y
    assert anisotropy_angle_z != k3d.anisotropy_angle_z


def test_uk3d_update_variogram_model(sample_data_3d):

    data, (gridx_ref, gridy_ref, gridz_ref), mask_ref = sample_data_3d

    with pytest.raises(ValueError):
        UniversalKriging3D(
            data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="blurg"
        )

    uk3d = UniversalKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3])
    variogram_model = uk3d.variogram_model
    variogram_parameters = uk3d.variogram_model_parameters
    anisotropy_scaling_y = uk3d.anisotropy_scaling_y
    anisotropy_scaling_z = uk3d.anisotropy_scaling_z
    anisotropy_angle_x = uk3d.anisotropy_angle_x
    anisotropy_angle_y = uk3d.anisotropy_angle_y
    anisotropy_angle_z = uk3d.anisotropy_angle_z

    with pytest.raises(ValueError):
        uk3d.update_variogram_model("blurg")
    uk3d.update_variogram_model(
        "power",
        anisotropy_scaling_y=3.0,
        anisotropy_scaling_z=3.0,
        anisotropy_angle_x=45.0,
        anisotropy_angle_y=45.0,
        anisotropy_angle_z=45.0,
    )
    assert not variogram_model == uk3d.variogram_model
    assert not np.array_equal(variogram_parameters, uk3d.variogram_model_parameters)
    assert not anisotropy_scaling_y == uk3d.anisotropy_scaling_y
    assert not anisotropy_scaling_z == uk3d.anisotropy_scaling_z
    assert not anisotropy_angle_x == uk3d.anisotropy_angle_x
    assert not anisotropy_angle_y == uk3d.anisotropy_angle_y
    assert not anisotropy_angle_z == uk3d.anisotropy_angle_z


def test_ok3d_backends_produce_same_result(sample_data_3d):

    data, (gridx_ref, gridy_ref, gridz_ref), mask_ref = sample_data_3d

    k3d = OrdinaryKriging3D(
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="linear"
    )

    ok3d_non_exact = OrdinaryKriging3D(
        data[:, 0],
        data[:, 1],
        np.zeros(data[:, 1].shape),
        data[:, 2],
        variogram_model="exponential",
        variogram_parameters=[500.0, 3000.0, 0.0],
        exact_values=False,
    )

    k_k3d_v, ss_k3d_v = k3d.execute(
        "grid", gridx_ref, gridy_ref, gridz_ref, backend="vectorized"
    )
    k_k3d_l, ss_k3d_l = k3d.execute(
        "grid", gridx_ref, gridy_ref, gridz_ref, backend="loop"
    )
    assert_allclose(k_k3d_v, k_k3d_l, rtol=1e-05, atol=1e-8)
    assert_allclose(ss_k3d_v, ss_k3d_l, rtol=1e-05, atol=1e-8)

    k, ss = ok3d_non_exact.execute(
        "grid", np.arange(10.0), np.arange(10.0), np.arange(10.0), backend="loop"
    )
    k1, ss1 = ok3d_non_exact.execute(
        "grid", np.arange(10.0), np.arange(10.0), np.arange(10.0), backend="vectorized"
    )
    assert_allclose(k1, k)
    assert_allclose(ss1, ss)


def test_ok3d_execute(sample_data_3d):

    data, (gridx_ref, gridy_ref, gridz_ref), mask_ref = sample_data_3d

    k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

    with pytest.raises(ValueError):
        k3d.execute("blurg", gridx_ref, gridy_ref, gridz_ref)

    k, ss = k3d.execute("grid", gridx_ref, gridy_ref, gridz_ref, backend="vectorized")
    shape = (gridz_ref.size, gridy_ref.size, gridx_ref.size)
    assert k.shape == shape
    assert ss.shape == shape
    assert np.amax(k) != np.amin(k)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(k)

    k, ss = k3d.execute("grid", gridx_ref, gridy_ref, gridz_ref, backend="loop")
    shape = (gridz_ref.size, gridy_ref.size, gridx_ref.size)
    assert k.shape == shape
    assert ss.shape == shape
    assert np.amax(k) != np.amin(k)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(k)

    with pytest.raises(IOError):
        k3d.execute("masked", gridx_ref, gridy_ref, gridz_ref, backend="vectorized")
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        k3d.execute(
            "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask, backend="vectorized"
        )
    k, ss = k3d.execute(
        "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask_ref, backend="vectorized"
    )
    assert np.ma.is_masked(k)
    assert np.ma.is_masked(ss)
    assert k[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked
    z, ss = k3d.execute(
        "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask_ref.T, backend="vectorized"
    )
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked

    with pytest.raises(IOError):
        k3d.execute("masked", gridx_ref, gridy_ref, gridz_ref, backend="loop")
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        k3d.execute(
            "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask, backend="loop"
        )
    k, ss = k3d.execute(
        "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask_ref, backend="loop"
    )
    assert np.ma.is_masked(k)
    assert np.ma.is_masked(ss)
    assert k[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked
    z, ss = k3d.execute(
        "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask_ref.T, backend="loop"
    )
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked

    with pytest.raises(ValueError):
        k3d.execute(
            "points",
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([1.0]),
            backend="vectorized",
        )
    k, ss = k3d.execute(
        "points", gridx_ref[0], gridy_ref[0], gridz_ref[0], backend="vectorized"
    )
    assert k.shape == (1,)
    assert ss.shape == (1,)

    with pytest.raises(ValueError):
        k3d.execute(
            "points",
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([1.0]),
            backend="loop",
        )
    k, ss = k3d.execute(
        "points", gridx_ref[0], gridy_ref[0], gridz_ref[0], backend="loop"
    )
    assert k.shape == (1,)
    assert ss.shape == (1,)

    data = np.zeros((125, 4))
    z, y, x = np.meshgrid(
        np.arange(0.0, 5.0, 1.0), np.arange(0.0, 5.0, 1.0), np.arange(0.0, 5.0, 1.0)
    )
    data[:, 0] = np.ravel(x)
    data[:, 1] = np.ravel(y)
    data[:, 2] = np.ravel(z)
    data[:, 3] = np.ravel(z)
    k3d = OrdinaryKriging3D(
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="linear"
    )
    k, ss = k3d.execute(
        "grid",
        np.arange(2.0, 3.0, 0.1),
        np.arange(2.0, 3.0, 0.1),
        np.arange(0.0, 4.0, 1.0),
        backend="vectorized",
    )
    assert_allclose(k[0, :, :], 0.0, atol=0.01)
    assert_allclose(k[1, :, :], 1.0, rtol=1.0e-2)
    assert_allclose(k[2, :, :], 2.0, rtol=1.0e-2)
    assert_allclose(k[3, :, :], 3.0, rtol=1.0e-2)
    k, ss = k3d.execute(
        "grid",
        np.arange(2.0, 3.0, 0.1),
        np.arange(2.0, 3.0, 0.1),
        np.arange(0.0, 4.0, 1.0),
        backend="loop",
    )
    assert_allclose(k[0, :, :], 0.0, atol=0.01)
    assert_allclose(k[1, :, :], 1.0, rtol=1.0e-2)
    assert_allclose(k[2, :, :], 2.0, rtol=1.0e-2)
    assert_allclose(k[3, :, :], 3.0, rtol=1.0e-2)
    k3d = OrdinaryKriging3D(
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="linear"
    )
    k, ss = k3d.execute(
        "points",
        [2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5],
        [1.0, 2.0, 3.0],
        backend="vectorized",
    )
    assert_allclose(k[0], 1.0, atol=0.01)
    assert_allclose(k[1], 2.0, rtol=1.0e-2)
    assert_allclose(k[2], 3.0, rtol=1.0e-2)
    k, ss = k3d.execute(
        "points", [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1.0, 2.0, 3.0], backend="loop"
    )
    assert_allclose(k[0], 1.0, atol=0.01)
    assert_allclose(k[1], 2.0, rtol=1.0e-2)
    assert_allclose(k[2], 3.0, rtol=1.0e-2)


def test_uk3d_execute(sample_data_3d):

    data, (gridx_ref, gridy_ref, gridz_ref), mask_ref = sample_data_3d

    uk3d = UniversalKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

    with pytest.raises(ValueError):
        uk3d.execute("blurg", gridx_ref, gridy_ref, gridz_ref)

    k, ss = uk3d.execute("grid", gridx_ref, gridy_ref, gridz_ref, backend="vectorized")
    shape = (gridz_ref.size, gridy_ref.size, gridx_ref.size)
    assert k.shape == shape
    assert ss.shape == shape
    assert np.amax(k) != np.amin(k)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(k)

    k, ss = uk3d.execute("grid", gridx_ref, gridy_ref, gridz_ref, backend="loop")
    shape = (gridz_ref.size, gridy_ref.size, gridx_ref.size)
    assert k.shape == shape
    assert ss.shape == shape
    assert np.amax(k) != np.amin(k)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(k)

    with pytest.raises(IOError):
        uk3d.execute("masked", gridx_ref, gridy_ref, gridz_ref, backend="vectorized")
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        uk3d.execute(
            "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask, backend="vectorized"
        )
    k, ss = uk3d.execute(
        "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask_ref, backend="vectorized"
    )
    assert np.ma.is_masked(k)
    assert np.ma.is_masked(ss)
    assert k[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked
    z, ss = uk3d.execute(
        "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask_ref.T, backend="vectorized"
    )
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked

    with pytest.raises(IOError):
        uk3d.execute("masked", gridx_ref, gridy_ref, gridz_ref, backend="loop")
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        uk3d.execute(
            "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask, backend="loop"
        )
    k, ss = uk3d.execute(
        "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask_ref, backend="loop"
    )
    assert np.ma.is_masked(k)
    assert np.ma.is_masked(ss)
    assert k[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked
    z, ss = uk3d.execute(
        "masked", gridx_ref, gridy_ref, gridz_ref, mask=mask_ref.T, backend="loop"
    )
    assert np.ma.is_masked(z)
    assert np.ma.is_masked(ss)
    assert z[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked

    with pytest.raises(ValueError):
        uk3d.execute(
            "points",
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([1.0]),
            backend="vectorized",
        )
    k, ss = uk3d.execute(
        "points", gridx_ref[0], gridy_ref[0], gridz_ref[0], backend="vectorized"
    )
    assert k.shape == (1,)
    assert ss.shape == (1,)

    with pytest.raises(ValueError):
        uk3d.execute(
            "points",
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([1.0]),
            backend="loop",
        )
    k, ss = uk3d.execute(
        "points", gridx_ref[0], gridy_ref[0], gridz_ref[0], backend="loop"
    )
    assert k.shape == (1,)
    assert ss.shape == (1,)

    data = np.zeros((125, 4))
    z, y, x = np.meshgrid(
        np.arange(0.0, 5.0, 1.0), np.arange(0.0, 5.0, 1.0), np.arange(0.0, 5.0, 1.0)
    )
    data[:, 0] = np.ravel(x)
    data[:, 1] = np.ravel(y)
    data[:, 2] = np.ravel(z)
    data[:, 3] = np.ravel(z)
    k3d = UniversalKriging3D(
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="linear"
    )
    k, ss = k3d.execute(
        "grid",
        np.arange(2.0, 3.0, 0.1),
        np.arange(2.0, 3.0, 0.1),
        np.arange(0.0, 4.0, 1.0),
        backend="vectorized",
    )
    assert_allclose(k[0, :, :], 0.0, atol=0.01)
    assert_allclose(k[1, :, :], 1.0, rtol=1.0e-2)
    assert_allclose(k[2, :, :], 2.0, rtol=1.0e-2)
    assert_allclose(k[3, :, :], 3.0, rtol=1.0e-2)
    k, ss = k3d.execute(
        "grid",
        np.arange(2.0, 3.0, 0.1),
        np.arange(2.0, 3.0, 0.1),
        np.arange(0.0, 4.0, 1.0),
        backend="loop",
    )
    assert_allclose(k[0, :, :], 0.0, atol=0.01)
    assert_allclose(k[1, :, :], 1.0, rtol=1.0e-2)
    assert_allclose(k[2, :, :], 2.0, rtol=1.0e-2)
    assert_allclose(k[3, :, :], 3.0, rtol=1.0e-2)
    k3d = UniversalKriging3D(
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="linear"
    )
    k, ss = k3d.execute(
        "points",
        [2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5],
        [1.0, 2.0, 3.0],
        backend="vectorized",
    )
    assert_allclose(k[0], 1.0, atol=0.01)
    assert_allclose(k[1], 2.0, rtol=1.0e-2)
    assert_allclose(k[2], 3.0, rtol=1.0e-2)
    k, ss = k3d.execute(
        "points", [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1.0, 2.0, 3.0], backend="loop"
    )
    assert_allclose(k[0], 1.0, atol=0.01)
    assert_allclose(k[1], 2.0, rtol=1.0e-2)
    assert_allclose(k[2], 3.0, rtol=1.0e-2)


def test_force_exact_3d(sample_data_3d):

    data, (gridx_ref, gridy_ref, gridz_ref), mask_ref = sample_data_3d

    k3d = OrdinaryKriging3D(
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="linear"
    )
    k, ss = k3d.execute(
        "grid", [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], backend="vectorized"
    )
    assert k[2, 0, 0] == approx(0.9)
    assert ss[2, 0, 0] == approx(0.0)
    assert k[0, 2, 0] == approx(0.9)
    assert ss[0, 2, 0] == approx(0.0)
    assert k[1, 2, 2] == approx(0.7)
    assert ss[1, 2, 2] == approx(0.0)
    assert ss[2, 2, 2] != approx(0.0)
    assert ss[0, 0, 0] != approx(0.0)

    k, ss = k3d.execute(
        "grid", [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], backend="loop"
    )
    assert k[2, 0, 0] == approx(0.9)
    assert ss[2, 0, 0] == approx(0.0)
    assert k[0, 2, 0] == approx(0.9)
    assert ss[0, 2, 0] == approx(0.0)
    assert k[1, 2, 2] == approx(0.7)
    assert ss[1, 2, 2] == approx(0.0)
    assert ss[2, 2, 2] != approx(0.0)
    assert ss[0, 0, 0] != approx(0.0)

    k3d = UniversalKriging3D(
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="linear"
    )
    k, ss = k3d.execute(
        "grid", [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], backend="vectorized"
    )
    assert k[2, 0, 0] == approx(0.9)
    assert ss[2, 0, 0] == approx(0.0)
    assert k[0, 2, 0] == approx(0.9)
    assert ss[0, 2, 0] == approx(0.0)
    assert k[1, 2, 2] == approx(0.7)
    assert ss[1, 2, 2] == approx(0.0)
    assert ss[2, 2, 2] != approx(0.0)
    assert ss[0, 0, 0] != approx(0.0)

    k, ss = k3d.execute(
        "grid", [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], backend="loop"
    )
    assert k[2, 0, 0] == approx(0.9)
    assert ss[2, 0, 0] == approx(0.0)
    assert k[0, 2, 0] == approx(0.9)
    assert ss[0, 2, 0] == approx(0.0)
    assert k[1, 2, 2] == approx(0.7)
    assert ss[1, 2, 2] == approx(0.0)
    assert ss[2, 2, 2] != approx(0.0)
    assert ss[0, 0, 0] != approx(0.0)


def test_uk3d_specified_drift(sample_data_3d):

    data, (gridx_ref, gridy_ref, gridz_ref), mask_ref = sample_data_3d

    zg, yg, xg = np.meshgrid(gridz_ref, gridy_ref, gridx_ref, indexing="ij")

    with pytest.raises(ValueError):
        UniversalKriging3D(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            data[:, 3],
            variogram_model="linear",
            drift_terms=["specified"],
        )
    with pytest.raises(TypeError):
        UniversalKriging3D(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            data[:, 3],
            variogram_model="linear",
            drift_terms=["specified"],
            specified_drift=data[:, 0],
        )
    with pytest.raises(ValueError):
        UniversalKriging3D(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            data[:, 3],
            variogram_model="linear",
            drift_terms=["specified"],
            specified_drift=[data[:2, 0]],
        )

    uk_spec = UniversalKriging3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        variogram_model="linear",
        drift_terms=["specified"],
        specified_drift=[data[:, 0], data[:, 1], data[:, 2]],
    )
    with pytest.raises(ValueError):
        uk_spec.execute(
            "grid",
            gridx_ref,
            gridy_ref,
            gridz_ref,
            specified_drift_arrays=[gridx_ref, gridy_ref, gridz_ref],
        )
    with pytest.raises(TypeError):
        uk_spec.execute(
            "grid", gridx_ref, gridy_ref, gridz_ref, specified_drift_arrays=gridx_ref
        )
    with pytest.raises(ValueError):
        uk_spec.execute(
            "grid", gridx_ref, gridy_ref, gridz_ref, specified_drift_arrays=[zg]
        )
    z_spec, ss_spec = uk_spec.execute(
        "grid", gridx_ref, gridy_ref, gridz_ref, specified_drift_arrays=[xg, yg, zg]
    )

    uk_lin = UniversalKriging3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        variogram_model="linear",
        drift_terms=["regional_linear"],
    )
    z_lin, ss_lin = uk_lin.execute("grid", gridx_ref, gridy_ref, gridz_ref)

    assert_allclose(z_spec, z_lin)
    assert_allclose(ss_spec, ss_lin)


def test_uk3d_functional_drift(sample_data_3d):

    data, (gridx, gridy, gridz), mask_ref = sample_data_3d

    func_x = lambda x, y, z: x  # noqa
    func_y = lambda x, y, z: y  # noqa
    func_z = lambda x, y, z: z  # noqa

    with pytest.raises(ValueError):
        UniversalKriging3D(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            data[:, 3],
            variogram_model="linear",
            drift_terms=["functional"],
        )
    with pytest.raises(TypeError):
        UniversalKriging3D(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            data[:, 3],
            variogram_model="linear",
            drift_terms=["functional"],
            functional_drift=func_x,
        )

    uk_func = UniversalKriging3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        variogram_model="linear",
        drift_terms=["functional"],
        functional_drift=[func_x, func_y, func_z],
    )
    z_func, ss_func = uk_func.execute("grid", gridx, gridy, gridz)
    uk_lin = UniversalKriging3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        variogram_model="linear",
        drift_terms=["regional_linear"],
    )
    z_lin, ss_lin = uk_lin.execute("grid", gridx, gridy, gridz)
    assert_allclose(z_func, z_lin)
    assert_allclose(ss_func, ss_lin)


def test_geometric_code():

    # Create selected points distributed across the sphere:
    N = 4
    lon = np.array([7.0, 7.0, 187.0, 73.231])
    lat = np.array([13.23, 13.2301, -13.23, -79.3])

    # For the points generated with this reference seed, the distance matrix
    # has been calculated using geopy (v. 1.11.0) as follows:
    # >>>   from geopy.distance import great_circle
    # >>>   g = great_circle(radius=1.0)
    # >>>   d = np.zeros((N,N), dtype=float)
    # >>>   for i in range(N):
    # >>>       for j in range(N):
    # >>>           d[i,j] = g.measure((lat[i],lon[i]),(lat[j],lon[j]))
    # >>>   d *= 180.0/np.pi
    # From that distance matrix, the reference values have been obtained.
    d_ref = np.array(
        [
            [0.0, 1e-4, 180.0, 98.744848317171801],
            [1e-4, 0.0, 179.9999, 98.744946828324345],
            [180.0, 179.9999, 0.0, 81.255151682828213],
            [98.744848317171801, 98.744946828324345, 81.255151682828213, 0.0],
        ]
    )

    # Calculate distance matrix using the PyKrige code:
    d = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d[i, j] = core.great_circle_distance(lon[i], lat[i], lon[j], lat[j])

    # Test agains reference values:
    assert_allclose(d, d_ref)

    # Test general features:
    assert_allclose(d[np.eye(N, dtype=bool)], 0.0)
    np.testing.assert_equal(d >= 0.0, np.ones((N, N), dtype=bool))
    assert_allclose(d, d.T)
    np.testing.assert_equal(d <= 180.0, np.ones((N, N), dtype=bool))

    # Test great_circle_distance and euclid3_to_great_circle against each other
    lon_ref = lon
    lat_ref = lat
    for i in range(len(lon_ref)):
        lon, lat = np.meshgrid(np.linspace(0, 360.0, 20), np.linspace(-90.0, 90.0, 20))
        dx = np.cos(np.pi / 180.0 * lon) * np.cos(np.pi / 180.0 * lat) - np.cos(
            np.pi / 180.0 * lon_ref[i]
        ) * np.cos(np.pi / 180.0 * lat_ref[i])
        dy = np.sin(np.pi / 180.0 * lon) * np.cos(np.pi / 180.0 * lat) - np.sin(
            np.pi / 180.0 * lon_ref[i]
        ) * np.cos(np.pi / 180.0 * lat_ref[i])
        dz = np.sin(np.pi / 180.0 * lat) - np.sin(np.pi / 180.0 * lat_ref[i])
        assert_allclose(
            core.great_circle_distance(lon_ref[i], lat_ref[i], lon, lat),
            core.euclid3_to_great_circle(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)),
            rtol=1e-5,
        )


def test_ok_geographic():
    # Generate random data:
    np.random.seed(89239413)
    lon = 360.0 * np.random.rand(50, 1)
    lat = 180.0 * np.random.rand(50, 1) - 90.0
    z = np.random.rand(50, 1)

    # Generate grid:
    grid_lon = 360.0 * np.random.rand(120, 1)
    grid_lat = 180.0 * np.random.rand(120, 1) - 90.0

    # Create ordinary kriging object:
    OK = OrdinaryKriging(
        lon,
        lat,
        z,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic",
    )

    # Execute on grid:
    z, ss = OK.execute("grid", grid_lon, grid_lat)


def test_ok_geographic_vs_euclid():
    # Generate some random data close to the north pole.
    # Then we use a polar projected 2d euclidean coordinate
    # system and compare the kriging results in that coordinate
    # system with the geographic-option results.
    # If data point distance to the north pole is small enough
    # (choose maximum 0.01 degrees), the differences due to curvature
    # should be negligible.
    np.random.seed(89239413)
    from_north = 1e-2 * np.random.random(5)
    lat = 90.0 - from_north
    lon = 360.0 * np.random.random(5)
    z = np.random.random(5)
    z -= z.mean()
    x = from_north * np.cos(np.deg2rad(lon))
    y = from_north * np.sin(np.deg2rad(lon))

    # Generate grids:
    grid_lon = 360.0 * np.linspace(0, 1, 50)
    grid_from_north = np.linspace(0, 0.01, 10)
    grid_lat = 90.0 - grid_from_north
    grid_x = grid_from_north[:, np.newaxis] * np.cos(
        np.deg2rad(grid_lon[np.newaxis, :])
    )
    grid_y = grid_from_north[:, np.newaxis] * np.sin(
        np.deg2rad(grid_lon[np.newaxis, :])
    )
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat, indexing="xy")

    # Flatten the grids:
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    grid_lon = grid_lon.flatten()
    grid_lat = grid_lat.flatten()

    # Calculate and compare distance matrices ensuring that that part
    # of the workflow works as intended (tested: 2e-9 is currently the
    # match for this setup):
    d_eucl = cdist(
        np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1),
        np.concatenate([grid_x[:, np.newaxis], grid_y[:, np.newaxis]], axis=1),
    )
    d_geo = core.great_circle_distance(
        lon[:, np.newaxis],
        lat[:, np.newaxis],
        grid_lon[np.newaxis, :],
        grid_lat[np.newaxis, :],
    )
    assert_allclose(d_eucl, d_geo, rtol=2e-9)

    # Create ordinary kriging objects:
    OK_geo = OrdinaryKriging(
        lon,
        lat,
        z,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic",
    )
    OK_xy = OrdinaryKriging(
        x, y, z, variogram_model="linear", verbose=False, enable_plotting=False
    )
    OK_wrong = OrdinaryKriging(
        lon, lat, z, variogram_model="linear", verbose=False, enable_plotting=False
    )

    # Execute on grid:
    zgeo, ss = OK_geo.execute("points", grid_lon, grid_lat)
    zxy, ss = OK_xy.execute("points", grid_x, grid_y)
    zwrong, ss = OK_wrong.execute("points", grid_lon, grid_lat)

    # Assert equivalence / difference (tested: 2e-5 is currently the
    # match for this setup):
    assert_allclose(zgeo, zxy, rtol=2e-5)
    assert not np.any(zgeo == 0)
    assert np.abs((zgeo - zwrong) / zgeo).max() > 1.0


def test_ok_geometric_closest_points():
    # Generate random data:
    np.random.seed(89239413)
    lon = 360.0 * np.random.rand(50, 1)
    lat = 180.0 * np.random.rand(50, 1) - 90.0
    z = np.random.rand(50, 1)

    # Generate grid:
    grid_lon = 360.0 * np.random.rand(120, 1)
    grid_lat = 180.0 * np.random.rand(120, 1) - 90.0

    # Create ordinary kriging object:
    OK = OrdinaryKriging(
        lon,
        lat,
        z,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic",
    )

    # Execute on grid:
    with pytest.raises(ValueError):
        # Test OK raising ValueError when closest_points == 1:
        z, ss = OK.execute("grid", grid_lon, grid_lat, n_closest_points=1, backend="C")
    z, ss = OK.execute("grid", grid_lon, grid_lat, n_closest_points=5, backend="C")


@pytest.mark.parametrize("model", [OrdinaryKriging, UniversalKriging])
def test_gstools_variogram(model):
    gstools = pytest.importorskip("gstools")
    # test data
    data = np.array(
        [
            [0.3, 1.2, 0.47],
            [1.9, 0.6, 0.56],
            [1.1, 3.2, 0.74],
            [3.3, 4.4, 1.47],
            [4.7, 3.8, 1.74],
        ]
    )
    gridx = np.arange(0.0, 5.5, 0.1)
    gridy = np.arange(0.0, 6.5, 0.1)
    # a GSTools based covariance model
    cov_model = gstools.Gaussian(
        dim=2, len_scale=1, anis=0.2, angles=-0.5, var=0.5, nugget=0.1
    )
    # create the krige field
    krige = model(data[:, 0], data[:, 1], data[:, 2], cov_model)
    z1, ss1 = krige.execute("grid", gridx, gridy)
    # check if the field coincides with the data
    for i in range(5):
        y_id = int(data[i, 1] * 10)
        x_id = int(data[i, 0] * 10)
        assert np.isclose(z1[y_id, x_id], data[i, 2])


@pytest.mark.parametrize("model", [OrdinaryKriging, UniversalKriging])
def test_pseudo_2d(model):
    # test data
    data = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [1.0, 0.0, 6.0]])
    for p_type in ["pinv", "pinv2", "pinvh"]:
        # create the krige field
        krige = model(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_parameters=[1.0, 0.0],
            pseudo_inv=True,
            pseudo_inv_type=p_type,
        )
        z1, ss1 = krige.execute("points", 0.0, 0.0)
        # check if the field coincides with the mean of the redundant data
        assert np.isclose(z1.item(), 2.0)


@pytest.mark.parametrize("model", [OrdinaryKriging3D, UniversalKriging3D])
def test_pseudo_3d(model):
    # test data
    data = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 3.0], [1.0, 0.0, 0.0, 6.0]])
    for p_type in ["pinv", "pinv2", "pinvh"]:
        # create the krige field
        krige = model(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            data[:, 3],
            variogram_parameters=[1.0, 0.0],
            pseudo_inv=True,
            pseudo_inv_type=p_type,
        )
        z1, ss1 = krige.execute("points", 0.0, 0.0, 0.0)
        # check if the field coincides with the mean of the redundant data
        assert np.isclose(z1.item(), 2.0)
