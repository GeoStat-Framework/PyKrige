# coding: utf-8
"""
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Methods for reading/writing ASCII grid files.

Copyright (c) 2015-2020, PyKrige Developers
"""
import numpy as np
import warnings
import io
import datetime
import os


def write_asc_grid(x, y, z, filename="output.asc", no_data=-999.0, style=1):
    r"""Writes gridded data to ASCII grid file (\*.asc).

    This is useful for exporting data to a GIS program.

    Parameters
    ----------
    x : array_like, shape (N,) or (N, 1)
        X-coordinates of grid points at center of cells.
    y : array_like, shape (M,) or (M, 1)
        Y-coordinates of grid points at center of cells.
    z : array_like, shape (M, N)
        Gridded data values. May be a masked array.
    filename : string, optional
        Name of output \*.asc file. Default name is 'output.asc'.
    no_data : float, optional
        no data value to be used
    style : int, optional
        Determines how to write the \*.asc file header.
        Specifying 1 writes out DX, DY, XLLCENTER, YLLCENTER.
        Specifying 2 writes out CELLSIZE (note DX must be the same as DY),
        XLLCORNER, YLLCORNER. Default is 1.
    """

    if np.ma.is_masked(z):
        z = np.array(z.tolist(no_data))

    x = np.squeeze(np.array(x))
    y = np.squeeze(np.array(y))
    z = np.squeeze(np.array(z))
    nrows = z.shape[0]
    ncols = z.shape[1]

    if z.ndim != 2:
        raise ValueError("Two-dimensional grid is required to write *.asc grid.")
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError(
            "Dimensions of X and/or Y coordinate arrays are not "
            "as expected. Could not write *.asc grid."
        )
    if z.shape != (y.size, x.size):
        warnings.warn(
            "Grid dimensions are not as expected. "
            "Incorrect *.asc file generation may result.",
            RuntimeWarning,
        )
    if np.amin(x) != x[0] or np.amin(y) != y[0]:
        warnings.warn(
            "Order of X or Y coordinates is not as expected. "
            "Incorrect *.asc file generation may result.",
            RuntimeWarning,
        )

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    if not np.isclose(abs((x[-1] - x[0]) / (x.shape[0] - 1)), dx) or not np.isclose(
        abs((y[-1] - y[0]) / (y.shape[0] - 1)), dy
    ):
        raise ValueError(
            "X or Y spacing is not constant; *.asc grid cannot be written."
        )
    cellsize = -1
    if style == 2:
        if dx != dy:
            raise ValueError(
                "X and Y spacing is not the same. "
                "Cannot write *.asc file in the specified format."
            )
        cellsize = dx

    xllcenter = x[0]
    yllcenter = y[0]

    # Note that these values are flagged as -1. If there is a problem in trying
    # to write out style 2, the -1 value will appear in the output file.
    xllcorner = -1
    yllcorner = -1
    if style == 2:
        xllcorner = xllcenter - dx / 2.0
        yllcorner = yllcenter - dy / 2.0

    with io.open(filename, "w") as f:
        if style == 1:
            f.write("NCOLS          " + "{:<10n}".format(ncols) + "\n")
            f.write("NROWS          " + "{:<10n}".format(nrows) + "\n")
            f.write("XLLCENTER      " + "{:<10.2f}".format(xllcenter) + "\n")
            f.write("YLLCENTER      " + "{:<10.2f}".format(yllcenter) + "\n")
            f.write("DX             " + "{:<10.2f}".format(dx) + "\n")
            f.write("DY             " + "{:<10.2f}".format(dy) + "\n")
            f.write("NODATA_VALUE   " + "{:<10.2f}".format(no_data) + "\n")
        elif style == 2:
            f.write("NCOLS          " + "{:<10n}".format(ncols) + "\n")
            f.write("NROWS          " + "{:<10n}".format(nrows) + "\n")
            f.write("XLLCORNER      " + "{:<10.2f}".format(xllcorner) + "\n")
            f.write("YLLCORNER      " + "{:<10.2f}".format(yllcorner) + "\n")
            f.write("CELLSIZE       " + "{:<10.2f}".format(cellsize) + "\n")
            f.write("NODATA_VALUE   " + "{:<10.2f}".format(no_data) + "\n")
        else:
            raise ValueError("style kwarg must be either 1 or 2.")

        for m in range(z.shape[0] - 1, -1, -1):
            for n in range(z.shape[1]):
                f.write("{:<16.2f}".format(z[m, n]))
            if m != 0:
                f.write("\n")


def read_asc_grid(filename, footer=0):
    r"""Reads ASCII grid file (\*.asc).

    Parameters
    ----------
    filename : str
        Name of \*.asc file.
    footer : int, optional
        Number of lines at bottom of \*.asc file to skip.

    Returns
    -------
    grid_array : numpy array, shape (M, N)
        (M, N) array of grid values, where M is number of Y-coordinates and
        N is number of X-coordinates. The array entry corresponding to
        the lower-left coordinates is at index [M, 0], so that
        the array is oriented as it would be in X-Y space.
    x : numpy array, shape (N,)
        1D array of N X-coordinates.
    y : numpy array, shape (M,)
        1D array of M Y-coordinates.
    CELLSIZE : tuple or float
        Either a two-tuple of (x-cell size, y-cell size),
        or a float that specifies the uniform cell size.
    NODATA : float
        Value that specifies which entries are not actual data.
    """

    ncols = None
    nrows = None
    xllcorner = None
    xllcenter = None
    yllcorner = None
    yllcenter = None
    cellsize = None
    dx = None
    dy = None
    no_data = None
    header_lines = 0
    with io.open(filename, "r") as f:
        while True:
            string, value = f.readline().split()
            header_lines += 1
            if string.lower() == "ncols":
                ncols = int(value)
            elif string.lower() == "nrows":
                nrows = int(value)
            elif string.lower() == "xllcorner":
                xllcorner = float(value)
            elif string.lower() == "xllcenter":
                xllcenter = float(value)
            elif string.lower() == "yllcorner":
                yllcorner = float(value)
            elif string.lower() == "yllcenter":
                yllcenter = float(value)
            elif string.lower() == "cellsize":
                cellsize = float(value)
            elif string.lower() == "cell_size":
                cellsize = float(value)
            elif string.lower() == "dx":
                dx = float(value)
            elif string.lower() == "dy":
                dy = float(value)
            elif string.lower() == "nodata_value":
                no_data = float(value)
            elif string.lower() == "nodatavalue":
                no_data = float(value)
            else:
                raise IOError("could not read *.asc file. Error in header.")

            if (
                (ncols is not None)
                and (nrows is not None)
                and (
                    ((xllcorner is not None) and (yllcorner is not None))
                    or ((xllcenter is not None) and (yllcenter is not None))
                )
                and ((cellsize is not None) or ((dx is not None) and (dy is not None)))
                and (no_data is not None)
            ):
                break

    raw_grid_array = np.genfromtxt(
        filename, skip_header=header_lines, skip_footer=footer
    )
    grid_array = np.flipud(raw_grid_array)

    if nrows != grid_array.shape[0] or ncols != grid_array.shape[1]:
        raise IOError(
            "Error reading *.asc file. Encountered problem "
            "with header: NCOLS and/or NROWS does not match "
            "number of columns/rows in data file body."
        )

    if xllcorner is not None and yllcorner is not None:
        if dx is not None and dy is not None:
            xllcenter = xllcorner + dx / 2.0
            yllcenter = yllcorner + dy / 2.0
        else:
            xllcenter = xllcorner + cellsize / 2.0
            yllcenter = yllcorner + cellsize / 2.0

    if dx is not None and dy is not None:
        x = np.arange(xllcenter, xllcenter + ncols * dx, dx)
        y = np.arange(yllcenter, yllcenter + nrows * dy, dy)
    else:
        x = np.arange(xllcenter, xllcenter + ncols * cellsize, cellsize)
        y = np.arange(yllcenter, yllcenter + nrows * cellsize, cellsize)

    # Sometimes x and y and can be an entry too long due to imprecision
    # in calculating the upper cutoff for np.arange(); this bit takes care of
    # that potential problem.
    if x.size == ncols + 1:
        x = x[:-1]
    if y.size == nrows + 1:
        y = y[:-1]

    if cellsize is None:
        cellsize = (dx, dy)

    return grid_array, x, y, cellsize, no_data


def write_zmap_grid(
    x, y, z, filename="output.zmap", no_data=-999.0, coord_sys="<null>"
):
    r"""Writes gridded data to ASCII grid file in zmap format (\*.zmap).

    This is useful for exporting data to a GIS program, or Petrel
    https://gdal.org/drivers/raster/zmap.html

    Parameters
    ----------
    x : array_like, shape (N,) or (N, 1)
        X-coordinates of grid points at center of cells.
    y : array_like, shape (M,) or (M, 1)
        Y-coordinates of grid points at center of cells.
    z : array_like, shape (M, N)
        Gridded data values. May be a masked array.
    filename : string, optional
        Name of output \*.zmap file. Default name is 'output.zmap'.
    no_data : float, optional
        no data value to be used
    coord_sys : String, optional
        coordinate sytem description
    """

    nodes_per_line = 5
    field_width = 15

    if np.ma.is_masked(z):
        z = np.array(z.tolist(no_data))

    x = np.squeeze(np.array(x))
    y = np.squeeze(np.array(y))
    z = np.squeeze(np.array(z))
    nx = len(x)
    ny = len(y)

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])

    if not np.isclose(abs((x[-1] - x[0]) / (x.shape[0] - 1)), dx) or not np.isclose(
        abs((y[-1] - y[0]) / (y.shape[0] - 1)), dy
    ):
        raise ValueError(
            "X or Y spacing is not constant; *.asc grid cannot be written."
        )

    xllcenter = x[0]
    yllcenter = y[0]

    hix = xllcenter + (nx - 1) * dx
    hiy = yllcenter + (ny - 1) * dy

    now = datetime.datetime.now()

    with io.open(filename, "w") as f:
        f.write("!" + "\n")
        f.write("!     ZIMS FILE NAME :  " + os.path.basename(filename) + "\n")
        f.write(
            "!     FORMATTED FILE CREATION DATE: " + now.strftime("%d/%m/%Y") + "\n"
        )
        f.write(
            "!     FORMATTED FILE CREATION TIME: " + now.strftime("%H:%M:%S") + "\n"
        )
        f.write("!     COORDINATE REFERENCE SYSTEM: " + coord_sys + "\n")
        f.write("!" + "\n")
        f.write("@Grid HEADER, GRID, " + str(nodes_per_line) + "\n")
        f.write(" " + str(field_width) + ", " + str(no_data) + ",  , 1 , 1" + "\n")
        f.write(
            "   "
            + str(ny)
            + ",  "
            + str(nx)
            + ",  "
            + str(xllcenter)
            + ",  "
            + str(hix)
            + ",  "
            + str(yllcenter)
            + ",  "
            + str(hiy)
            + "\n"
        )
        f.write("   " + str(dx) + ",  0.0,  0.0    " + "\n")
        f.write("@" + "\n")

        for n in range(z.shape[1]):
            count = 0
            for m in range(z.shape[0] - 1, -1, -1):
                count += 1
                if np.isnan(z[m, n]):
                    f.write(space_back_to_front(format(no_data, "13.7E") + "  "))
                else:
                    if abs(z[m, n]) >= 1e100:  # one tailing space less
                        f.write(space_back_to_front(format(z[m, n], "13.7E") + " "))
                    elif abs(z[m, n]) >= 1e6:
                        f.write(space_back_to_front(format(z[m, n], "13.7E") + "  "))
                    else:
                        f.write(space_back_to_front("{:<13.4f}".format(z[m, n]) + "  "))
                if count % nodes_per_line == 0 or m == 0:
                    f.write("\n")


def read_zmap_grid(filename):
    r"""Reads ASCII grid file in zmap format (\*.zmap).
    https://gdal.org/drivers/raster/zmap.html

    Parameters
    ----------
    filename : str
        Name of \*.zmap file.

    Returns
    -------
    grid_array : numpy array, shape (M, N)
        (M, N) array of grid values, where M is number of Y-coordinates and
        N is number of X-coordinates. The array entry corresponding to
        the lower-left coordinates is at index [M, 0], so that
        the array is oriented as it would be in X-Y space.
    x : numpy array, shape (N,)
        1D array of N X-coordinates.
    y : numpy array, shape (M,)
        1D array of M Y-coordinates.
    cellsize : tuple or float
        Either a two-tuple of (x-cell size, y-cell size),
        or a float that specifies the uniform cell size.
    no_data_value : float
        Value that specifies which entries are not actual data.
    coord_sys : String
        Coordinate system name
    """

    no_data_value, nx, ny, originx, originy, maxx, maxy, dx, dy = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    data_values = np.empty(1)
    coord_sys = "<null>"

    i_header_line, i_value = 0, 0
    with io.open(filename, "r") as f:
        while True:
            line = f.readline()
            if line.startswith("!"):
                line_strings = line.split(":")
                if line_strings[0].__contains__("COORDINATE REFERENCE SYSTEM"):
                    coord_sys = line_strings[1].replace("\n", "")
            else:
                line_strings = line.split()
                line_strings = [string.replace(",", "") for string in line_strings]

            if len(line_strings) == 0:
                break

            if i_header_line == -1 and not line_strings[0].startswith("!"):
                for i_string in range(len(line_strings)):
                    data_values[i_value] = float(line_strings[i_string])
                    i_value += 1

            if line_strings[0].startswith("@"):
                if i_header_line == 0:
                    i_header_line += 1
                else:
                    i_header_line = -1

            if i_header_line > 0:
                if i_header_line == 2:
                    no_data_value = float(line_strings[1])
                elif i_header_line == 3:
                    ny = int(line_strings[0])
                    nx = int(line_strings[1])
                    originx = float(line_strings[2])
                    maxx = float(line_strings[3])
                    originy = float(line_strings[4])
                    maxy = float(line_strings[5])
                    data_values = np.empty(ny * nx)
                i_header_line += 1

    if nx * ny != len(data_values):
        raise IOError(
            "Error reading *.zmap file. Encountered problem "
            "with header: (nx * ny) does not match with the "
            "number items in data file body."
        )

    z = np.empty([ny, nx])
    i_value = 0
    for n in range(z.shape[1]):
        for m in range(z.shape[0] - 1, -1, -1):
            z[m, n] = data_values[i_value]
            i_value += 1

    dx = (maxx - originx) / (nx - 1)
    dy = (maxy - originy) / (ny - 1)

    gridx = np.arange(originx, originx + nx * dx, dx)
    gridy = np.arange(originy, originy + ny * dy, dy)

    cellsize = (dx, dy)

    return z, gridx, gridy, cellsize, no_data_value, coord_sys


def space_back_to_front(string):
    net = string.replace(" ", "")
    return "".join(string.rsplit(net)) + net
