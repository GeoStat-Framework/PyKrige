__doc__ = """Code by Benjamin S. Murphy
bscott.murphy@gmail.com

Dependencies:
    numpy

Callable Methods:
    write_asc_grid(X, Y, Z, filename='output.asc', style=1): Writes an MxN data grid
        to an ASCII grid file (.*asc).
        Inputs:
            X (array-like, dim Nx1): X-coordinates of grid points at center
                of cells.
            Y (array-like, dim Mx1): Y-coordinates of grid points at center
                of cells.
            Z (array-like, dim MxN): Gridded data values. May be a masked array.
            filename (string, optional): Name of output *.asc file.
            style (int, optional): Determines how to write the *.asc file
                header. Specifying 1 writes out DX, DY, XLLCENTER, YLLCENTER.
                Specifying 2 writes out CELLSIZE (note DX must be the same
                as DY), XLLCORNER, YLLCORNER. Default is 1.

    read_asc_grid(filename, footer=0): Reads ASCII grid file (*.asc).
        Inputs:
            filename (string): Name of *.asc file.
            footer (int, optional): Number of lines at bottom of *.asc file to skip.
        Outputs:
            grid_array (numpy array): MxN array of grid values,
                where M is number of Y-coordinates and N is number
                of X-coordinates. The array entry corresponding to
                the lower-left coordinates is at index [M, 0], so that
                the array is oriented as it would be in X-Y space.
            x (numpy array): 1D array of N X-coordinates.
            y (numpy array): 1D array of M Y-coordinates.
            CELLSIZE (tuple or float): Either a two-tuple of (x-cell size,
                y-cell size), or a float that specifies the uniform cell size.
            NODATA (float): Value that specifies which entries are not
                actual data.

Copyright (c) 2015 Benjamin S. Murphy
"""

import numpy as np


def write_asc_grid(x, y, z, filename='output.asc', style=1):
    """Writes gridded data to ASCII grid file (*.asc)"""

    if np.ma.is_masked(z):
        z = np.array(z.tolist(-999.))

    x = np.squeeze(np.array(x))
    y = np.squeeze(np.array(y))
    z = np.squeeze(np.array(z))
    nrows = z.shape[0]
    ncols = z.shape[1]

    if z.ndim != 2:
        raise ValueError("Two-dimensional grid is required to write *.asc grid.")
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("Dimensions of X and/or Y coordinate arrays are not as "
                         "expected. Could not write *.asc grid.")
    if z.shape != (y.size, x.size):
        print "WARNING: Grid dimensions are not as expected. " \
              "Incorrect *.asc file generation may result."
    if np.amin(x) != x[0] or np.amin(y) != y[0]:
        print "WARNING: Order of X or Y coordinates is not as expected. " \
              "Incorrect *.asc file generation may result."

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    if abs((x[-1] - x[0])/(x.shape[0] - 1)) != dx or \
       abs((y[-1] - y[0])/(y.shape[0] - 1)) != dy:
        raise ValueError("X or Y spacing is not constant; *.asc grid cannot "
                         "be written.")
    cellsize = -1
    if style == 2:
        if dx != dy:
            raise ValueError("X and Y spacing is not the same. Cannot write "
                             "*.asc file in the specified format.")
        cellsize = dx

    xllcenter = x[0]
    yllcenter = y[0]

    xllcorner = -1   # Note that these values are flagged as -1. If there is a problem in trying
    yllcorner = -1   # to write out style 2, the -1 value will appear in the output file.
    if style == 2:
        xllcorner = xllcenter - dx/2.0
        yllcorner = yllcenter - dy/2.0

    no_data = -999.

    with open(filename, 'w') as f:
        if style == 1:
            f.write("NCOLS          " + '{:<10n}'.format(ncols) + '\n')
            f.write("NROWS          " + '{:<10n}'.format(nrows) + '\n')
            f.write("XLLCENTER      " + '{:<10.2f}'.format(xllcenter) + '\n')
            f.write("YLLCENTER      " + '{:<10.2f}'.format(yllcenter) + '\n')
            f.write("DX             " + '{:<10.2f}'.format(dx) + '\n')
            f.write("DY             " + '{:<10.2f}'.format(dy) + '\n')
            f.write("NODATA_VALUE   " + '{:<10.2f}'.format(no_data) + '\n')
        elif style == 2:
            f.write("NCOLS          " + '{:<10n}'.format(ncols) + '\n')
            f.write("NROWS          " + '{:<10n}'.format(nrows) + '\n')
            f.write("XLLCORNER      " + '{:<10.2f}'.format(xllcorner) + '\n')
            f.write("YLLCORNER      " + '{:<10.2f}'.format(yllcorner) + '\n')
            f.write("CELLSIZE       " + '{:<10.2f}'.format(cellsize) + '\n')
            f.write("NODATA_VALUE   " + '{:<10.2f}'.format(no_data) + '\n')
        else:
            raise ValueError("style kwarg must be either 1 or 2.")

        for m in range(z.shape[0] - 1, -1, -1):
            for n in range(z.shape[1]):
                f.write('{:<16.2f}'.format(z[m, n]))
            if m != 0:
                f.write('\n')
    f.close()


def read_asc_grid(filename, footer=0):
    """Reads ASCII grid file (*.asc).
    footer kwarg specifies how many lines at end of *.asc file to skip.
    Returns a NumPy array of the values (dim MxN, where M is
    the number of Y-coordinates and N is the number of
    X-coordinates); a NumPy array of the X-coordinates (dim N);
    a NumPy array of the Y-coordinates (dim M); either a tuple
    of the grid cell size in the x direction and the grid cell
    size in the y direction (DX, DY) or the uniform grid cell size;
    and the NO_DATA value.
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
    with open(filename, 'rU') as f:
        while True:
            string, value = f.readline().split()
            header_lines += 1
            if string.lower() == 'ncols':
                ncols = int(value)
            elif string.lower() == 'nrows':
                nrows = int(value)
            elif string.lower() == 'xllcorner':
                xllcorner = float(value)
            elif string.lower() == 'xllcenter':
                xllcenter = float(value)
            elif string.lower() == 'yllcorner':
                yllcorner = float(value)
            elif string.lower() == 'yllcenter':
                yllcenter = float(value)
            elif string.lower() == 'cellsize':
                cellsize = float(value)
            elif string.lower() == 'cell_size':
                cellsize = float(value)
            elif string.lower() == 'dx':
                dx = float(value)
            elif string.lower() == 'dy':
                dy = float(value)
            elif string.lower() == 'nodata_value':
                no_data = float(value)
            elif string.lower() == 'nodatavalue':
                no_data = float(value)
            else:
                raise IOError("could not read *.asc file. Error in header.")

            if (ncols is not None) and (nrows is not None) and \
               (((xllcorner is not None) and (yllcorner is not None)) or
                ((xllcenter is not None) and (yllcenter is not None))) and \
               ((cellsize is not None) or ((dx is not None) and (dy is not None))) and \
               (no_data is not None):
                break
    f.close()

    raw_grid_array = np.genfromtxt(filename, skip_header=header_lines, skip_footer=footer)
    grid_array = np.flipud(raw_grid_array)

    if nrows != grid_array.shape[0] or ncols != grid_array.shape[1]:
        raise IOError("Error reading *.asc file. Encountered problem "
                      "with header: NCOLS and/or NROWS does not match "
                      "number of columns/rows in data file body.")

    if xllcorner is not None and yllcorner is not None:
        if dx is not None and dy is not None:
            xllcenter = xllcorner + dx/2.0
            yllcenter = yllcorner + dy/2.0
        else:
            xllcenter = xllcorner + cellsize/2.0
            yllcenter = yllcorner + cellsize/2.0

    if dx is not None and dy is not None:
        x = np.arange(xllcenter, xllcenter + ncols*dx, dx)
        y = np.arange(yllcenter, yllcenter + nrows*dy, dy)
    else:
        x = np.arange(xllcenter, xllcenter + ncols*cellsize, cellsize)
        y = np.arange(yllcenter, yllcenter + nrows*cellsize, cellsize)

    # Sometimes x and y and can be an entry too long due to imprecision in calculating
    # the upper cutoff for np.arange(); this bit takes care of that potential problem.
    if x.size == ncols + 1:
        x = x[:-1]
    if y.size == nrows + 1:
        y = y[:-1]

    if cellsize is None:
        cellsize = (dx, dy)

    return grid_array, x, y, cellsize, no_data
