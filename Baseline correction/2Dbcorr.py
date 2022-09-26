#!/usr/bin/python3

# Loads data matrix and performs two dimensional baseline correction using asymmetrically reweighted penalized least 
# squares (arPLS) method, solved with discrete cosine transform (DCT). DCT method is based on the Robust smoother in
# the following paper: 10.1016/j.csda.2009.09.020. arPLS method for optimising the weights is described here: 10.1039/c4an01061b.
# Matlab implementation of smoothing algorithm is here: https://www.biomecardio.com/matlab/smoothn.m

# It basically smooths the data where there is no signal and interpolates the points where the signal is. This is ensured 
# by the zero weights in those regions.

import argparse
import sys
import os
from glob import glob
import numpy as np
from numpy.linalg import norm
from scipy.fftpack import dct, idct

from typing import Union, Iterable, List


# from https://stackoverflow.com/questions/40104377/issiue-with-implementation-of-2d-discrete-cosine-transform-in-python
def dct2(block):
    """
    Computes 2D discrete cosine transform. 
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    """
    Computes 2D inverse discrete cosine transform. 
    """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def fi(array: np.ndarray, values: Union[int, float, Iterable]) -> Union[int, List[int]]:
    """
    Finds index of nearest `value` in `array`. If value >  max(array), the last index of array
    is returned, if value < min(array), 0 is returned. Array must be sorted. Also works for value
    to be array, then array of indexes is returned.

    Parameters
    ----------
    array : ndarray
        Array to be searched.
    values : {int, float, list}
        Value or values to look for.

    Returns
    -------
    out : int, np.ndarray
        Found nearest index/es to value/s.
    """
    if not np.iterable(values):
        values = [values]

    ret_idx = [np.argmin(np.abs(array - i)) for i in values]

    return ret_idx[0] if len(ret_idx) == 1 else ret_idx


def load_matrix(fname, encoding='utf8', delimiter=','):
    _data = np.genfromtxt(fname, dtype=np.float64, skip_header=0, delimiter=delimiter, filling_values=None,
                          autostrip=True, encoding=encoding)
    data = _data[1:, 1:]
    times, wavelengths = _data[1:, 0], _data[0, 1:]
    return data, times, wavelengths


def save_mat2csv(fname, matrix, times=None, wls=None, unit='', delimiter=','):
    """Saves matrix with 6 significat digits."""
    times = np.arange(0, matrix.shape[0]) if times is None else times
    wls = np.arange(0, matrix.shape[1]) if wls is None else wls

    mat = np.hstack((times[:, None], matrix))
    buffer = f'unit: {unit} - Time | Wavelength->'
    buffer += delimiter + delimiter.join(f"{num}" for num in wls) + '\n'
    buffer += '\n'.join(delimiter.join(f"{num:.6g}" for num in row) for row in mat)

    with open(fname, 'w', encoding='utf8') as f:
        f.write(buffer)


def baseline_2D_arPLS(Y: np.ndarray, lam0: float = 1e7, lam1: float = 0.01, niter: int = 100, tol: float = 2e-3):
    """
    Performs two dimensional baseline correction using asymmetrically reweighted penalized least squares (arPLS) method,
    solved with discrete cosine transform (DCT). DCT method is based on the Robust smoother in the following
    paper: 10.1016/j.csda.2009.09.020. arPLS method for optimising the weights is described here: 10.1039/c4an01061b.
    Matlab implementation of smoothing algorithm is here: https://www.biomecardio.com/matlab/smoothn.m

    Parameters
    ----------
    Y : numpy.ndarray
        Data matrix.
    lam0 : float
        Smoothing parameter in the first dimension (axis == 0).
    lam1 : float
        Smoothing parameter in the second dimension (axis == 1).
    niter : int
        Maximum number of iterations.
    tol: float
        Tolerance for convergence based on weight matrix.

    """

    assert isinstance(Y, np.ndarray), "Y must be type of ndarray"
    assert Y.ndim == 2, "Y must be a matrix"

    N = Y.shape[0]
    K = Y.shape[1]

    W = np.ones_like(Y)  # weight matrix

    l0 = np.sqrt(lam0) * (2 - 2 * np.cos(np.arange(N) * np.pi / N))  # eigenvalues of diff matrix for 1st dimension
    l1 = np.sqrt(lam1) * (2 - 2 * np.cos(np.arange(K) * np.pi / K))  # eigenvalues of diff matrix for 2nd dimension

    gamma = 1 / (1 + (l0[:, None] + l1[None, :]) ** 2)

    Z = Y  # intialize the baseline
    D = None

    i = 0
    crit = 1

    while crit > tol and i < niter:
        Z = idct2(gamma * dct2(W * (Y - Z) + Z))  # calculate the baseline

        D = Y - Z  # data corrected for baseline
        Dn = D[D < 0]  # negative data values

        m = np.mean(Dn)
        s = np.std(Dn)

        new_W = 1 / (1 + np.exp(2 * (D - (2 * s - m)) / s))  # update weights with logistic function

        crit = norm(new_W - W) / norm(W)
        W = new_W

        if (i + 1) % int(np.sqrt(niter)) == 0:
            print(f'Iteration={i + 1}, {crit=:.2g}')
        i += 1

    return Z, D, W


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--delimiter", nargs="?", default=',', type=str,
                        help="Delimiter in input text files, default is ','.")
    parser.add_argument("--transpose", action="store_true",
                        help="If specified, transposes the input matrix.")

    parser.add_argument("--w0", nargs="?", default=None, type=float,
                        help="Start wavelength to crop the data.")
    parser.add_argument("--w1", nargs="?", default=None, type=float,
                        help="End wavelength to crop the data.")

    parser.add_argument("--t0", nargs="?", default=None, type=float,
                        help="Start time to crop the data.")
    parser.add_argument("--t1", nargs="?", default=None, type=float,
                        help="End time to crop the data.")

    parser.add_argument("--lam0", nargs="?", default=1e6, type=float,
                        help="Smoothing parameter in the first dimension (axis == 0), default is 1e7.")
    parser.add_argument("--lam1", nargs="?", default=0.0, type=float,
                        help="Smoothing parameter in the second dimension (axis == 1), default is 0.01.")
    parser.add_argument("--niter", nargs="?", default=100, type=int,
                        help="Maximum number of iterations, default is 100.")
    parser.add_argument("--tol", nargs="?", default=2e-3, type=float,
                        help="Tolerance for convergence based on weight matrix, default is 2e-3.")

    parser.add_argument("--save_baseline", action="store_true",
                        help="If specified, saves estimated baseline.")
    parser.add_argument("--save_weights", action="store_true",
                        help="If specified, saves final weight matrix.")

    parser.add_argument('files', nargs=argparse.ONE_OR_MORE)

    args, _ = parser.parse_known_args()

    # print(f'{args.w0=},{args.w1=},{args.t0=},{args.t1=}')

    fnames = []
    for fname in args.files:
        fnames += glob(fname)

    # assert len(fnames) > 0, "No filename specified"
    assert args.niter > 0, "Number of iterations must be > 1"

    for fpath in fnames:
        if not os.path.isfile(fpath):
            continue

        _dir, fname = os.path.split(fpath)  # get dir and filename
        fname, ext = os.path.splitext(fname)  # get filename without extension

        print(f'Processing \'{fname}{ext}\'...')

        data, times, wavelengths = load_matrix(fpath, delimiter=args.delimiter)
        if args.transpose:
            times, wavelengths = wavelengths, times
            data = data.T

        # crop the data if provided
        # find indicies
        it0 = fi(times, args.t0) if args.t0 is not None else 0
        ti1 = fi(times, args.t1) + 1 if args.t1 is not None else data.shape[0]

        iw0 = fi(wavelengths, args.w0) if args.w0 is not None else 0
        iw1 = fi(wavelengths, args.w1) + 1 if args.w1 is not None else data.shape[1]

        # crop
        data = data[it0:ti1, iw0:iw1]
        times = times[it0:ti1]
        wavelengths = wavelengths[iw0:iw1]

        # perform baseline correction
        Z, D, W = baseline_2D_arPLS(data, args.lam0, args.lam1, args.niter, args.tol)

        save_mat2csv(os.path.join(_dir, f'{fname}-b_corr.csv'), D, times, wavelengths, unit='min')

        if args.save_baseline:
            save_mat2csv(os.path.join(_dir, f'{fname}-b_corr_baseline.csv'), Z, times, wavelengths, unit='min')

        if args.save_weights:
            save_mat2csv(os.path.join(_dir, f'{fname}-b_corr_weights.csv'), W, times, wavelengths, unit='min')

        print(f'\'{fname}{ext}\' finished.\n')
