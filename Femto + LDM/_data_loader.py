import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator, MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import math
import os
from copy import deepcopy
import glob
from scipy.interpolate import interp1d
from scipy.special import erfc
from scipy.linalg import lstsq
from matplotlib.ticker import *

from sklearn import linear_model as lm
# from numba import njit, prange




# works for numbers and array of numbers
def find_nearest_idx(array, value):
    if isinstance(value, (int, float)):
        value = np.asarray([value])
    else:
        value = np.asarray(value)

    result = np.empty_like(value, dtype=int)
    for i in range(value.shape[0]):
        idx = np.searchsorted(array, value[i], side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value[i] - array[idx - 1]) < math.fabs(value[i] - array[idx])):
            result[i] = idx - 1
        else:
            result[i] = idx
    return result if result.shape[0] > 1 else result[0]


def find_nearest(array, value):
    idx = find_nearest_idx(array, value)
    return array[idx]


# needed for correctly display tics for symlog scale
class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]  # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper e	nd

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (
                dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] * 10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] - self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (
                dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1] * 10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1] + self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))

# nice colormap for heat maps, this ensures that zero is always white color
def register_div_cmap(zmin, zmax):
    """Registers `diverging` diverging color map just suited for data."""

    diff = zmax - zmin
    w = np.abs(zmin / diff)  # white color point set to zero z value

    _cdict = {'red': ((0.0, 0.0, 0.0),
                      (w / 2, 0.0, 0.0),
                      (w, 1.0, 1.0),
                      (w + (1 - w) / 3, 1.0, 1.0),
                      (w + (1 - w) * 2 / 3, 1.0, 1.0),
                      (1.0, 0.3, 0.3)),

              'green': ((0.0, 0, 0),
                        (w / 2, 0.0, 0.0),
                        (w, 1.0, 1.0),
                        (w + (1 - w) / 3, 1.0, 1.0),
                        (w + (1 - w) * 2 / 3, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.3, 0.3),
                       (w / 2, 1.0, 1.0),
                       (w, 1.0, 1.0),
                       (w + (1 - w) / 3, 0.0, 0.0),
                       (w + (1 - w) * 2 / 3, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
              }

    custom_cmap = LinearSegmentedColormap('diverging', _cdict)
    cm.register_cmap('diverging', custom_cmap)

# seismic heatmap suited for data similar as above
def register_seismic__cmap(zmin, zmax):
    """Registers `seismic_` diverging color map just suited for data."""

    diff = zmax - zmin
    w = np.abs(zmin / diff)  # white color point set to zero z value

    _cdict = {'red': ((0, 0.0, 0.0),
                      (w / 2, 0.0, 0.0),
                      (w, 1.0, 1.0),
                      (1.5 * w, 1.0, 1.0),
                      (1.0, 0.2, 0.2)),

              'green': ((0.0, 0, 0),
                        (w / 2, 0.0, 0.0),
                        (w, 1.0, 1.0),
                        (1.5 * w, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.2, 0.2),
                       (w / 2, 1.0, 1.0),
                       (w, 1.0, 1.0),
                       (1.5 * w, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
              }

    custom_cmap = LinearSegmentedColormap('seismic_', _cdict)
    cm.register_cmap('seismic_', custom_cmap)


# read dir to get all a? files
def get_groups(directory, ext='a?', condition=lambda grp_len: True):
    """Finds all *.a? files in a `directory` and sorts then into groups."""

    groups = []
    last_group = []
    last_name = ''
    for file in glob.glob(os.path.join(directory + f'\*.{ext}')):

        path = os.path.splitext(file)[0]
        name = os.path.split(path)[1]

        if last_name == '':
            last_name = name

        if last_name == name:
            last_group.append(file)
        else:
            if condition(len(last_group)):
                groups.append(last_group)

            last_group = []
            last_name = name
            last_group.append(file)

    if condition(len(last_group)):
        groups.append(last_group)

    return groups


# class data for storing data matrix
class Data:
    def __init__(self, fname=None, delimiter='\t', t_axis_mul=1e-3, skiprows=0, transpose=False):
        """Loading of a? files."""
        self.wavelengths = None
        self.times = None
        self.D = None
        self.fname = None

        if fname is not None:
            data = np.genfromtxt(fname, delimiter=delimiter, skip_header=skiprows, dtype=np.float64)
            self.wavelengths = data[1:, 0] if transpose else data[0, 1:]
            self.times = data[0, 1:] if transpose else data[1:, 0]
            self.times *= t_axis_mul
            self.D = data[1:, 1:].T if transpose else data[1:, 1:]
            self.fname = fname

    @classmethod
    def from_matrix(cls, data_mat, times, wavelengths):
        _data = cls()

        assert data_mat.shape[0] == times.shape[0]
        assert data_mat.shape[1] == wavelengths.shape[0]

        _data.D = data_mat
        _data.times = times
        _data.wavelengths = wavelengths
        return _data
    
    def crop(self, t_lim=(None, None), w_lim=(None, None)):
        
        t_start = t_lim[0] if t_lim[0] is not None else self.times[0]
        t_stop = t_lim[1] if t_lim[1] is not None else self.times[-1]
        t_idxs = find_nearest_idx(self.times, [t_start, t_stop])
        
        w_start = w_lim[0] if w_lim[0] is not None else self.wavelengths[0]
        w_stop = w_lim[1] if w_lim[1] is not None else self.wavelengths[-1]
        w_idxs = find_nearest_idx(self.wavelengths, [w_start, w_stop])
        
        self.times = self.times[t_idxs[0]:t_idxs[1] + 1]
        self.wavelengths = self.wavelengths[w_idxs[0]:w_idxs[1] + 1]
        
        self.D = self.D[t_idxs[0]:t_idxs[1] + 1, w_idxs[0]:w_idxs[1] + 1]

    def __str__(self):
        if self.D is not None:
            return (f'{self.fname}: {str(self.D.shape)}')


def load_groups(groups, **kwargs):
    """**kwargs are passed to Data class constructor"""
    n_rows = len(groups)
    n_cols = max((len(l) for l in groups))

    data = np.empty((n_rows, n_cols), dtype=Data)
    for i, group in enumerate(groups):
        for j, fname in enumerate(group):
            data[i, j] = Data(fname, **kwargs)
    #             print(data[i, j])

    return data


def plot_matrix(data, t_axis_mul=1, t_unit='$ps$', cmap='diverging', z_unit='$\Delta A$ (mOD)'):
    """data are matrix of loaded datasets"""

    assert isinstance(data, np.ndarray)
    assert type(data[0][0]) == Data

    n_rows = data.shape[0]
    n_cols = data.shape[1]

    plt.rcParams['figure.figsize'] = (5 * n_cols, 4 * n_rows)

    for i in range(n_rows):
        for j in range(n_cols):
            plt.subplot(n_rows, n_cols, j + 1 + i * n_cols)  # # of rows, # of columns, index counting form 1

            if data[i, j] is None:
                continue

            fname = data[i, j].fname

            zmin = data[i, j].D.min()
            zmax = data[i, j].D.max()

            register_div_cmap(zmin, zmax)

            x, y = np.meshgrid(data[i, j].wavelengths,
                               data[i, j].times * t_axis_mul)  # needed for pcolormesh to correctly scale the image

            # plot data matrix D

            plt.pcolormesh(x, y, data[i, j].D, cmap=cmap, vmin=zmin, vmax=zmax)
            plt.colorbar(label=z_unit)
            plt.title(os.path.split(fname)[1])
            plt.ylabel(f'$\leftarrow$ Time delay ({t_unit})')
            plt.xlabel(r'Wavelength ($nm$) $\rightarrow$')

            plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

#     plt.savefig(fname='output.png', format='png', transparent=True, dpi=500)


def plot_data(data, symlog=False, title='TA data', t_unit='ps',
              z_unit='$10^3\ \Delta A$', cmap='diverging', z_lim=(None, None),
              t_lim=(None, None), w_lim=(None, None), fig_size=(6, 4), dpi=500, filepath=None, transparent=True,
              linthresh=10, linscale=1, D_mul_factor=1e3, y_major_formatter=ScalarFormatter()):
    """data is individual dataset"""

    assert type(data) == Data

    plt.rcParams['figure.figsize'] = fig_size
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.23, hspace=0.26)

    times = data.times
    wavelengths = data.wavelengths
    D = data.D * D_mul_factor

    # cut data if necessary

    t_lim = (data.times[0] if t_lim[0] is None else t_lim[0], data.times[-1] if t_lim[1] is None else t_lim[1])
    w_lim = (
        data.wavelengths[0] if w_lim[0] is None else w_lim[0], data.wavelengths[-1] if w_lim[1] is None else w_lim[1])

    # t_idx_start = find_nearest_idx(times, t0) if t0 is not None else 0
    # t_idx_end = find_nearest_idx(times, t1) + 1 if t1 is not None else D.shape[0]
    #
    # wl_idx_start = find_nearest_idx(wavelengths, w0) if w0 is not None else 0
    # wl_idx_end = find_nearest_idx(wavelengths, w1) + 1 if w1 is not None else D.shape[1]
    #
    # # crop the data if necessary
    # D = D[t_idx_start:t_idx_end, wl_idx_start:wl_idx_end]
    # times = times[t_idx_start:t_idx_end]
    # wavelengths = wavelengths[wl_idx_start:wl_idx_end]

    zmin = np.min(D) if z_lim[0] is None else z_lim[0]
    zmax = np.max(D) if z_lim[1] is None else z_lim[1]

    register_div_cmap(zmin, zmax)

    x, y = np.meshgrid(wavelengths, times)  # needed for pcolormesh to correctly scale the image

    # plot data matrix D

    plt.pcolormesh(x, y, D, cmap=cmap, vmin=zmin, vmax=zmax)

    plt.colorbar(label=z_unit)
    plt.title(title)
    plt.ylabel(f'$\leftarrow$ Time delay / {t_unit}')
    plt.xlabel(r'Wavelength / nm $\rightarrow$')

    plt.ylim(t_lim)
    plt.xlim(w_lim)

    plt.gca().invert_yaxis()
    if y_major_formatter:
        plt.gca().yaxis.set_major_formatter(y_major_formatter)

    if symlog:
        plt.yscale('symlog', subsy=[2, 3, 4, 5, 6, 7, 8, 9], linscaley=linscale, linthreshy=linthresh)
        yaxis = plt.gca().yaxis
        yaxis.set_minor_locator(MinorSymLogLocator(linthresh))

    plt.tight_layout()

    # save to file
    if filepath:
        ext = os.path.splitext(filepath)[1].lower()[1:]
        plt.savefig(fname=filepath, format=ext, transparent=transparent, dpi=dpi)

    plt.show()

# averaging
def average(data, keepdims=True):
    """Averages rows of matrix"""

    assert isinstance(data, np.ndarray)
    assert type(data[0][0]) == Data

    data_avrg = np.empty(data.shape[0], dtype=Data)

    for i in range(data.shape[0]):

        # combine all datasets in a row into a 3D array (creates new axis)
        D_all = np.stack([dataset.D for dataset in data[i] if dataset is not None], axis=2)

        data_avrg[i] = deepcopy(data[i, 0])
        data_avrg[i].D = D_all.mean(axis=2, keepdims=False)  # average all datasets in a row
        data_avrg[i].fname += '-avrg'

    return data_avrg[:, None] if keepdims else data_avrg


def merge(data_avrg, average_same=True):
    """Merges multiple datasets and returns one matrix as Data object"""

    assert isinstance(data_avrg, np.ndarray)
    _data_avrg = deepcopy(data_avrg.squeeze())

    final_data = _data_avrg[0]

    # stack all matrices vertically and times
    new_D = np.vstack(tuple(dataset.D for dataset in _data_avrg))
    new_times = np.hstack(tuple(dataset.times for dataset in _data_avrg))

    # sort times and full matrix

    i_sort = np.argsort(new_times)
    new_times = new_times[i_sort]
    new_D = new_D[i_sort, :]

    # find indices and counts of same values in new_times
    vals, indices, counts = np.unique(new_times, return_index=True, return_counts=True)
    
    if average_same:
        for i in range(indices.shape[0]):
            if counts[i] < 2:
                continue

            idx = indices[i]

            # replace the first occurrence by average of all of them
            new_D[idx] = new_D[idx:idx + counts[i], :].mean(axis=0)

    # select only that elements of first occurrences
    new_times = new_times[indices]
    new_D = new_D[indices]

    final_data.D = new_D
    final_data.times = new_times

    return final_data



def save_matrix_to_Glotaran(data, fname='output-GLOTARAN.ascii', delimiter='\t', encoding='utf8'):
    assert type(data) == Data
    mat = np.vstack((data.wavelengths, data.D))
    buffer = f'Header\nOriginal filename: fname\nTime explicit\nintervalnr {data.times.shape[0]}\n'
    buffer += delimiter + delimiter.join(f"{num}" for num in data.times) + '\n'
    buffer += '\n'.join(delimiter.join(f"{num}" for num in row) for row in mat.T)

    with open(fname, 'w', encoding=encoding) as f:
        f.write(buffer)


def save_matrix(data, fname='output.txt', delimiter='\t', encoding='utf8', t0=None, t1=None, w0=None, w1=None):
    # cut data if necessary
    assert type(data) == Data

    t_idx_start = find_nearest_idx(data.times, t0) if t0 is not None else 0
    t_idx_end = find_nearest_idx(data.times, t1) + 1 if t1 is not None else data.D.shape[0]

    wl_idx_start = find_nearest_idx(data.wavelengths, w0) if w0 is not None else 0
    wl_idx_end = find_nearest_idx(data.wavelengths, w1) + 1 if w1 is not None else data.D.shape[1]

    # crop the data if necessary
    D_crop = data.D[t_idx_start:t_idx_end, wl_idx_start:wl_idx_end]
    times_crop = data.times[t_idx_start:t_idx_end]
    wavelengths_crop = data.wavelengths[wl_idx_start:wl_idx_end]

    mat = np.vstack((wavelengths_crop, D_crop))
    buffer = delimiter + delimiter.join(f"{num}" for num in times_crop) + '\n'
    buffer += '\n'.join(delimiter.join(f"{num}" for num in row) for row in mat.T)

    with open(fname, 'w', encoding=encoding) as f:
        f.write(buffer)


def baseline_corr(data, t0=0, t1=200):
    """Subtracts a average of specified time range from all data.
    Deep copies the object and new averaged one is returned."""

    assert type(data) == Data

    _data = deepcopy(data)

    t_idx_start = find_nearest_idx(_data.times, t0) if t0 is not None else 0
    t_idx_end = find_nearest_idx(_data.times, t1) + 1 if t1 is not None else _data.D.shape[0]

    D_selection = _data.D[t_idx_start:t_idx_end, :]
    _data.D -= D_selection.mean(axis=0)

    return _data


def chirp_correct(data, lambda_c=388, mu=(1,), time_offset=0.2):
    """Time offset from zero, if """

    assert isinstance(mu, tuple) and isinstance(lambda_c, (int, float)) and len(mu) >= 1

    _data = deepcopy(data)

    u = np.ones(_data.wavelengths.shape[0], dtype=np.float64) * mu[0]
    for i in range(1, len(mu)):
        u += mu[i] * ((_data.wavelengths - lambda_c) / 100) ** i

    idx0 = find_nearest_idx(u, time_offset)
    if u[idx0] < time_offset:
        idx0 += 1

    max_u = u.max()

    # plt.plot(_data.wavelengths, u)

    # crop wavelengths data from idx0 to end
    _data.D = _data.D[:, idx0:]
    _data.wavelengths = _data.wavelengths[idx0:]
    u = u[idx0:]

    new_times = np.copy(_data.times) - max_u

    # remove first entries to correct for time_offset
    idx_t_min = find_nearest_idx(new_times, -time_offset)
    if new_times[idx_t_min] < -time_offset:
        idx_t_min += 1

    new_times = new_times[idx_t_min:]
    new_D = np.zeros((new_times.shape[0], _data.wavelengths.shape[0]), dtype=np.float64)

    for i in range(_data.wavelengths.shape[0]):
        f = interp1d(data.times - u[i], _data.D[:, i], kind='linear', copy=False, assume_sorted=True)
        new_D[:, i] = f(new_times)  # interpolate the y values

    new_data = Data.from_matrix(new_D, new_times, _data.wavelengths)

    return new_data



# # generates the matrix of folded exponentials for corresponding lifetimes
# @njit(parallel=True, fastmath=True)  # speed up the calculation with numba, ~3 orders of magnitude of improvement
# def _X(taus, times, sigma=0):
#     # time zero mu is hardcoded to 0! so chirped data must be used
#     # C is t x n matrix
#     X = np.zeros((times.shape[0], taus.shape[0]))
#     #     w = FWHM / (2 * np.sqrt(np.log(2)))
#     w = sigma / np.sqrt(2)
#     ks = 1 / taus
#     for i in prange(X.shape[0]):
#         for j in prange(X.shape[1]):
#             t = times[i]
#             k = ks[j]
#             if w != 0:
#                 X[i, j] = 0.5 * np.exp(k * (k * w * w / 4 - t)) * math.erfc(w * k / 2 - t / w)
#             else:
#                 X[i, j] = np.exp(-k * t) if t >= 0 else 0
#     return X



def _X(taus, times, irf_fwhm=0):
    """Calculates the X matrix (time x tau) for LDM analysis."""
    # X = np.zeros((times.shape[0], taus.shape[0]))
    w = irf_fwhm / (2 * np.sqrt(np.log(2)))

    ks = 1 / taus[None, :]
    ts = times[:, None]

    if w > 0:
        X = 0.5 * np.exp(ks * (ks * w * w / 4 - ts)) * erfc(w * ks / 2 - ts / w)
    else:
        X = np.exp(-ts * ks) * np.heaviside(ts, 1)

    return X

# TODO: CV estimation of best alpha
def LDM(data, irf_fwhm=0.1, n_taus=None, alpha=1, p=0, cv=False, max_iter=1e4, lim_log=(None, None)):

    """irf_fwhm=0.1 is in ps in default, the same as data time units """

    assert type(data) == Data

    dt = data.times[1] - data.times[0]
    max_t = data.times[-1]
    start = np.floor(np.log10(dt))
    end = np.ceil(np.log10(max_t))
    n = int(30 * (end - start)) if n_taus is None else n_taus
    lim_log = (start if lim_log[0] is None else lim_log[0], end if lim_log[1] is None else lim_log[1])

    taus = np.logspace(lim_log[0], lim_log[1], n, endpoint=True)

    X = _X(taus, data.times, irf_fwhm)

    if p == 0:
        if not cv:
            mod = lm.Ridge(alpha=alpha,
                           max_iter=None,
                           solver='svd')
        else:
            mod = lm.RidgeCV()
    else:
        mod = lm.MultiTaskElasticNet(alpha=alpha,
                                     l1_ratio=p,
                                     max_iter=max_iter)
    mod.verbose = 0
    mod.fit_intercept = False
    mod.copy_X = True

    #     coefs = np.empty((n, data.wavelengths.shape[0]))
    #     fit = np.empty_like(data.D)
    #     alphas = np.empty(data.wavelengths.shape[0])
    #     for i in range(data.wavelengths.shape[0]):
    #         mod.fit(X, data.D[:, i])
    #         coefs[:, i] = mod.coef_.copy()
    #         fit[:, i] = mod.predict(X)
    #         if hasattr(mod, 'alpha_'):
    #             alphas[i] = mod.alpha_

    mod.fit(X, data.D)
    fit = mod.predict(X)
    if hasattr(mod, 'alpha_'):
        alpha = mod.alpha_
    return mod.coef_.T, Data.from_matrix(fit, data.times, data.wavelengths), taus


def _convert_lifetime(tau):
    """Converts lifetime in ps to fs, ns or keeps it in ps and returns its corresponding unit."""
    unit = 'ps'
    tau_ = tau
    if tau < 1:
        tau_, unit = tau * 1e3, 'fs'
    elif tau >= 1e3:
        tau_, unit = tau * 1e-3, 'ns'
    return tau_, unit


def _x_symlog(start, end, n, linscale=1, linthresh=1):
    """Generates n points from symlog space given start and end points
     as used in matplotlib."""
    start = np.sign(start) * linthresh * ((np.log10(np.abs(start)) - np.log10(linthresh)) / linscale + 1) if np.abs(start) >= linthresh else start
    end = np.sign(end) * linthresh * ((np.log10(np.abs(end)) - np.log10(linthresh)) / linscale + 1) if np.abs(end) >= linthresh else end
    lin = np.linspace(start, end, n)
    symlog = np.empty_like(lin)

    for i in range(lin.shape[0]):
        symlog[i] = lin[i] if np.abs(lin[i]) < linthresh else np.sign(lin[i]) * 10 ** (linscale * (np.abs(lin[i]) / linthresh - 1) + np.log10(linthresh))
    return symlog


def plot_LDA_data(data, taus, s_opt, symlog=False, title='Transient Absorption Data', t_unit='$ps$',
                  z_unit='$\Delta A$ ($m$OD)', cmap='diverging', z_lim=(None, None),
                  fig_size=(12, 16), dpi=500, filepath=None, transparent=True, t_lim=(-0.3, 70), w_lim=(350, 800),
                  linthresh=10, linscale=1, D_mul_factor=1e3, lifetimes=None, tidx_animate=None, FWHM=0.1):

    # NO tidx_animate is a tuple, fist element is all indexes of shown spectra, the second element is index of bolded spectrum
    # cmap for LDA is custom made `seismic_` cmap

    t_lim = (data.times[0] if t_lim[0] is None else t_lim[0], data.times[-1] if t_lim[1] is None else t_lim[1])
    w_lim = (
    data.wavelengths[0] if w_lim[0] is None else w_lim[0], data.wavelengths[-1] if w_lim[1] is None else w_lim[1])

    fig, ax = plt.subplots(2, 2 if tidx_animate is not None else 1, figsize=fig_size)
    ax = ax.reshape((ax.shape[0], -1))

    D = data.D * D_mul_factor

    zmin = np.min(D) if z_lim[0] is None else z_lim[0]
    zmax = np.max(D) if z_lim[1] is None else z_lim[1]

    register_div_cmap(zmin, zmax)

    x, y = np.meshgrid(data.wavelengths, data.times)  # needed for pcolormesh to correctly scale the image

    # plot data matrix D

    mappable = ax[0, 0].pcolormesh(x, y, D, cmap=cmap, vmin=zmin, vmax=zmax)

    fig.colorbar(mappable, ax=ax[0, 0], label=z_unit)
    ax[0, 0].set_title(title)
    ax[0, 0].set_ylabel(f'$\leftarrow$ Time delay ({t_unit})')
    ax[0, 0].set_xlabel(r'Wavelength ($nm$) $\rightarrow$')
    ax[0, 0].set_ylim(t_lim)
    ax[0, 0].set_xlim(w_lim)

    ax[0, 0].invert_yaxis()

    if tidx_animate is not None:
        assert isinstance(tidx_animate, np.ndarray)
        ax[0, 0].axhline(y=data.times[tidx_animate[-1]], color='black', ls='-', linewidth=1)

    #     ax[0].xaxis.set_ticks([])

    if symlog:
        ax[0, 0].set_yscale('symlog', subsy=[2, 3, 4, 5, 6, 7, 8, 9], linscaley=linscale, linthreshy=linthresh)
        ax[0, 0].yaxis.set_minor_locator(MinorSymLogLocator(linthresh))

    # LDM plot

    x, y = np.meshgrid(data.wavelengths, taus)  # needed for pcolormesh to correctly scale the image

    register_seismic__cmap(s_opt.min(), s_opt.max())

    #     levels = MaxNLocator(nbins=30).tick_values(-zlim, +zlim)
    #     plt.contourf(x, y, s_opt, cmap='seismic', levels=levels)

    mappable = ax[1, 0].pcolormesh(x, y, s_opt, cmap='seismic_', vmin=s_opt.min(), vmax=+s_opt.max())

    fig.colorbar(mappable, ax=ax[1, 0], label='Amplitude')

    ax[1, 0].set_title("Lifetime Density Map")
    ax[1, 0].set_ylabel(r'Lifetime ({}) $\rightarrow$'.format(t_unit))
    ax[1, 0].set_xlabel(r'Wavelength ($nm$) $\rightarrow$')

    if lifetimes is not None:  # lifetimes are in ps
        lifetimes_text = []
        for i, tau in enumerate(lifetimes):
            ax[1, 0].axhline(y=tau, color='black', ls='--', linewidth=1)
            x_text = data.wavelengths[0] + (data.wavelengths[-1] - data.wavelengths[0]) * 0.71
            tau_, unit = _convert_lifetime(tau)
            lifetimes_text.append(f'$\\tau_{i + 1} = {tau_:.3g}\\ {unit}$')
            ax[1, 0].text(x_text, tau * 1.3, lifetimes_text[i])

    ax[1, 0].set_yscale('log')
    ax[1, 0].set_xlim(w_lim)

    if tidx_animate is not None:
        # spectrum plot
        ax[0, 1].set_title("Spectrum")

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        # place a text box in upper left in axes coords
        tau_, unit = _convert_lifetime(data.times[tidx_animate[-1]])
        ax[0, 1].text(0.05, 0.90, f'$t={tau_:.3g}\\ {unit}$', transform=ax[0, 1].transAxes,
                      verticalalignment='top', bbox=props, size=12)

        ax[0, 1].set_ylabel(z_unit)
        ax[0, 1].set_xlabel(r'Wavelength ($nm$) $\rightarrow$')
        ax[0, 1].set_ylim(D.min() - 0.1 * np.abs(D.min()), D.max() + 0.1 * np.abs(D.max()))
        ax[0, 1].set_xlim(w_lim)

        # plot all spectra in gray
        #         spectrum_cmap = cm.get_cmap(spectrum_cmap)
        #         num = 10
        #         end = tidx_animate[1]
        #         start = end - num if end - num >= 0 else 0
        for i in range(tidx_animate.shape[0]):
            ax[0, 1].plot(data.wavelengths, D[tidx_animate[i]], color='grey', lw=0.2)
        # plot bold one
        ax[0, 1].plot(data.wavelengths, D[tidx_animate[-1]], color='black', lw=2)

        if lifetimes is not None:
            # DADS plot

            ax[1, 1].set_title("Decay Associated Difference Spectra (DADS)")
            ax[1, 1].set_ylabel('Amplitude')
            ax[1, 1].set_xlabel(r'Wavelength ($nm$) $\rightarrow$')
            ax[1, 1].set_xlim(w_lim)

            # DADS calculation
            C = _X(np.asarray(lifetimes), data.times, FWHM)
            S_T = lstsq(C, data.D)[0]

            for i in range(S_T.shape[0]):
                ax[1, 1].plot(data.wavelengths, S_T[i], label=lifetimes_text[i])

            ax[1, 1].legend(prop={'size': 10})

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    if filepath:
        plt.savefig(fname=filepath, format='png', transparent=transparent, dpi=dpi)
        plt.close(fig)
        return

    plt.show()






