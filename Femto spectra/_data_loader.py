import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import Locator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import os
from copy import deepcopy
import glob
from scipy.interpolate import interp1d


from sklearn import linear_model as lm
from numba import njit, prange



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
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

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
def register_seismic_cmap(zmin, zmax):
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
def get_groups(directory, condition=lambda grp_len: True):
    """Finds all *.a? files in a `directory` and sorts then into groups."""

    groups = []
    last_group = []
    last_name = ''
    for file in glob.glob(os.path.join(directory + '\*.a?')):

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


def plot_matrix(data, t_axis_mul=1, t_unit='$ps$', cmap='diverging', z_unit='$\Delta A$ ($m$OD)'):
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


def plot_data(data, symlog=False, title='TA data', t_unit='$ps$',
              z_unit='$\Delta A$ (mOD)', cmap='diverging', zmin=None, zmax=None,
              w0=None, w1=None, t0=None, t1=None, fig_size=(6, 4), dpi=500, filepath=None, transparent=True,
              linthresh=10, linscale=1, D_mul_factor=1e3):
    """data is individual dataset"""

    assert type(data) == Data

    plt.rcParams['figure.figsize'] = fig_size
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.23, hspace=0.26)

    times = data.times
    wavelengths = data.wavelengths
    D = data.D * D_mul_factor

    # cut data if necessary

    t_idx_start = find_nearest_idx(times, t0) if t0 is not None else 0
    t_idx_end = find_nearest_idx(times, t1) + 1 if t1 is not None else D.shape[0]

    wl_idx_start = find_nearest_idx(wavelengths, w0) if w0 is not None else 0
    wl_idx_end = find_nearest_idx(wavelengths, w1) + 1 if w1 is not None else D.shape[1]

    # crop the data if necessary
    D = D[t_idx_start:t_idx_end, wl_idx_start:wl_idx_end]
    times = times[t_idx_start:t_idx_end]
    wavelengths = wavelengths[wl_idx_start:wl_idx_end]

    zmin = np.min(D) if zmin is None else zmin
    zmax = np.max(D) if zmax is None else zmax

    register_div_cmap(zmin, zmax)

    x, y = np.meshgrid(wavelengths, times)  # needed for pcolormesh to correctly scale the image

    # plot data matrix D

    plt.pcolormesh(x, y, D, cmap=cmap, vmin=zmin, vmax=zmax)

    plt.colorbar(label=z_unit)
    plt.title(title)
    plt.ylabel(f'$\leftarrow$ Time delay ({t_unit})')
    plt.xlabel(r'Wavelength ($nm$) $\rightarrow$')

    plt.gca().invert_yaxis()

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


def merge(data_avrg):
    """Merges multiple datasets"""

    assert isinstance(data, np.ndarray)
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



def save_matrix_to_Glotaran(data, fname='output-GLOTARAN-long.ascii', delimiter='\t', encoding='utf8'):
    mat = np.vstack((data.wavelengths, data.D))
    buffer = f'Header\nOriginal filename: fname\nTime explicit\nintervalnr {data.times.shape[0]}\n'
    buffer += delimiter + delimiter.join(f"{num}" for num in data.times) + '\n'
    buffer += '\n'.join(delimiter.join(f"{num}" for num in row) for row in mat.T)

    with open(fname, 'w', encoding=encoding) as f:
        f.write(buffer)


def save_matrix(data, fname='output.txt', delimiter='\t', encoding='utf8', t0=None, t1=None, w0=None, w1=None):
    # cut data if necessary

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
    t_idx_start = find_nearest_idx(data.times, t0) if t0 is not None else 0
    t_idx_end = find_nearest_idx(data.times, t1) + 1 if t1 is not None else data.D.shape[0]

    # crop the data if necessary
    D_cut = data.D[t_idx_start:t_idx_end, :]
    avrg = np.average(D_cut, axis=0)

    data.D -= avrg

    return data




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

    plt.plot(_data.wavelengths, u)

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





# generates the matrix of folded exponentials for corresponding lifetimes
@njit(parallel=True, fastmath=True)  # speed up the calculation with numba, ~3 orders of magnitude of improvement
def _X(taus, times, sigma=0):
    # time zero mu is hardcoded to 0! so chirped data must be used
    # C is t x n matrix
    X = np.zeros((times.shape[0], taus.shape[0]))
    #     w = FWHM / (2 * np.sqrt(np.log(2)))
    w = sigma / np.sqrt(2)
    ks = 1 / taus
    for i in prange(X.shape[0]):
        for j in prange(X.shape[1]):
            t = times[i]
            k = ks[j]
            if w != 0:
                X[i, j] = 0.5 * np.exp(k * (k * w * w / 4 - t)) * math.erfc(w * k / 2 - t / w)
            else:
                X[i, j] = np.exp(-k * t) if t >= 0 else 0
    return X


def LDM(data, FWHM=0, alpha=1, p=0.5, cv=True, max_iter=1e4, lim_log=(None, None)):
    n = 50
    #     taus = np.logspace(-1.2, 3, n) # n lifetimes in logspace

    dt = data.times[1] - data.times[0]
    max_t = data.times[-1]
    start = np.floor(np.log10(dt))
    end = np.ceil(np.log10(max_t))
    n = int(30 * (end - start))
    lim_log = (start if lim_log[0] is None else lim_log[0], end if lim_log[1] is None else lim_log[1])
    taus = np.logspace(lim_log[0], lim_log[1], n, endpoint=True)

    X = _X(taus, data.times, FWHM)

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
    return alpha, mod.coef_.T, Data.from_matrix(fit, data.times, data.wavelengths), taus













