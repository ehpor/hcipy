from .fourier_transform import _time_it
from .fast_fourier_transform import FastFourierTransform, make_fft_grid
from .matrix_fourier_transform import MatrixFourierTransform
from .zoom_fast_fourier_transform import ZoomFastFourierTransform
from ..field import CartesianGrid, SeparatedCoords, make_pupil_grid

from tqdm import tqdm
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

def compute_fourier_performance_dataset(fourier_class, Ns, qs, fovs, t_max=0.01):
    """Compute a dataset of performance measurements for a Fourier transform class.

    Parameters
    ----------
    fourier_class : FourierTransform
        The Fourier transform class to measure the performance of.
    Ns : array_like
        The pupil grid sizes to measure.
    qs : array_like
        The oversampling factors to measure.
    fovs : array_like
        The fields of view to measure.
    t_max : float
        The maximum time to spend on a single performance measurement.

    Returns
    -------
    complexities : array_like
        The computational complexities for each of the measurements in GFLOPS.
    execution_times : array_like
        The execution times for each of the measurements in ms.
    """
    parameter_grid = CartesianGrid(SeparatedCoords((Ns, qs, fovs)))

    complexities = []
    execution_times = []

    for N, q, fov in tqdm(parameter_grid.points):
        N = int(N)

        input_grid = make_pupil_grid(N)
        output_grid = make_fft_grid(input_grid, q=q, fov=fov)

        # Don't run the largest Fourier transforms.
        if output_grid.size > 1e7:
            continue

        field = input_grid.ones()

        if fourier_class.is_supported(input_grid, output_grid):
            complexity = fourier_class.compute_complexity(input_grid, output_grid).num_operations
            if fourier_class == FastFourierTransform:
                ft = FastFourierTransform(input_grid, q, fov)
            else:
                ft = fourier_class(input_grid, output_grid)
            execution_time = _time_it(lambda: ft.forward(field), t_max=t_max)

        # Convert complexity to GFLOPS.
        complexities.append(complexity / 1e9)

        # Convert execution time to ms.
        execution_times.append(execution_time * 1e3)

    return complexities, execution_times

def fit_fourier_performance_data(complexities, execution_times):
    """Fit a power-law to the performance data.

    Parameters
    ----------
    complexities : array_like
        The computational complexities for each of the measurements in GFLOPS.
    execution_times : array_like
        The execution times for each of the measurements in ms.

    Returns
    -------
    popt : array_like
        The optimal parameters for the power-law fit.
    func : function
        The fitted power-law function.
    """
    # Fit a power law to our data.
    def powerlaw_in_log_space(x, a, b, c):
        return np.logaddexp(a + x * b, c)

    popt, _ = scipy.optimize.curve_fit(powerlaw_in_log_space, np.log(complexities), np.log(execution_times))

    return popt, lambda x: np.exp(powerlaw_in_log_space(np.log(x), *popt))

def plot_fourier_performance_data(datasets, ax=None):
    """Plot the fourier performance data.

    Parameters
    ----------
    datasets : dict
        A dictionary of datasets. The keys are the names of the datasets,
        and the values are tuples of (complexities, execution_times).
    ax : matplotlib axes
        The axes to plot on. If not given, a new figure and axes will be created.
    """
    if ax is None:
        ax = plt.gca()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of floating-point operations [GFLOPS]')
    ax.set_ylabel('Time taken for Fourier transform [ms]')
    ax.grid(c='0.7', ls=':')

    for label, data in datasets.items():
        x, y = data

        plotted_data = plt.plot(x, y, '.', label=label)
        c = plotted_data[0].get_color()

        _, f = fit_fourier_performance_data(x, y)

        x_fit = np.exp(np.linspace(np.log(x).min() - 1, np.log(x).max() + 1))
        plt.plot(x_fit, f(x_fit), ls='--', c=c)

    plt.legend()

def _cli():
    """A command-line interface for tuning Fourier transforms.
    """
    Ns = np.array([32, 64, 128, 256, 512, 1024])
    qs = np.array([1, 2, 4, 8, 16])
    fovs = np.array([1, 0.5, 0.3, 0.1])

    fourier_transforms = {
        'fft': FastFourierTransform,
        'mft': MatrixFourierTransform,
        'zfft': ZoomFastFourierTransform,
    }

    datasets = {
        label: compute_fourier_performance_dataset(fourier_class, Ns, qs, fovs)
        for label, fourier_class in fourier_transforms.items()
    }

    plot_fourier_performance_data(datasets)
    plt.show()

    print('fit_results:')
    for label, dataset in datasets.items():
        popt, _ = fit_fourier_performance_data(*dataset)
        print(f'  {label}:')
        print(f'    a: {popt[0]:.3f}')
        print(f'    b: {popt[1]:.3f}')
        print(f'    c: {popt[2]:.3f}')
