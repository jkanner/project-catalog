from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.stats import gaussian_kde
from project_catalog.galactic_binary import GalacticBinary


def compute_1d_hdrs(galactic_binary: GalacticBinary, candidate_index: int, grid_size=1000):
    """
    Compute the HDR coverage percentile of x_true for each parameter and return a dictionary.

    Parameters
    ----------
    galactic_binary : GalacticBinary
        The GalacticBinary object containing the posterior samples and candidates.
    grid_size : int
        Number of grid points for KDE.

    Returns
    -------
    coverage_level : float
        The smallest (1 - alpha) such that x_true lies in the HDR.
    """
    hdrs = {}
    for param in galactic_binary.params:
        
        samples = np.asarray(galactic_binary.subset_chain()[param])
        kde = gaussian_kde(samples)
        x_grid = np.linspace(samples.min(), samples.max(), grid_size)
        pdf = kde.evaluate(x_grid)
        dx = x_grid[1] - x_grid[0]
        # Interpolate PDF at x_true
        p_true = kde.evaluate([galactic_binary.subset_candidates()[param].iloc[candidate_index]])[0]

        # Compute cumulative probability above p_true
        mask = pdf >= p_true
        coverage = np.sum(pdf[mask]) * dx
        hdrs[param] = coverage

    return hdrs

def compute_8d_hdr(galactic_binary: GalacticBinary, candidate_index: int):
    """
    Compute the multivariate HDR coverage percentile of true_vals.

    samples_df : pandas.DataFrame, shape (N, D)
    true_vals   : array-like length D
    bw_method   : passed to gaussian_kde
    """
    # 1) prepare the data for gaussian_kde: shape (D, N)
    X = galactic_binary.subset_chain().values.T
    kde = KernelDensity(kernel='gaussian').fit(X.T)
    # kde = gaussian_kde(X, bw_method=bw_method)

    # 2) densities at the samples
    dens = kde.score_samples(X.T)  # length N array
    # dens = kde.evaluate(X)      # length N array

    # 3) density at the true point
    #    must be shape (D, M); here M=1
    tv = np.asarray(galactic_binary.subset_candidates().iloc[candidate_index]).reshape(-1,1)
    p_true = kde.score_samples(tv.T)[0]

    # 4) fraction of posterior mass with density ≥ p_true
    return np.count_nonzero(dens >= p_true) / len(dens)
