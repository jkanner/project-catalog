from KDEpy.FFTKDE import FFTKDE
from sklearn.neighbors import KernelDensity
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.stats import gaussian_kde
from project_catalog.galactic_binary import GalacticBinary
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold


def compute_1d_hdrs(galactic_binary: GalacticBinary, candidate_index: int):
    """
    Faster version with data normalization for numerical stability.
    """
    hdrs = {}

    for param in galactic_binary.binary_params:
        # Get samples
        samples = np.asarray(galactic_binary.get_binary_parameters_posterior()[param])

        # Normalize data for numerical stability
        loc = np.median(samples)
        scale = np.std(samples)
        if scale > 0:
            samples_norm = (samples - loc) / scale
        else:
            # Handle constant data
            hdrs[param] = 1.0 if np.allclose(samples, samples[0]) else 0.0
            continue

        # Fit KDE on normalized data
        kde = FFTKDE(kernel='gaussian', bw='ISJ')
        kde.fit(samples_norm)

        # Get candidate value and normalize it
        x_true = galactic_binary.get_binary_parameters_injections().iloc[candidate_index][param]
        x_true_norm = (x_true - loc) / scale

        # Evaluate on default grid
        x_grid_norm, pdf_norm = kde.evaluate()

        # Method 1: Work in normalized space (more efficient)
        p_true_norm = np.interp(x_true_norm, x_grid_norm, pdf_norm)
        densities_at_samples_norm = np.interp(samples_norm, x_grid_norm, pdf_norm)

        # HDR coverage is scale-invariant, so we can compute it in normalized space
        coverage = np.mean(densities_at_samples_norm >= p_true_norm)

        hdrs[param] = coverage

    return hdrs

def compute_sky_location_hdr(
    galactic_binary: GalacticBinary,
    candidate_index: int,
    method: str = "kde_cv",
    standardize: bool = True,
    weight_concentration_prior: float = 0.8,   # smaller => fewer active comps
    random_state: int = 42,
):
    """
    Compute 2D HDR for sky location using KDE/BGMM with robust settings.
    Returns: fraction of posterior mass with density ≥ candidate's density.
    """

    # 1) Load & clean
    X = np.asarray(galactic_binary.get_sky_location_posterior().values, dtype=float)  # (N,2)
    cand = np.asarray(
        galactic_binary.get_sky_location_injections().iloc[candidate_index].values,
        dtype=float,
    ).reshape(1, -1)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Expected (N,2) data; got {X.shape}")

    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if X.size == 0:
        raise ValueError("No finite samples after cleaning.")

    # 2) Standardize (helps both KDE and BGMM when using isotropic bandwidths)
    scaler = StandardScaler().fit(X) if standardize else None
    Xz = scaler.transform(X) if scaler is not None else X
    cand_z = scaler.transform(cand) if scaler is not None else cand

    N, D = Xz.shape  # D should be 2

    # ---------- Helper: robust BGMM fit with retries ----------
    def _fit_bgmm(Xz, init_params="kmeans", reg=1e-5, rs=None):
        model = BayesianGaussianMixture(
            n_components=15,                                   # upper bound in 2D
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=weight_concentration_prior,
            mean_prior=np.zeros(D),
            mean_precision_prior=1.0,
            degrees_of_freedom_prior=D + 2,                   # 4 for D=2
            covariance_prior=np.eye(D),
            reg_covar=reg,
            init_params=init_params,
            n_init=10,
            max_iter=1000,
            tol=1e-3,
            random_state=rs,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(Xz)
        return model

    # ---------- Choose method ----------
    if method == "kde_cv":
        # Scott factor on standardized data (σ≈1): h = n^(-1/(d+4))
        scott = N ** (-1.0 / (D + 4))
        # Search a reasonable range around Scott
        bw_grid = scott * np.geomspace(0.25, 4.0, 21)

        kde = KernelDensity(kernel="gaussian")
        # Use shuffled KFold to reduce MCMC-correlation bias in CV
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        grid = GridSearchCV(kde, {"bandwidth": bw_grid}, cv=cv)
        grid.fit(Xz)
        best_kde = grid.best_estimator_
        log_dens = best_kde.score_samples(Xz)
        log_p_cand = float(best_kde.score_samples(cand_z)[0])

    elif method == "kde_scott":
        scott = N ** (-1.0 / (D + 4))
        kde = KernelDensity(kernel="gaussian", bandwidth=scott).fit(Xz)
        log_dens = kde.score_samples(Xz)
        log_p_cand = float(kde.score_samples(cand_z)[0])

    elif method == "kde_silverman":
        # Silverman factor on standardized data:
        # h = (4/(d+2))^(1/(d+4)) * n^(-1/(d+4))
        silver = (4.0 / (D + 2.0)) ** (1.0 / (D + 4.0)) * N ** (-1.0 / (D + 4.0))
        kde = KernelDensity(kernel="gaussian", bandwidth=silver).fit(Xz)
        log_dens = kde.score_samples(Xz)
        log_p_cand = float(kde.score_samples(cand_z)[0])

    elif method == "bgmm":
        # Retry strategy if the main algorithm doesn't converge
        bgmm = _fit_bgmm(Xz, init_params="kmeans", reg=1e-5, rs=random_state)
        if not getattr(bgmm, "converged_", True):
            bgmm = _fit_bgmm(Xz, init_params="random", reg=1e-4, rs=random_state + 1)
        if not getattr(bgmm, "converged_", True):
            # Final attempt: smaller tol + larger reg
            bgmm = _fit_bgmm(Xz, init_params="random", reg=1e-3, rs=random_state + 2)

        if not getattr(bgmm, "converged_", True):
            # warnings.warn("BGMM did not converge after retries; HDR may be noisy.", RuntimeWarning)
            raise ValueError("BGMM did not converge after retries; HDR may be noisy.")

        log_dens = bgmm.score_samples(Xz)
        log_p_cand = float(bgmm.score_samples(cand_z)[0])

    elif method == "scipy_kde":
        # gaussian_kde uses Scott's rule by default; works fine in 2D
        kde = gaussian_kde(Xz.T, bw_method="scott")
        dens = kde(Xz.T)
        p_cand = float(kde(cand_z.T)[0])
        # guard against zeros
        eps = np.finfo(float).tiny
        log_dens = np.log(dens + eps)
        log_p_cand = np.log(p_cand + eps)

    else:
        raise ValueError(f"Unknown method: {method}")

    # 3) HDR in the log-domain (avoids under/overflow)
    hdr_level = float((log_dens >= log_p_cand).mean())
    return hdr_level

def compute_8d_hdr(
    galactic_binary: GalacticBinary,
    candidate_index: int,
    n_components: int = 30,
    standardize: bool = True,
    weight_concentration_prior:float = 1.0,   # default to 1.0; smaller => fewer active comps
    random_state: int = 42,
):
    """
    Compute the multivariate HDR coverage percentile using a Bayesian GMM.
    Returns the fraction of posterior mass with density ≥ the candidate's density.
    """

    # 1) Get samples and clean
    samples_df = galactic_binary.get_binary_parameters_posterior()
    X = np.asarray(samples_df.values, dtype=float)
    # Drop non-finite rows
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    N, D = X.shape
    if D == 0 or N == 0:
        raise ValueError("Empty or invalid sample matrix after cleaning.")

    # 2) Standardize (priors assume this)
    scaler = StandardScaler().fit(X) if standardize else None
    Xz = scaler.transform(X) if scaler is not None else X

    # 3) Fit BGMM with sensible priors & stability
    def _fit_bgmm(Xz, init_params="kmeans", reg=1e-5, rs=random_state):
        with warnings.catch_warnings():
            model = BayesianGaussianMixture(
                n_components=n_components,                      # upper bound
                covariance_type="full",
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=weight_concentration_prior,
                mean_prior=np.zeros(D),
                mean_precision_prior=1.0,                       # stronger than 0.1
                degrees_of_freedom_prior=D + 2,                 # > D-1
                covariance_prior=np.eye(D),
                reg_covar=reg,
                init_params=init_params,
                n_init=10,
                max_iter=1000,
                tol=1e-3,
                random_state=rs,
            )
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(Xz)
        return model

    bgmm = _fit_bgmm(Xz, init_params="kmeans", reg=1e-5, rs=random_state)

    # Retry with a different init and stronger ridge if needed
    if not getattr(bgmm, "converged_", True):
        bgmm = _fit_bgmm(Xz, init_params="random", reg=1e-4, rs=random_state + 1)

    if not getattr(bgmm, "converged_", True):
        # warnings.warn("BGMM did not converge after retries; HDR may be noisy.", RuntimeWarning)
        raise ValueError("BGMM did not converge after retries; HDR may be noisy.")

    # 4) Log densities for HDR (avoid exp underflow)
    log_dens = bgmm.score_samples(Xz)  # shape (N,)

    # 5) Candidate params → same preprocessing
    candidates_df = galactic_binary.get_binary_parameters_injections()
    candidate = np.asarray(candidates_df.iloc[candidate_index].values, dtype=float).reshape(1, -1)
    candidate_z = scaler.transform(candidate) if scaler is not None else candidate
    log_p_cand = float(bgmm.score_samples(candidate_z)[0])

    # 6) HDR fraction using log-densities
    hdr_level = float((log_dens >= log_p_cand).mean())

    return hdr_level

def check_in_support(galactic_binary: GalacticBinary, candidate_index: int):
    in_support = {}
    for param in galactic_binary.binary_params:
        # Get samples
        samples = np.asarray(galactic_binary.get_binary_parameters_posterior()[param])
        x_true = galactic_binary.get_binary_parameters_injections().iloc[candidate_index][param]
        if x_true <= np.max(samples) and x_true >= np.min(samples):
            in_support[param] = 1
        else:
            in_support[param] = 0
    return in_support
