import os
import corner
import matplotlib.pyplot as plt
from KDEpy.FFTKDE import FFTKDE
import numpy as np
from project_catalog.galactic_binary import GalacticBinary
from coppuccino.hdr import compute_injection_hdr
import jax


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


def compute_6d_hdr_all_injections(galactic_binary: GalacticBinary, plot_dir: str|None = None):
    parameters_6d = ['Frequency', 'Amplitude', 'Inclination',
                     'Ecliptic Latitude', 'Ecliptic Longitude',
                     'Frequency Derivative']
    # Get samples
    samples_array = galactic_binary.get_posterior()[parameters_6d].to_numpy()
    injected_array = galactic_binary.get_injections()[parameters_6d].to_numpy()

    if plot_dir is not None:
        hdrs, flow = compute_injection_hdr(samples_array, injected_array, return_flow=True)
        os.makedirs(plot_dir, exist_ok=True)

        labels = list(galactic_binary.binary_params)
        for i in range(injected_array.shape[0]):
            injected_values = injected_array[i]
            # If flow samples are available, overlay them; otherwise plot only samples_array
            fig = corner.corner(
                samples_array,
                labels=labels,
                color='C0',
                hist_kwargs={'density': True},
                label_kwargs={'fontsize': 8},
                truths=injected_values
            )

            if flow is not None:
                key = jax.random.key(990)
                flow_samples = flow.sample(key, (10_000,))
                corner.corner(
                    np.asarray(flow_samples),
                    fig=fig,
                    labels=labels,
                    color='C1',
                    hist_kwargs={'density': True},
                    label_kwargs={'fontsize': 8},
                )
                fig.suptitle('Posterior samples (blue) vs Flow samples (orange) HDR: {:.3f}'.format(hdrs[i]))
            else:
                fig.suptitle('Posterior samples')

            out_path = os.path.join(plot_dir, f'corner_samples_vs_flow_6d_{galactic_binary.name}_injection_idx_{i}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    else:
        hdrs = compute_injection_hdr(samples_array, injected_array)

    return hdrs


def compute_8d_hdr_all_injections(galactic_binary: GalacticBinary, plot_dir: str|None = None):
    # Get samples
    samples_array = galactic_binary.get_binary_parameters_posterior().to_numpy()
    injected_array = galactic_binary.get_binary_parameters_injections().to_numpy()

    if plot_dir is not None:
        hdrs, flow = compute_injection_hdr(samples_array, injected_array, return_flow=True)
        os.makedirs(plot_dir, exist_ok=True)

        labels = list(galactic_binary.binary_params)
        for i in range(injected_array.shape[0]):
            injected_values = injected_array[i]
            # If flow samples are available, overlay them; otherwise plot only samples_array
            fig = corner.corner(
                samples_array,
                labels=labels,
                color='C0',
                hist_kwargs={'density': True},
                label_kwargs={'fontsize': 8},
                truths=injected_values
            )

            if flow is not None:
                key = jax.random.key(990)
                flow_samples = flow.sample(key, (10_000,))
                corner.corner(
                    np.asarray(flow_samples),
                    fig=fig,
                    labels=labels,
                    color='C1',
                    hist_kwargs={'density': True},
                    label_kwargs={'fontsize': 8},
                )
                fig.suptitle('Posterior samples (blue) vs Flow samples (orange) HDR: {:.3f}'.format(hdrs[i]))
            else:
                fig.suptitle('Posterior samples')

            out_path = os.path.join(plot_dir, f'corner_samples_vs_flow_8d_{galactic_binary.name}_injection_idx_{i}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    else:
        hdrs = compute_injection_hdr(samples_array, injected_array)

    return hdrs
