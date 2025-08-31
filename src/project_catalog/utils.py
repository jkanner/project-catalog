import glob
import numpy as np
import pandas as pd
from project_catalog import GalacticBinary

def get_galactic_binary_names():
    """
    Get the names of all galactic binaries from feather files in the data directory.
    """
    files = glob.glob("../data/posterior_chains/*.feather")
    names = [f.split("/")[-1].split(".")[0].split("_")[0] for f in files]
    return names

def check_injection_match(galactic_binary: GalacticBinary, injection_index: int, hdr_dataframe: pd.DataFrame, threshold:float = 0.9):
    # threshold is the maximum allowed HDR (1 - alpha)%
    alpha = 1.0 - threshold
    # correct for multiple 1D trials (8 parameters)
    threshold = multiple_trial_correction(alpha, ntrials=8)

    hdr_params = ['Frequency HDR', 'Amplitude HDR', 'Inclination HDR',
                   'Initial Phase HDR', 'Ecliptic Latitude HDR',
                   'Ecliptic Longitude HDR', 'Polarization HDR',
                   'Frequency Derivative HDR']
    name = galactic_binary.name
    injection_name = galactic_binary.injections["Name"][injection_index]
    masked_df = hdr_dataframe[(hdr_dataframe["Name"] == name) & (hdr_dataframe["Candidate"] == injection_name)][hdr_params]
    if masked_df.empty is True:
        print(name, injection_name)
        return False
    if np.all(masked_df.iloc[0].to_numpy() < threshold):
        return True
    else:
        return False

def check_sky_location_support(galactic_binary: GalacticBinary, injection_index: int, hdr_dataframe: pd.DataFrame, threshold:float = 0.9):
    # threshold is the maximum allowed HDR (1 - alpha)%
    alpha = 1.0 - threshold
    # correct for multiple 1D trials (8 parameters)
    threshold = multiple_trial_correction(alpha, ntrials=8)

    hdr_params = ['Sky Location HDR']
    name = galactic_binary.name
    injection_name = galactic_binary.injections["Name"][injection_index]
    masked_df = hdr_dataframe[(hdr_dataframe["Name"] == name) & (hdr_dataframe["Candidate"] == injection_name)][hdr_params]
    if masked_df.empty is True:
        print(name, injection_name)
        return False
    if np.all(masked_df.iloc[0].to_numpy() < threshold):
        return True
    else:
        return False

def multiple_trial_correction(alpha: float, ntrials: int = 8):
    """
    Apply multiple trial (Bonferroni) correction to a given alpha value.

    Parameters:
    ----------
    alpha (float): The original alpha value.
    ndims (int): The number of dimensions (default is 8).

    Returns:
    -------
    float: The corrected alpha value.
    """
    return 1 - (alpha / ntrials)
