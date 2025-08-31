import pandas as pd
from pathlib import Path
import numpy as np
from corner import corner

binary_parameters = ['Frequency', 'Amplitude', 'Inclination',
                              'Initial Phase', 'Ecliptic Latitude',
                              'Ecliptic Longitude', 'Polarization',
                              'Frequency Derivative']
sky_location_parameters = ['Ecliptic Latitude', 'Ecliptic Longitude']

def wrap_polarization_angle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrap the polarization angle to be between 0 and pi.
    """
    theta = df["Polarization"].to_numpy().squeeze()
    wrapped = np.abs(np.angle(np.exp(1j * theta)))
    df["Polarization"] = wrapped
    return df


class GalacticBinary:
    """
    Container for a posterior chain samples and their corresponding possible injection matches.
    """
    def __init__(self, name: str, posterior: pd.DataFrame, injections: pd.DataFrame):
        self.name = name
        self.posterior = posterior
        self.injections = injections
        self.binary_params = binary_parameters
        print(f"Loaded {self.name} with {len(self.posterior)} samples and {len(self.injections)} possible injection matches.")

    @classmethod
    def load_feather(cls,
                     name: str,
                     posterior_dir: Path|None = None,
                     injections_dir: Path|None = None):
        """
        Load posterior samples and possible injection matches (SNR > 1) from Feather files.
        name : base name (e.g. 'LDC0017720857')
        posterior_dir : path to directory containing '*_posterior.feather'
        _dir : path to directory containing '*_injections.feather'
        """
        if posterior_dir is None:
            posterior_dir = Path(__file__).parent.parent.parent / "data/posterior_chains/"
        if injections_dir is None:
            injections_dir = Path(__file__).parent.parent.parent / "data/injection_matches/"
        posterior_fp = Path(posterior_dir) / f"{name}_posterior.feather"
        suffix = "_injections.feather"
        injections_fp = Path(injections_dir) / f"{name}{suffix}"
        try:
            posterior_df = pd.read_feather(posterior_fp)
            # TODO (Aaron): Come back and try to use all samples
            if posterior_df.shape[0] > 10_000:
                thin_factor = 10
                posterior_df = posterior_df.iloc[::thin_factor].reset_index(drop=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Posterior file {injections_fp} not found.")
        try:
            injections_df  = pd.read_feather(injections_fp)
        except FileNotFoundError:
            print(f"Injections file {injections_fp} not found. Using empty DataFrame.")
            injections_df = pd.DataFrame(columns=binary_parameters + ['Name', 'SNR'])

        # Wrap the polarization angle of injections to be between 0 and pi (to match posterior prior)
        injections_df = wrap_polarization_angle(injections_df)
        return cls(name, posterior_df, injections_df)

    def save_feathers(self,
                      posterior_dir: str,
                      injections_dir: str):
        """
        Save posterior and potential injections matches to Feather files (resetting their indices).
        """
        Path(posterior_dir).mkdir(exist_ok=True, parents=True)
        Path(injections_dir).mkdir(exist_ok=True, parents=True)

        posterior_fp = Path(posterior_dir) / f"{self.name}_posterior.feather"
        suffix = "_injections.feather"
        injections_fp = Path(injections_dir) / f"{self.name}{suffix}"

        # Feather requires a default RangeIndex
        self.posterior.reset_index(drop=True).to_feather(posterior_fp)
        self.injections.reset_index(drop=True).to_feather(injections_fp)

    def get_binary_parameters_posterior(self) -> pd.DataFrame:
        """
        Return a DataFrame with only the galactic binary parameter columns from the posterior.
        """
        return self.posterior[self.binary_params]

    def get_binary_parameters_injections(self) -> pd.DataFrame:
        """
        Return a DataFrame with only the galactic binary parameter columns from the possible injection matches.
        """
        return self.injections[self.binary_params]
    
    def get_sky_location_posterior(self) -> pd.DataFrame:
        """
        Return a DataFrame with the sky location parameters from the posterior.
        """
        return self.posterior[sky_location_parameters]

    def get_sky_location_injections(self) -> pd.DataFrame:
        """
        Return a DataFrame with the sky location parameters from the possible injection matches.
        """
        return self.injections[sky_location_parameters]

    def corner_plot(self, injection_index: int, **plot_kwargs):
        """
        Show the corner plot of the posterior.
        """
        fig = corner(self.get_binary_parameters_posterior(),
                     truths = self.get_binary_parameters_injections().iloc[injection_index],
                     labels = self.binary_params,
                     **plot_kwargs)
        fig.suptitle(self.name)
        return fig
