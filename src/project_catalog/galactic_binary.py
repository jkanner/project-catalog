import pandas as pd
from pathlib import Path
import numpy as np
from corner import corner

class GalacticBinary:
    """
    Container for a posterior chain and its corresponding injection candidates.
    """
    def __init__(self, name: str, chain: pd.DataFrame, candidates: pd.DataFrame):
        self.name = name
        self.chain = chain
        self.candidates = candidates
        self.binary_params = ['Frequency', 'Amplitude', 'Inclination',
                              'Initial Phase', 'Ecliptic Latitude',
                              'Ecliptic Longitude', 'Polarization',
                              'Frequency Derivative']
        print(f"Loaded {self.name} with {len(self.chain)} samples and {len(self.candidates)} candidates.")

    @classmethod
    def load_feather(cls,
                      name: str,
                      chain_dir: str = None,
                      candidates_dir: str = None):
        """
        Load chain and candidates (SNR > 1) from Feather files.
        name           : base name (e.g. 'LDC0017720857')
        chain_dir      : path to directory containing '*_chain.feather'
        candidates_dir : path to directory containing '*_candidates.feather'
        snr_cut        : whether to look for *_candidates.feather
        """
        if chain_dir is None:
            chain_dir = Path(__file__).parent.parent.parent / "data/posterior_chains/"
        if candidates_dir is None:
            candidates_dir = Path(__file__).parent.parent.parent / "data/injection_candidates/"
        chain_fp = Path(chain_dir) / f"{name}_chain.feather"
        suffix = "_candidates.feather"
        cand_fp = Path(candidates_dir) / f"{name}{suffix}"
        try:
            chain_df = pd.read_feather(chain_fp)
            # TODO (Aaron): Come back and try to use all samples
            if chain_df.shape[0] > 10_000:
                thin_factor = 10
                chain_df = chain_df.iloc[::thin_factor].reset_index(drop=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Chain file {chain_fp} not found.")
        try:
            cand_df  = pd.read_feather(cand_fp)
        except FileNotFoundError:
            print(f"Candidates file {cand_fp} not found. Using empty DataFrame.")
            cand_df = pd.DataFrame(columns=['Frequency', 'Amplitude', 'Inclination',
                                            'Initial Phase', 'Ecliptic Latitude',
                                            'Ecliptic Longitude', 'Polarization',
                                            'Frequency Derivative'])

        # Wrap the polarization angle of candidates to be between 0 and pi (to match chain prior)
        theta = cand_df["Polarization"].to_numpy().squeeze()
        wrapped = np.abs(np.angle(np.exp(1j * theta)))
        cand_df["Polarization"] = wrapped
        return cls(name, chain_df, cand_df)

    def save_feathers(self,
                      chain_dir: str,
                      candidates_dir: str):
        """
        Save chain and candidates to Feather files (resetting their indices).
        """
        Path(chain_dir).mkdir(exist_ok=True, parents=True)
        Path(candidates_dir).mkdir(exist_ok=True, parents=True)

        chain_fp = Path(chain_dir) / f"{self.name}_chain.feather"
        suffix = "_candidates.feather"
        cand_fp = Path(candidates_dir) / f"{self.name}{suffix}"

        # Feather requires a default RangeIndex
        self.chain.reset_index(drop=True).to_feather(chain_fp)
        self.candidates.reset_index(drop=True).to_feather(cand_fp)

    def get_binary_parameters_chain(self) -> pd.DataFrame:
        """
        Return a DataFrame with only the galactic binary parameter columns from the chain.
        """
        return self.chain[self.binary_params]

    def get_binary_parameters_candidates(self) -> pd.DataFrame:
        """
        Return a DataFrame with only the galactic binary parameter columns from the candidates.
        """
        return self.candidates[self.binary_params]
    
    def get_sky_location_chain(self) -> pd.DataFrame:
        """
        Return a DataFrame with the sky location parameters from the chain.
        """
        return self.chain[['Ecliptic Latitude', 'Ecliptic Longitude']]

    def get_sky_location_candidates(self) -> pd.DataFrame:
        """
        Return a DataFrame with the sky location parameters from the candidates.
        """
        return self.candidates[['Ecliptic Latitude', 'Ecliptic Longitude']]

    def corner_plot(self, candidate_index: int, **kwargs):
        """
        Plot the corner plot of the chain.
        """
        fig = corner(self.get_binary_parameters_chain(),
                     truths = self.get_binary_parameters_candidates().iloc[candidate_index],
                     labels = self.binary_params,
                     **kwargs)
        fig.suptitle(self.name)
        return fig
