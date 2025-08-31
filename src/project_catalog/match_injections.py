import pandas as pd
from tqdm import tqdm
from project_catalog.utils import check_injection_match, get_galactic_binary_names
from project_catalog.galactic_binary import GalacticBinary


def get_all_matches(hdrs: pd.DataFrame, threshold: float = 0.89):
    """
    Iterate through all galactic binaries and check which injections have matches to posteriors based on HDR values.
    """
    names = get_galactic_binary_names()
    matches = {}

    for i, name in tqdm(enumerate(names)):
        matches[name] = []
        gb = GalacticBinary.load_feather(name)
        for injection_index in range(len(gb.injections)):
            if check_injection_match(gb, injection_index, hdrs, threshold=threshold):
                matches[name].append(i)
    return matches
