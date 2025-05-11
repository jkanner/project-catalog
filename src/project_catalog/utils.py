import glob

def get_galactic_binary_names():
    """
    Get the names of all galactic binaries from feather files in the data directory.
    """
    files = glob.glob("../data/posterior_chains/*.feather")
    names = [f.split("/")[-1].split(".")[0].split("_")[0] for f in files]
    return names
