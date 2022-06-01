import numpy as np
import pandas as pd
import torch.backends
import torch.cuda


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # make sure that all randomness is deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_from_path(path: str) -> pd.DataFrame:
    """Loads data from a csv file at the given path.

    :param path:  path to the file containing data
    :return:
    """
    df = pd.read_csv(path, sep="\t").drop('id', axis=1)
    return df