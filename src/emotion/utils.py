def set_seed(seed: int) -> None:
    import os
    import random

    import numpy as np
    import polars as pl

    np.random.seed(seed)
    random.seed(seed)
    pl.set_random_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
