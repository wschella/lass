def setup_rng(seed: int, include_algos: bool = False) -> None:
    import numpy as np
    import torch
    import random
    import os

    random.seed(seed)
    torch.manual_seed(random.randint(1, 1_000_000))
    torch.cuda.manual_seed(random.randint(1, 1_000_000))
    np.random.seed(random.randint(1, 1_000_000))
    os.environ["PYTHONHASHSEED"] = str(random.randint(1, 1_000_000))

    if include_algos:
        import torch.backends.cudnn

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(mode=True)
