import pytorch_lightning


def set_seed(
        seed,
        set_deterministic: bool = False,  # False = optimal performance
        set_benchmark: bool = True,  # True = optimal performance, unless input tensor shapes vary
):  # pragma: no cover
    """Set the provided seed in python/numpy/DL framework."""
    pytorch_lightning.seed_everything(seed)
    import torch.backends.cudnn
    torch.backends.cudnn.deterministic = set_deterministic
    torch.backends.cudnn.benchmark = set_benchmark
