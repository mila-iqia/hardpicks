import numpy as np
import torch


class FakeEncoder(torch.nn.Module):

    def __init__(self, hyper_params):
        super().__init__()
        self.input_patch_size = hyper_params["input_patch_size"]
        coordch = int(hyper_params["use_coord_channels"]) * 3
        self.out_channels = int(coordch + np.prod(hyper_params["input_patch_size"]))

    def forward(self, input_tensor):
        assert input_tensor.ndim == 4
        assert input_tensor.shape[-2:] == self.input_patch_size
        out = input_tensor.reshape(len(input_tensor), -1)
        return out.reshape((*out.shape, 1, 1))
