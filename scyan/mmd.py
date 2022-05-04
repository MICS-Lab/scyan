import torch
from torch import Tensor


def energy_kernel(eps: float = 1e-8):
    return lambda d2: -torch.sqrt(d2.clamp(eps)).mean()


def gaussian_kernel(std: float = 0.5):
    return lambda d2: torch.exp(-d2 / (2 * std ** 2)).mean()


def inverse_multiquadratic(C: float = 1.0):
    return lambda d2: (C / (C + d2)).mean()


class LossMMD:
    def __init__(self, kernel: str = "energy", **kwargs):
        if kernel == "energy":
            self.kernel = energy_kernel(**kwargs)
        elif kernel == "gaussian":
            self.kernel = gaussian_kernel(**kwargs)
        elif kernel == "inverse_multiquadratic":
            self.kernel = inverse_multiquadratic(**kwargs)
        else:
            raise NameError(f"kernel {kernel} is not known.")

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        xx, yy, xy = x.mm(x.T), y.mm(y.T), x.mm(y.T)  # TODO: double_grad ?

        x2 = (x * x).sum(dim=-1)
        y2 = (y * y).sum(dim=-1)

        dxx = x2[:, None] + x2[None, :] - 2 * xx
        dxy = x2[:, None] + y2[None, :] - 2 * xy
        dyy = y2[:, None] + y2[None, :] - 2 * yy

        return self.kernel(dxx) - 2 * self.kernel(dxy) + self.kernel(dyy)
