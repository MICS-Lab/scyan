import torch
from torch import Tensor


def energy_kernel(eps: float = 1e-8) -> Tensor:
    return lambda d2: -torch.sqrt(d2.clamp(eps))


def gaussian_kernel(std: float = 0.5) -> Tensor:
    return lambda d2: torch.exp(-d2 / (2 * std ** 2))


def inverse_multiquadratic(C: float = 1.0) -> Tensor:
    return lambda d2: (C / (C + d2))


kernel_dict = {
    "energy": energy_kernel,
    "gaussian": gaussian_kernel,
    "inverse_multiquadratic": inverse_multiquadratic,
}


class LossMMD:
    def __init__(self, kernel: str = "energy", **kwargs):
        if kernel in kernel_dict:
            self.kernel = kernel_dict[kernel](**kwargs)
        else:
            raise NameError(
                f"MMD kernel has to be one of {list(kernel_dict.keys())}, found {kernel}"
            )

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        xx, yy, xy = x.mm(x.T), y.mm(y.T), x.mm(y.T)

        x2 = (x * x).sum(dim=-1)
        y2 = (y * y).sum(dim=-1)

        dxx = x2[:, None] + x2[None, :] - 2 * xx
        dxy = x2[:, None] + y2[None, :] - 2 * xy
        dyy = y2[:, None] + y2[None, :] - 2 * yy

        return (self.kernel(dxx) - 2 * self.kernel(dxy) + self.kernel(dyy)).mean()
