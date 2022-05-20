import torch
from torch import Tensor


def gaussian_kernel(scale: float = 0.5) -> Tensor:
    return lambda dist_squared: torch.exp(-dist_squared / (2 * scale ** 2))


class LossMMD:
    scales = [0.01, 0.1, 1.0, 10.0]

    def __init__(self):
        self.kernels = [gaussian_kernel(scale) for scale in self.scales]

    def one_kernel_mmd(self, kernel, dxx, dxy, dyy):
        return (kernel(dxx) - 2 * kernel(dxy) + kernel(dyy)).mean()

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        xx, yy, xy = x.mm(x.T), y.mm(y.T), x.mm(y.T)

        x2 = (x * x).sum(dim=-1)
        y2 = (y * y).sum(dim=-1)

        dxx = x2[:, None] + x2[None, :] - 2 * xx
        dxy = x2[:, None] + y2[None, :] - 2 * xy
        dyy = y2[:, None] + y2[None, :] - 2 * yy

        return sum(
            [self.one_kernel_mmd(kernel, dxx, dxy, dyy) for kernel in self.kernels]
        )
