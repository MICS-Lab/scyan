import torch
from torch import Tensor
from typing import Callable, List


def gaussian_kernel(scale: float) -> Callable:
    return lambda dist_squared: torch.exp(-dist_squared / scale)


class LossMMD:
    def __init__(self, n_markers: int, prior_std: float, mean_na: float):
        self.scales = self.get_heuristic_scales(n_markers, prior_std, mean_na)
        self.kernels = [gaussian_kernel(scale) for scale in self.scales]

    def get_heuristic_scales(
        self, n_markers: int, prior_std: float, mean_na: float
    ) -> List[float]:
        internal_scale = 2 * n_markers * prior_std ** 2
        adjacent_modes_scale = internal_scale + 2 * mean_na / 3 + 4
        return [internal_scale, adjacent_modes_scale]

    def one_kernel_mmd(
        self, kernel: Callable, dxx: Tensor, dxy: Tensor, dyy: Tensor
    ) -> Tensor:
        return (kernel(dxx) - 2 * kernel(dxy) + kernel(dyy)).mean()

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        xx, yy, xy = x.mm(x.T), y.mm(y.T), x.mm(y.T)

        x2 = (x ** 2).sum(dim=-1)
        y2 = (y ** 2).sum(dim=-1)

        dxx = x2[:, None] + x2[None, :] - 2 * xx
        dxy = x2[:, None] + y2[None, :] - 2 * xy
        dyy = y2[:, None] + y2[None, :] - 2 * yy

        return sum(self.one_kernel_mmd(kernel, dxx, dxy, dyy) for kernel in self.kernels)
