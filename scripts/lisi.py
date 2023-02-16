# Updated from Harmonypy - https://github.com/slowkow/harmonypy
#
# LISI - The Local Inverse Simpson Index
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import scyan


def compute_lisi(
    adata,
    label: str,
    obsm_key: Optional[str] = None,
    max_cells: int = 100_000,
    perplexity: float = 30,
):
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata.

    LISI is a statistic computed for each item (row) in the data matrix X.

    The following example may help to interpret the LISI values.

    Suppose one of the columns in metadata is a categorical variable with 3 categories.

        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.

        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.

    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].

    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    """
    # We need at least 3 * n_neigbhors to compute the perplexity
    adata_ = scyan.preprocess.subsample(adata, n_obs=max_cells, copy=True)
    X = adata_.X if obsm_key is None else adata_.obsm[obsm_key]

    knn = NearestNeighbors(n_neighbors=perplexity * 3, algorithm="kd_tree").fit(X)
    distances, indices = knn.kneighbors(X)

    indices = indices[:, 1:]
    distances = distances[:, 1:]
    labels = pd.Categorical(adata_.obs[label])

    simpson = compute_simpson(distances.T, indices.T, labels, perplexity)
    return (1 / simpson).mean()


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    perplexity: float,
    tol: float = 1e-5,
):
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:, i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson
