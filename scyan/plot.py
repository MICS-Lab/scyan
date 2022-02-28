import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .model import Scyan


def matrix_pop_reconstruction(scyan: Scyan, show=True):
    predictions = scyan.predict(key_added=None)
    predictions.name = "population"
    h = scyan.module(scyan.x)[0].detach().numpy()
    df = pd.concat(
        [predictions, pd.DataFrame(h, columns=scyan.marker_pop_matrix.index)], axis=1
    )
    df = df.groupby("population").median()
    sns.heatmap(
        df.loc[scyan.marker_pop_matrix.columns] - scyan.marker_pop_matrix.T,
        cmap="coolwarm",
    )
    if show:
        plt.show()
