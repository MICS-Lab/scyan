import wandb
import matplotlib.pyplot as plt
import io
from PIL import Image


def wandb_plt_image(fun, figsize=[7, 5]):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    fun()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    return wandb.Image(Image.open(img_buf))
