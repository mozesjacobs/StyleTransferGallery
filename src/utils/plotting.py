import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import torch
import imageio
import math


# https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
def canvas2rgb_array(canvas):
    """Adapted from: https://stackoverflow.com/a/21940031/959926"""
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()
    scale = int(round(math.sqrt(buf.size / 3 / nrows / ncols)))
    return buf.reshape(scale * nrows, scale * ncols, 3)