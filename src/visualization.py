import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from src.visualization import colorful_curve, plot_image, plot_scatter
from src.utils import show_plot


def colorful_curve(xs, ys):
    if isinstance(xs, torch.Tensor):
        xs = xs.detach().cpu().numpy()
    if isinstance(ys, torch.Tensor):
        ys = ys.detach().cpu().numpy()
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='cool', norm=plt.Normalize(0, 1), alpha=0.5)
    lc.set_array(np.linspace(0, 1, len(xs)))
    lc.set_linewidth(2)
    plt.gca().add_collection(lc)
    return lc

def plot_mean_and_std(idxs, mean, std):
    if isinstance(idxs, torch.Tensor):
        idxs = idxs.detach().cpu().numpy()
    if isinstance(mean, torch.Tensor):
        mean = mean.detach().cpu().numpy()
    if isinstance(std, torch.Tensor):
        std = std.detach().cpu().numpy()
    plt.plot(idxs, mean)
    plt.fill_between(idxs, mean - std, mean + std, alpha=0.2)


def plot_image(image, ax=None, nx=None, ny=None, lx=None, ly=None, rx=None, ry=None):
    if ax is None:
        ax = plt.gca()
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if (nx is not None and ny is not None):
        image = np.rot90(image.reshape(nx, ny))
    kwargs = {}
    if (lx is not None and ly is not None and rx is not None and ry is not None):
        extent=[lx, rx, ly, ry]
        kwargs['extent'] = extent
    im = ax.imshow(image, **kwargs)
    plt.colorbar(im, ax=ax)
    return im

def plot_line(ys):
    if isinstance(ys, torch.Tensor):
        ys = ys.detach().cpu().numpy()
    plt.plot(ys)

def plot_scatter(pts, ax=None, color="black", s=0.2):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    if ax is None:
        ax = plt.gca()
    ax.scatter(pts[:, 0], pts[:, 1], color=color, s=s)

def plot_vector_field(pos, vec, scatter_pos=False, ax=None, title=None, arrow_color='black', scale=5):
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if isinstance(vec, torch.Tensor):
        vec = vec.detach().cpu().numpy()
    def get_stats():
        vec_len = vec.norm(dim=-1)
        print("max norm: ", vec_len.max())
        print("min norm: ", vec_len.min())
        print("mean norm: ", vec_len.mean())
    # bb("plot_vector_field", get_stats, ignore=True)

    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if isinstance(vec, torch.Tensor):
        vec = vec.detach().cpu().numpy()
    ax_plt = plt if ax is None else ax
    if scatter_pos:
        ax_plt.scatter(pos[:, 0], pos[:, 1])
    ax_plt.quiver(pos[:, 0], pos[:, 1], vec[:, 0], vec[:, 1], color=arrow_color, angles='xy', scale_units='xy', scale=1/scale)
    # ax_plt.quiver(pos[:, 0], pos[:, 1], vec[:, 0], vec[:, 1], color=arrow_color)
    ax_plt.axis('equal')
    set_title = lambda ax: ax.set_title(title) if ax else plt.title(title)
    if title:
        set_title(title)
