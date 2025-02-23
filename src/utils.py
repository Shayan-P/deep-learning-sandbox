import sys
import pdb
import torch
import numpy as np
import os
import imageio
import wandb
from PIL import Image
from ml_collections import ConfigDict

from matplotlib import pyplot as plt
from src.paths import TMP_IMAGES_DIR, CHECKPOINT_DIR
from tqdm import tqdm


def show_plot(name="a", fig=None):
    name = name.replace(" ", '_')
    path = TMP_IMAGES_DIR.joinpath(f"{name}.png")
    if fig is None:
        plt.savefig(path, bbox_inches='tight', dpi=100)
        plt.clf()
    else:
        fig.savefig(path, bbox_inches='tight', dpi=100)
    print("image saved to: " + str(path))
    if fig is None:
        plt.close()
    else:
        plt.close(fig)

def read_plot(name="a"):
    path = TMP_IMAGES_DIR.joinpath(f"{name}.png")
    return plt.imread(path)


def save_gif(images, name="a"):
    path = TMP_IMAGES_DIR.joinpath(f"{name}.gif")
    
    # Get the shape of the first image
    target_shape = images[0].shape[:2]  # Only height and width, ignore channels
    
    # Resize all images to match the first image
    resized_images = []
    for img in images:
        # Convert to uint8 if not already
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if img.shape[:2] != target_shape:
            # Preserve the number of channels by reshaping only height and width
            resized_img = np.array(Image.fromarray(img).resize(
                (target_shape[1], target_shape[0]),  # PIL uses (width, height)
                Image.Resampling.LANCZOS
            ))
            resized_images.append(resized_img)
        else:
            resized_images.append(img)
    
    imageio.mimsave(path, resized_images, duration=100, loop=0)
    print("gif saved to: " + str(path))


IGNORE_BREAKPOINTS = False

# optional breakpoint
def bb(name=None, func=None, ignore=False):
    if IGNORE_BREAKPOINTS or ignore:
        print("ignoring breakpoint ", name)
        return
    if name:
        print(f"Breakpoint: {name}")
    if func:
        func()
    print("calling breakpoint: ")
    frame = sys._getframe(1)  # Get the caller's frame
    pdb.Pdb().set_trace(frame)
    print("Continuing...")


class Checkpointer:
    def __init__(self, name="model"):
        self.checkpoint_dir = CHECKPOINT_DIR
        self.name = name

    def get_path(self):
        return self.checkpoint_dir / f'{self.name}.pth'

    def save_model(self, model):
        path = self.get_path()
        torch.save(model.state_dict(), path)

    def load_model(self, model):    
        path = self.get_path()
        if not os.path.isfile(path):
            print("No checkpoint found at: ", path)
            return
        device = (next(model.parameters())).device
        model.load_state_dict(torch.load(path, map_location=device))


def batchify_function(func, batch_size):
    def wrapper(*args):
        b = args[0].shape[0]
        if b <= batch_size:
            return func(*args)
        res = []
        for i in tqdm(range(0, b, batch_size), desc="batchifying"):
            new_args = [arg[i:i+batch_size] for arg in args]
            sub_res = func(*new_args)
            res.append(sub_res)
        return torch.cat(res)
    return wrapper


def get_pts_mesh(lx=-2, rx=2, ly=-2, ry=2, nx=15, ny=15):
    x = torch.linspace(lx, rx, nx)
    y = torch.linspace(ly, ry, ny)
    x, y = torch.meshgrid(x, y, indexing='ij')
    pts = torch.stack([x, y], dim=-1).reshape(-1, 2)
    return pts


class Logger:
    def __init__(self, experiment_name, use_wandb=True, config=None):
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.step = 0
        self.config = config
        if isinstance(config, ConfigDict):
            self.config = config.to_dict()
        elif not isinstance(config, dict):
            assert config is None, "config must be a ConfigDict or a dict"
        if use_wandb:
            self.run = wandb.init(project="diffusion_likelihood_estimation",
                                    name=experiment_name,
                                    config=self.config,
                                    resume="allow")
    
    def log_step(self, step):
        self.step = step
    
    def log(self, **kwargs):
        # print(f'Step {self.step}: {" ".join([f"{k}={v}" for k, v in kwargs.items()])}')
        if self.use_wandb:
            self.run.log({**kwargs}, step=self.step)
    
    def log_end_run(self):
        if self.use_wandb:
            self.run.finish()

    def log_plot(self, name):
        show_plot(name)
        if self.use_wandb:
            path = TMP_IMAGES_DIR.joinpath(f"{name}.png")
            self.run.log({name: wandb.Image(Image.open(path))}, step=self.step)

    def __del__(self):
        if self.use_wandb:
            self.run.finish()
