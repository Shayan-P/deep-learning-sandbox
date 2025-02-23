import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import einops as eo
import numpy as np

from src.utils import batchify_function
from functools import partial
from src.utils import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from ml_collections import ConfigDict
from src.utils import bb, Checkpointer, show_plot
from src.visualization import plot_vector_field, plot_scatter
from src.utils import get_pts_mesh
from src.visualization import colorful_curve, plot_mean_and_std, plot_image, plot_line, plot_scatter
from src.experiments.diffusion.model import PortableDiffusionModel, NoisePredictor
from src.experiments.diffusion.simple_trainer import train, visualize_config, eval_model
from src.dataset import DatasetWrapper, GMMDistributionMultivariate, GMMOnCircle2D, Spiral2D


def main():
	for name, dataset in [
		("mixture_gaussian_std=0.1", DatasetWrapper(n=10000, dist=GMMOnCircle2D(n=6, std=0.1))),
		("mixture_gaussian_std=0.4", DatasetWrapper(n=10000, dist=GMMOnCircle2D(n=6, std=0.4))),
		("mixture_gaussian_std=0.8", DatasetWrapper(n=10000, dist=GMMOnCircle2D(n=6, std=0.8))),
		("spiral", Spiral2D(n=10000)),
	]:
		config = ConfigDict(dict(
			train_epochs=150,
			resume=False,
			train_if_exists=True,
			experiment_name=f"langevin_optimizer_dataset={name}",
			lr=1e-3,
			loss_type='kl',
			mc_loss=False,
			var_type='beta_forward',
			use_wandb=True,
		))
		run_experiment(dataset, config)


def run_experiment(dataset, config: ConfigDict):
	logger = Logger(config.experiment_name, config.use_wandb, config=config)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	net = NoisePredictor(dim=2, fourier_features_dim=128).to(device)
	ddpm = PortableDiffusionModel(dim=2,
									n_steps=200,
									net=net,
									var_type=config.var_type,
									mc_loss=config.mc_loss,
									loss_type=config.loss_type,
									).to(device)

	dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
	eval_loader = DataLoader(dataset, batch_size=256, shuffle=True)

	ckpt = Checkpointer(name=config.experiment_name)
	if config.resume:
		ckpt.load_model(ddpm)
	
	if ckpt.any_checkpoint_exists() and (not config.train_if_exists):
		print("Checkpoint found, skipping training")
	else:
		visualize_config(ddpm, logger)
		# eval_model(ddpm, logger, next(iter(eval_loader)).to(device))
		train(dataset, dataloader, eval_loader, ddpm, logger, ckpt, config)

	## experiment starts here
	############################################################

	# eval_model(ddpm, logger, next(iter(eval_loader)).to(device))

	n = 1000
	samples = ddpm.sample(n=n)
	plot_scatter(samples)
	logger.log_plot("samples")

	# repeating the last denoising step
	fig, axs = plt.subplots(3, 3, figsize=(10, 10))
	axs = axs.reshape(-1)

	x = samples

	per_step = 20
	lamda = torch.tensor(1.0, device=device)
	for i in range(9):
		plot_scatter(x, ax=axs[i])
		axs[i].set_title(f"step {i * per_step} / lambda={lamda.item():.2f}")
		for j in range(per_step):
			t = torch.ones(n, dtype=torch.long, device=device) * 0 # repeat 0
			mean, var, log_var = ddpm.p_mean_variance(x, t, clip=10)
			noise = torch.randn_like(x)
			x = x + (mean - x) * lamda + torch.exp(0.5 * log_var) * noise

			# x = ddpm.p_sample(x, t, clip=10)

		lamda *= 1.7
	logger.log_plot("samples_repeated_denoising")

	logger.log_end_run()


if __name__ == "__main__":
	main()
