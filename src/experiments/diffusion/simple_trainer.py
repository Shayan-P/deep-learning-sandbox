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
from src.dataset import SimpleNormal2D, GMM2D, Spiral2D
from src.visualization import plot_vector_field
from src.utils import get_pts_mesh
from src.visualization import colorful_curve, plot_mean_and_std, plot_image, plot_line
from src.experiments.diffusion.model import PortableDiffusionModel, NoisePredictor


MAX_BATCH_SIZE = 10000

def run_batched(func, *args, **kwargs):
	func = partial(func, **kwargs)
	batched_func = batchify_function(func, batch_size=MAX_BATCH_SIZE)
	return batched_func(*args)


def visualize_config(ddpm: PortableDiffusionModel, logger: Logger):
	rng = slice(0, ddpm._n_steps - 10) # note this is not the whole range
	plot_line(ddpm._betas[rng])
	plot_line(ddpm._alphas[rng])
	plot_line(ddpm._alphas_cumprod[rng])
	plot_line(ddpm._sqrt_alphas_cumprod[rng])
	plot_line(ddpm._sqrt_recip_alphas[rng])
	plot_line(ddpm._sqrt_recip_alphas_cumprod[rng])
	plot_line(ddpm._sqrt_recipm1_alphas_cumprod[rng])
	plot_line(ddpm._sqrt_recipm1_alphas_cumprod_custom[rng])
	plt.legend([
		"betas",
		"alphas",
		"alphas_cumprod",
		"sqrt_alphas_cumprod",
		"sqrt_recip_alphas",
		"sqrt_recip_alphas_cumprod",
		"sqrt_recipm1_alphas_cumprod",
	])
	logger.log_plot("scheduler_config")


def train(dataset, dataloader, eval_loader, ddpm: PortableDiffusionModel, logger: Logger, ckpt: Checkpointer, config: ConfigDict):
	losses = []
	grad_norms = []
	grad_max_norms = []

	last_loss = 0.0
	step = 0
	step_per_iter = 100
	batches_per_ema_update = 1 # todo adjust this for better gpu utilization
	first_ema_update = True
	epoch_loss = 0.0

	device = next(ddpm.parameters()).device

	optimizer = torch.optim.Adam(ddpm.parameters(), lr=config.lr)

	EMA = 0.999
	ema_ddpm = copy.deepcopy(ddpm)
	ema_ddpm.load_state_dict({k: v.clone() for k, v in ddpm.state_dict().items()})

	for epoch in range(config.train_epochs):
		with tqdm(dataloader) as pbar:
			for x0 in pbar:
				x0 = x0.to(device, non_blocking=True)
				loss = ddpm.loss(x=x0)
				loss = loss.mean()
				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0) # should we do this?
				optimizer.step()

				# log
				loss = loss.item()
				epoch_loss += loss
				grad_norm = torch.norm(torch.stack([p.grad.flatten().norm() for p in ddpm.parameters() if p.grad is not None])).item()
				max_grad_norm = torch.max(torch.stack([p.grad.flatten().norm() for p in ddpm.parameters() if p.grad is not None])).item()
				grad_norms.append(grad_norm)
				grad_max_norms.append(max_grad_norm)

				pbar.set_description(f'loss: {loss:.2e}')
				step += 1
				last_loss += loss
				if step % step_per_iter == 0:
					losses.append(last_loss / step_per_iter)
					last_loss = 0.0
				
				# update EMA
				if step % batches_per_ema_update == 0:
					if first_ema_update:
						ema_ddpm.load_state_dict({k: v.clone() for k, v in ddpm.state_dict().items()})
						first_ema_update = False
					else:
						# this is a bottleneck of not utilizing gpu well
						ema_state_dict = ema_ddpm.state_dict()
						ddpm_state_dict = ddpm.state_dict()
						ema_ddpm.load_state_dict({
							k: v * EMA + ddpm_state_dict[k].clone() * (1 - EMA)
							for k, v in ema_state_dict.items()
						})


		epoch_loss /= len(dataloader)
		logger.log_step(epoch)
		logger.log(epoch_loss=epoch_loss)

		plt.figure(figsize=(7, 5))
		plt.plot(losses)
		show_plot("loss")

		grad_norms = grad_norms[-2000:]
		plt.plot(grad_norms)
		grad_max_norms = grad_max_norms[-2000:]
		plt.plot(grad_max_norms)
		plt.legend(["grad_norm", "grad_max_norm"])
		show_plot('grad')
		logger.log(grad_norm=grad_norm, max_grad_norm=max_grad_norm)

		if epoch % 30 == 0:
			ckpt.save_model(ddpm)

			if hasattr(dataset, 'analytic_score'):
				plot_logpx(ema_ddpm, logger, dataset.analytic_score, lx=-2, rx=2, ly=-2, ry=2, nx=30, ny=30)
			else:
				print("no analytical score")

			x0 = next(iter(eval_loader))
			x0 = x0.to(device)
			eval_model(ema_ddpm, logger, x0)
			# add eval loader later...



def eval_model(ddpm: PortableDiffusionModel, logger: Logger, x0):
	print("doing eval...")
	samples, samples_path = ddpm.sample(100, include_path=True)
	samples = samples.detach().cpu().numpy()
	samples_path = samples_path.detach().cpu().numpy()

	samples = samples.clip(-2, 2)
	samples_path = samples_path.clip(-2, 2)

	plt.figure(figsize=(7, 5))
	plt.scatter(samples[:, 0], samples[:, 1])

	for i in range(1): # increase later if you want to see the paths...
		lc = colorful_curve(samples_path[:, i, 0], samples_path[:, i, 1])
	plt.colorbar(lc)
	logger.log_plot("samples")

	# todo add more stuff here later...
	fig, axs = plt.subplots(4, 2, figsize=(10, 20))
	ts = torch.tensor([1, 5, 10, 50, 100, 300, 500, 900]) / 1000 * ddpm._n_steps
	ts = ts.round().long().tolist()
	axs = axs.reshape(-1)
	device = next(ddpm.parameters()).device

	lx=-2
	rx=2
	ly=-2
	ry=2
	nx=20
	ny=20
	pts = get_pts_mesh(lx, rx, ly, ry, nx, ny).to(device)
	x0_cpu = x0.detach().cpu().numpy()

	# todo fix the weighting of mu later...
	for i, t in enumerate(ts):
		t_ = torch.tensor(t, device=device, dtype=torch.long).repeat(pts.shape[0])
		mean, var, log_var = ddpm.p_mean_variance(pts, t_)
		mus_pred = mean
		plot_vector_field(pts, mus_pred - pts, ax=axs[i], arrow_color='red')
		axs[i].set_title(f"t={t}")
	logger.log_plot("fields_grid")

	for loss_type in ['kl', 'simple']:
		mc = 200
		b = pts.shape[0]
		t_start = 1
		t_end = ddpm._n_steps-1
		t_cnt = t_end - t_start + 1
		ts_ = eo.repeat(torch.arange(t_start, t_end+1, device=device), 't -> (b t mc)', b=b, mc=mc)
		x0_ = eo.repeat(pts, 'b d -> (b t mc) d', t=t_cnt, mc=mc)
		loss_fn = ddpm.p_loss_simple if loss_type == 'simple' else ddpm.p_loss_kl
		with torch.no_grad():
			losses = run_batched(loss_fn, x0_, ts_)
		losses = eo.rearrange(losses, '(b t mc) -> t b mc', b=b, t=t_cnt, mc=mc)
		losses_mean = losses.mean(dim=-1).mean(dim=-1)
		losses_std = losses.std(dim=-1).mean(dim=-1).div(torch.sqrt(torch.tensor(mc)))
		plot_mean_and_std(range(t_start, t_end+1), losses_mean, losses_std)
		logger.log_plot(f"loss_{loss_type}_t_variance")
	
	### Check T=0
	# loss_t_1 = ddpm.p_loss_simple(
	# 	pts, 
	# 	torch.ones(pts.shape[0], device=device, dtype=torch.long) * 1
	# )
	# print("loss at t=1: ", loss_t_1.mean())


def plot_logpx(ddpm: PortableDiffusionModel, logger: Logger, analytic_score, lx, rx, ly, ry, nx, ny):
	pts = get_pts_mesh(lx, rx, ly, ry, nx, ny)
	pts = pts.to(next(ddpm.parameters()).device)
	logpx = ddpm.logpx(pts)['logpx'].detach().cpu().numpy()
	logpx_std = ddpm.logpx(pts)['logpx_std'].detach().cpu().numpy()
	prob_integral = np.exp(logpx).sum() * (rx - lx) * (ry - ly) / (nx * ny)
	pts = pts.detach().cpu().numpy()

	fig, axs = plt.subplots(3, 2, figsize=(10, 14))
	axs = axs.reshape(-1)

	score = analytic_score(pts)
	title = "Analytic Score"
	sanity_check_prob = np.exp(score).sum() * (rx - lx) * (ry - ly) / (nx * ny)
	title += f"\nSanity check prob: {sanity_check_prob:.2e}"

	im = plot_image(score, ax=axs[0], nx=nx, ny=ny, lx=lx, ly=ly, rx=rx, ry=ry)
	plt.colorbar(im, ax=axs[0])
	sample = ddpm.sample(1000).detach().cpu().numpy()
	sample_masked = sample[(sample[:, 0] > lx) & (sample[:, 0] < rx) & (sample[:, 1] > ly) & (sample[:, 1] < ry)]
	axs[0].plot(sample_masked[:, 0], sample_masked[:, 1], 'k.', markersize=1)
	axs[0].set_title(title)

	im = plot_image(logpx, ax=axs[1], nx=nx, ny=ny, lx=lx, ly=ly, rx=rx, ry=ry)
	plt.colorbar(im, ax=axs[1])
	axs[1].set_title(f"Estimated NLL: Prob integral: {prob_integral:.2e}")

	mid_y_idx = ny // 2
	xs_idx = pts.reshape((nx, ny, 2))[:, mid_y_idx, 0]
	axs[2].plot(xs_idx, score.reshape((nx, ny))[:, mid_y_idx])
	axs[2].set_title("1D GC Score along Y=0")

	neg_vlb_mean_line = logpx.reshape((nx, ny))[:, mid_y_idx]
	neg_vlb_std_line = logpx_std.reshape((nx, ny))[:, mid_y_idx]
	axs[3].plot(xs_idx, neg_vlb_mean_line)
	axs[3].set_title("1D Estimate along Y=0")
	axs[3].fill_between(xs_idx, neg_vlb_mean_line - neg_vlb_std_line, neg_vlb_mean_line + neg_vlb_std_line, alpha=0.2)

	im = plot_image(logpx_std, ax=axs[4], nx=nx, ny=ny, lx=lx, ly=ly, rx=rx, ry=ry)
	plt.colorbar(im, ax=axs[4])
	axs[4].set_title("STD NLL")

	im = plot_image(logpx - np.log(prob_integral), ax=axs[5], nx=nx, ny=ny, lx=lx, ly=ly, rx=rx, ry=ry)
	plt.colorbar(im, ax=axs[5])
	axs[5].set_title("Estimated and Normalized NLL")
	logger.log_plot("nll_comparison")


def run_experiment(config: ConfigDict):
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

	if config.dataset == 'simple_gaussian':
		dataset = SimpleNormal2D(n=10000, std=1.0)
	elif config.dataset == 'mixture_gaussian':
		dataset = GMM2D(each=10000, std=0.1)
	elif config.dataset == 'spiral':
		dataset = Spiral2D(n=10000)
	else:
		raise ValueError(f"Dataset {config.dataset} not found")
	
	dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
	eval_loader = DataLoader(dataset, batch_size=256, shuffle=True)

	ckpt = Checkpointer(name=config.experiment_name)
	if config.resume:
			ckpt.load_model(ddpm)

	visualize_config(ddpm, logger)
	eval_model(ddpm, logger, next(iter(eval_loader)).to(device))
	train(dataset, dataloader, eval_loader, ddpm, logger, ckpt, config)

	logger.log_end_run()


def main():
	# default config
	config = ConfigDict(dict(
		train_epochs=200,
		resume=False, # change to True if you want to resume training
		experiment_name="portable_diffusion_test",
		lr=1e-6,
		loss_type='kl',
		mc_loss=False,
		var_type='beta_forward',
		use_wandb=True,
		# dataset='simple_gaussian',
		dataset='spiral',
	))
	run_experiment(config)


def sweep():
	base_config = ConfigDict(dict(
		train_epochs=200,
		resume=False, # change to True if you want to resume training
		experiment_name="sweep",
		mc_loss=False,
		use_wandb=True,
	))
	done = 0
	cnt = 0
	for lr in [5e-5, 1e-4, 1e-3]:
		for loss_type in ['kl', 'simple']:
			for var_type in ['beta_forward', 'beta_reverse', 'learned']:
				for dataset in ['simple_gaussian', 'mixture_gaussian', 'spiral']:
					cnt += 1
					if cnt <= done:
						continue
					config = ConfigDict(base_config.to_dict() | dict(
						lr=lr,
						loss_type=loss_type,
						var_type=var_type,
						dataset=dataset,
						experiment_name=f"{base_config.experiment_name}_lr={lr:.1e}_loss_type={loss_type}_var_type={var_type}_dataset={dataset}",
					))
					run_experiment(config)

if __name__ == "__main__":
	main()
