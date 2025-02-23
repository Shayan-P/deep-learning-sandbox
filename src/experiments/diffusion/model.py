import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import einops as eo
from src.utils import batchify_function
from functools import partial

MAX_BATCH_SIZE = 10000

def run_batched(func, *args, **kwargs):
	func = partial(func, **kwargs)
	batched_func = batchify_function(func, batch_size=MAX_BATCH_SIZE)
	return batched_func(*args)


def cosine_beta_schedule(timesteps, s=0.008):
	steps = timesteps + 1
	x = np.linspace(0, timesteps, steps, dtype=np.float64)
	alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5)**2
	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
	return np.clip(betas, 0, 0.999)


# todo later implement this following VDM paper
class FourierLayer(nn.Module):
    def __init__(self, dim, fourier_features_dim, frequency_range=1.0):
        super().__init__()
        self.dim = dim
        self.fourier_features_dim = fourier_features_dim
        
        assert fourier_features_dim % 2 == 0
        waves = torch.randn(fourier_features_dim // 2, dim) * frequency_range * torch.pi
        self.register_buffer("waves", waves)
        
    def forward(self, x):
        # waves = eo.einsum(x, self.waves, "b d, f d -> b f")
        waves = torch.matmul(x, self.waves.T)
        features = torch.cat([torch.sin(waves), torch.cos(waves)], dim=-1)
        return features


class NoisePredictor(nn.Module):
    def __init__(self, dim, fourier_features_dim=128):
        super().__init__()
        
        self.dim = dim
        # Wider network with residual connections
        self.dims = [fourier_features_dim + dim, 128, 256, 256, 128]
        
        self.fourier_layer = FourierLayer(dim, fourier_features_dim, frequency_range=7)

        # TODO implement a better time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Main layers with residual connections
        self.layers = nn.ModuleList([])
        for i in range(len(self.dims)-1):
            self.layers.append(nn.ModuleDict({
                'linear': nn.Linear(self.dims[i], self.dims[i+1]),
                'norm': nn.LayerNorm(self.dims[i+1]),
                'time_proj': nn.Linear(128, self.dims[i+1])
            }))
            
        self.final = nn.Linear(self.dims[-1], dim)
        
    def forward(self, x, t):
        """
        shape(x): [B, D]
        shape(t): [B]
        """
        b, _ = x.shape
        assert t.shape == (b,)

        fourier_features = self.fourier_layer(x)
        x = torch.cat([x, fourier_features], dim=-1)

        t_embed = self.time_embed(t.unsqueeze(-1))
        
        h = x
        for layer in self.layers:
            h_prev = h
            h = layer['linear'](h)
            h = h + layer['time_proj'](t_embed)
            h = layer['norm'](h)
            h = F.silu(h)
            if h.shape == h_prev.shape:  # Residual if shapes match
                h = h + h_prev
                
        return self.final(h)


class PortableDiffusionModel(nn.Module):
	"""Basic Diffusion Model."""

	def __init__(self,
							 dim,
							 n_steps,
							 net,
							 loss_type='simple',
							 mc_loss=True,
							 var_type='learned',
							 name=None):
		super().__init__()
		self.name = name
		assert var_type in ('beta_forward', 'beta_reverse', 'learned')
		self._var_type = var_type
		self.net = net
		self._n_steps = n_steps
		self._dim = dim
		self._loss_type = loss_type
		self._mc_loss = mc_loss
		
		self.register_buffer("_betas", torch.from_numpy(cosine_beta_schedule(n_steps)).float())
		self.register_buffer("_alphas", 1. - self._betas)
		self.register_buffer("_log_alphas", torch.log(self._alphas))
		
		alphas = 1. - self._betas
		
		self.register_buffer("_sqrt_alphas", torch.sqrt(alphas))
		self.register_buffer("_sqrt_recip_alphas", 1. / torch.sqrt(alphas))
		
		self.register_buffer("_alphas_cumprod", torch.cumprod(self._alphas, dim=0))
		self.register_buffer("_alphas_cumprod_prev", torch.cat([torch.ones(1), self._alphas_cumprod[:-1]]))
		self.register_buffer("_sqrt_alphas_cumprod", torch.sqrt(self._alphas_cumprod))
		self.register_buffer("_sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self._alphas_cumprod))
		self.register_buffer("_log_one_minus_alphas_cumprod", torch.log(1 - self._alphas_cumprod))
		
		self.register_buffer("_sqrt_recip_alphas_cumprod", 1. / torch.sqrt(self._alphas_cumprod))
		self.register_buffer("_sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / self._alphas_cumprod - 1))
		self.register_buffer("_sqrt_recipm1_alphas_cumprod_custom", torch.sqrt(1. / (1 - self._alphas_cumprod)))
		
		# calculations for posterior q(x_{t-1} | x_t, x_0)
		self.register_buffer("_posterior_variance", self._betas * (
				1. - self._alphas_cumprod_prev) / (1. - self._alphas_cumprod))
		
		self.register_buffer("_posterior_log_variance_clipped", torch.log(
				torch.clamp(self._posterior_variance, min=self._betas.min())))
		self.register_buffer("_posterior_mean_coef1", self._betas * torch.sqrt(
				self._alphas_cumprod_prev) / (1 - self._alphas_cumprod))
		self.register_buffer("_posterior_mean_coef2", (1 - self._alphas_cumprod_prev) * torch.sqrt(
				self._alphas) / (1 - self._alphas_cumprod))
		
		self.register_parameter('out_logvar', 
													nn.Parameter(torch.log(self._betas).clone()))

	def energy_scale(self, t):
		return self._sqrt_recipm1_alphas_cumprod[t]

	def data_scale(self, t):
		return self._sqrt_recip_alphas_cumprod[t]

	def forward(self, x, t):
		"""Get mu_t-1 given x_t."""
		x = torch.atleast_2d(x)
		t = torch.atleast_1d(t)

		# todo should we do this?
		# added this first normalize t between 0 and 1
		t = (t / self._n_steps).float()
		# t = t.float()
		outs = self.net(x, t)
		return outs

	def stats(self):
		"""Returns static variables for computing variances."""
		return {
				'betas': self._betas,
				'alphas': self._alphas,
				'alphas_cumprod': self._alphas_cumprod,
				'alphas_cumprod_prev': self._alphas_cumprod_prev,
				'sqrt_alphas_cumprod': self._sqrt_alphas_cumprod,
				'sqrt_one_minus_alphas_cumprod': self._sqrt_one_minus_alphas_cumprod,
				'log_one_minus_alphas_cumprod': self._log_one_minus_alphas_cumprod,
				'sqrt_recip_alphas_cumprod': self._sqrt_recip_alphas_cumprod,
				'sqrt_recipm1_alphas_cumprod': self._sqrt_recipm1_alphas_cumprod,
				'posterior_variance': self._posterior_variance,
				'posterior_log_variace_clipped': self._posterior_log_variance_clipped
		}

	def q_mean_variance(self, x_0, t):
		"""Returns parameters of q(x_t | x_0)."""
		mean = extract(self._sqrt_alphas_cumprod, t, x_0.shape) * x_0
		variance = extract(1. - self._alphas_cumprod, t, x_0.shape)
		log_variance = extract(self._log_one_minus_alphas_cumprod, t, x_0.shape)
		return mean, variance, log_variance
	
	def q_sample(self, x_0, t, noise=None):
		"""Sample from q(x_t | x_0)."""
		if noise is None:
			noise = torch.randn_like(x_0)
			
		x_t = (extract(self._sqrt_alphas_cumprod, t, x_0.shape) * x_0 + 
					 extract(self._sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise)
		return x_t

	def p_loss_simple(self, x_0, t):
		"""Training loss for given x_0 and t."""
		noise = torch.randn_like(x_0)
		x_noise = self.q_sample(x_0, t, noise)
		noise_recon = self.forward(x_noise, t)
		mse = (noise_recon - noise) ** 2

		mse = mse.sum(dim=-1)  # avg over the output dimension
		return mse

	def p_loss_kl(self, x_0, t):
		"""Training loss for given x_0 and t (KL-weighted)."""
		x_t = self.q_sample(x_0, t)
		q_mean, _, q_log_variance = self.q_posterior(x_0, x_t, t)
		p_mean, _, p_log_variance = self.p_mean_variance(x_t, t)

		dist_q = torch.distributions.Normal(q_mean, torch.exp(0.5 * q_log_variance))
		def _loss(pmu, plogvar):
			dist_p = torch.distributions.Normal(pmu, torch.exp(0.5 * plogvar))
			kl = torch.distributions.kl_divergence(dist_q, dist_p).sum(-1) # todo shouldn't this be sum(-1)?
			nll = -dist_p.log_prob(x_0).sum(-1) # todo shouldn't this be sum(-1)?
			return kl, nll, torch.where(t == 0, nll, kl)

		kl, nll, loss = _loss(p_mean, p_log_variance)
		return loss

	def q_posterior(self, x_0, x_t, t):
		"""Obtain parameters of q(x_{t-1} | x_0, x_t)."""
		mean = (
				extract(self._posterior_mean_coef1, t, x_t.shape) * x_0
				+ extract(self._posterior_mean_coef2, t, x_t.shape) * x_t
		)
		var = extract(self._posterior_variance, t, x_t.shape)
		log_var_clipped = extract(self._posterior_log_variance_clipped,
															t, x_t.shape)
		return mean, var, log_var_clipped

	def predict_start_from_noise(self, x_t, t, noise):
		"""Predict x_0 from x_t."""
		x_0 = (
				extract(self._sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
				- extract(self._sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
		)
		return x_0

	def p_mean_variance(self, x, t, clip=float('inf')):
		"""Parameters of p(x_{t-1} | x_t)."""

		x_recon = torch.clamp(
				self.predict_start_from_noise(x, t, noise=self.forward(x, t)), -clip,
				clip)

		mean, var, log_var = self.q_posterior(x_recon, x, t)

		if self._var_type == 'beta_reverse':
			pass
		elif self._var_type == 'beta_forward':
			var = extract(self._betas, t, x.shape)
			log_var = torch.log(var)
		elif self._var_type == 'learned':
			log_var = extract(self.out_logvar, t, x.shape)
			var = torch.exp(log_var)
		else:
			raise ValueError(f'{self._var_type} not recognised.')

		return mean, var, log_var

	def p_sample(self, x, t, clip=float('inf')):
		"""Sample from p(x_{t-1} | x_t)."""
		mean, _, log_var = self.p_mean_variance(x, t, clip=clip)

		noise = torch.randn_like(x)

		x_tm1 = mean + torch.exp(0.5 * log_var) * noise
		return x_tm1

	def _prior_kl(self, x_0):
		"""KL(q_T(x) || p(x))."""
		t = torch.ones(x_0.shape[0], dtype=torch.long, device=x_0.device) * (self._n_steps - 1)
		qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
		qt_dist = torch.distributions.Normal(qt_mean, torch.exp(0.5 * qt_log_variance))
		p_dist = torch.distributions.Normal(torch.zeros_like(qt_mean), torch.ones_like(qt_mean))
		kl = torch.distributions.kl_divergence(qt_dist, p_dist).mean(-1)
		return kl

	def logpx(self, x_0, samples_per_step=100):
		"""Full elbo estimate of model."""
		b = x_0.shape[0]
		x_0 = eo.repeat(x_0, 'b d -> (b s) d', s=samples_per_step)
		with torch.no_grad():
			e = self._prior_kl(x_0)
		with torch.no_grad():
			kls = run_batched(self.loss_all_t, x_0, loss_type='kl')
		kls = eo.reduce(kls, '(b t) -> b', 'sum', b=x_0.shape[0], t=self._n_steps)
		logpx = -(kls + e)
		logpx_mean = eo.rearrange(logpx, '(b s) -> b s', b=b, s=samples_per_step).mean(dim=-1)
		logpx_std = eo.rearrange(logpx, '(b s) -> b s', b=b, s=samples_per_step).std(dim=-1).div(torch.sqrt(torch.tensor(samples_per_step, device=logpx.device)))
		return {'logpx': logpx_mean, 'logpx_std': logpx_std}

	@torch.no_grad()
	def sample(self, n, include_path=False, clip=float('inf')):
		"""Sample from p(x)."""
		device = next(self.parameters()).device
		x = torch.randn(n, self._dim, device=device)
		if include_path:
			path = [x.clone()]
	
		for i in range(self._n_steps - 1, -1, -1):
			t = torch.ones(n, dtype=torch.long, device=device) * i
			t = t.to(x.device)
			x = self.p_sample(x, t, clip=clip)
			if include_path:
				path.append(x.clone())
		
		if include_path:
			return x, torch.stack(path)
		else:
			return x

	def loss(self, x):
		if self._mc_loss:
			return self.loss_mc(x, loss_type=self._loss_type)
		else:
			return self.loss_all_t(x, loss_type=self._loss_type)

	def loss_mc(self, x, loss_type=None):
		"""Compute training loss, uniformly sampling t's."""
		t = torch.randint(0, self._n_steps, (x.shape[0],))
		t = t.to(x.device)
		if loss_type == 'simple':
			loss = self.p_loss_simple(x, t)
		elif loss_type == 'kl':
			loss = self.p_loss_kl(x, t)
		else:
			raise ValueError(f'Unrecognized loss type: {loss_type}')

		return loss

	def loss_all_t(self, x, loss_type=None, samples_per_step=1):
		"""Compute training loss enumerated and averaged over all t's."""
		assert isinstance(x, torch.Tensor)
		b = x.shape[0]
		t = torch.arange(0, self._n_steps, device=x.device)
		t = eo.repeat(t, 't -> (b t s)', b=b, s=samples_per_step)
		x_r = eo.repeat(x, 'b d -> (b t s) d', t=self._n_steps, s=samples_per_step)

		if loss_type == 'simple':
			loss = self.p_loss_simple(x_r, t)
		elif loss_type == 'kl':
			loss = self.p_loss_kl(x_r, t)
		else:
			raise ValueError(f'Unrecognized loss type: {loss_type}')
		loss = eo.reduce(loss, '(b t s) -> (b t)', 'mean', b=b, t=self._n_steps, s=samples_per_step)
		return loss

	def p_gradient(self, x, t, clip=float('inf')):
		"""Compute mean and variance of Gaussian reverse model p(x_{t-1} | x_t)."""
		b = x.shape[0]
		gradient = self.forward(x, t)
		gradient = gradient * extract(self._sqrt_recipm1_alphas_cumprod_custom, t, gradient.shape)

		return gradient

#   def p_energy(self, x, t, clip=float('inf')):
#     """Compute mean and variance of Gaussian reverse model p(x_{t-1} | x_t)."""
#     b = x.shape[0]
#     x = x.view(b, -1)
#     t = t.view(b, -1)

#     energy = self.net.neg_logp_unnorm(x, t)
#     energy = energy * extract(self._sqrt_recipm1_alphas_cumprod_custom, t, energy.shape)

#     return energy

# Helper function to extract values at timesteps
def extract(a, t, broadcast_shape):
	"""Extract coefficients at timesteps t and reshape to broadcasting shape."""
	assert a.device == t.device
	# device = t.device
	b = t.shape[0]
	out = torch.gather(a, 0, t)
	return out.reshape(b, *((1,) * (len(broadcast_shape) - 1)))


class ResnetDiffusionModel(nn.Module):
		"""Resnet score model in PyTorch.
		
		Adds embedding for each scale after each linear layer.
		"""
		def __init__(self,
								n_steps,
								n_layers,
								x_dim,
								h_dim,
								emb_dim,
								widen=2,
								emb_type='learned',
								name=None):
				assert emb_type in ('learned',)
				super().__init__()
				self._n_layers = n_layers
				self._n_steps = n_steps
				self._x_dim = x_dim
				self._h_dim = h_dim
				self._emb_dim = emb_dim
				self._widen = widen
				self._emb_type = emb_type

				# Input embedding
				self.input_layer = nn.Linear(x_dim, h_dim)
				
				# Time embedding
				if emb_type == 'learned':
						self.time_embed = nn.Embedding(n_steps, emb_dim)
				
				# Resnet layers
				self.resnet_blocks = nn.ModuleList()
				for _ in range(n_layers):
						block = nn.ModuleDict({
								'norm': nn.LayerNorm(h_dim),
								'layer_h': nn.Linear(h_dim, h_dim * widen),
								'layer_emb': nn.Linear(emb_dim, h_dim * widen),
								'layer_int': nn.Linear(h_dim * widen, h_dim * widen),
								'layer_out': nn.Linear(h_dim * widen, h_dim, bias=False)
						})
						# Initialize output layer weights to zero
						nn.init.zeros_(block['layer_out'].weight)
						self.resnet_blocks.append(block)
						
				# Output layer
				self.output_layer = nn.Linear(h_dim, x_dim, bias=False)
				nn.init.zeros_(self.output_layer.weight)

		def forward(self, x, t):
				x = torch.atleast_2d(x)
				t = torch.atleast_1d(t)
				
				# Time embedding
				if self._emb_type == 'learned':
						# Convert continuous t to discrete steps
						t = (t * self._n_steps).long()
						emb = self.time_embed(t)
				
				# Initial projection
				x = self.input_layer(x)
				
				# Resnet blocks
				for block in self.resnet_blocks:
						h = block['norm'](x)
						h = F.silu(h)  # SiLU is the same as swish
						h = block['layer_h'](h)
						h = h + block['layer_emb'](emb)
						h = F.silu(h)
						h = block['layer_int'](h)
						h = F.silu(h)
						h = block['layer_out'](h)
						x = x + h
						
				x = self.output_layer(x)
				return x
