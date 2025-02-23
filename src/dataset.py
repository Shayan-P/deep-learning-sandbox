import torch
import einops as eo
from src.utils import bb
from src.visualization import plot_scatter, plot_image
from src.utils import show_plot


class GMMDistributionMultivariate(torch.distributions.Distribution):
    def __init__(self, means, stds, weights):
        """
        means: [..., N, D]
        stds: [..., N, D, D]
        weights: [..., N]
        """
        self.normals = torch.distributions.MultivariateNormal(means, stds)
        self.categorical = torch.distributions.Categorical(weights)

        self.weights = weights / weights.sum(dim=-1, keepdim=True) # normalize
        self.weights_log = torch.log(self.weights)

        self.n = means.shape[-2]
        assert stds.shape[-3] == self.n
        assert weights.shape[-1] == self.n

        self.d = means.shape[-1]
        assert stds.shape[-1] == self.d
        assert stds.shape[-2] == self.d

        self.repeat_shape = self.weights.shape[:-1]
        assert self.repeat_shape == means.shape[:-2]
        assert self.repeat_shape == stds.shape[:-3]
        assert self.repeat_shape == weights.shape[:-1]

    def log_prob(self, x):
        """
        shape(x): [..., D]
        """
        x = eo.repeat(x, "... d -> ... n d", n=self.n)
        log_prob = self.normals.log_prob(x) + self.weights_log
        agg_log_prob = torch.logsumexp(log_prob, dim=-1)
        return agg_log_prob
    
    def sample(self, shape):
        """
        shape: [..., D]
        """
        indices = self.categorical.sample(shape)
        samples = self.normals.sample(shape)
        samples_selected = torch.take_along_dim(samples, eo.repeat(indices, "... -> ... 1 d", d=self.d), dim=-2).squeeze(-2)
        return samples_selected


class GMMOnCircle2D(GMMDistributionMultivariate):
    def __init__(self, n, std):
        theta = torch.arange(n) * torch.pi * 2 / n
        means = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        stds = torch.eye(2).repeat(n, 1, 1) * (std ** 2)
        super().__init__(
            means=means,
            stds=stds,
            weights=torch.ones(n) / n,
        )
    

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, n, dist: torch.distributions.Distribution):
        self.dist = dist
        self.n = n
        self.samples = dist.sample((n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.samples[idx]

    def analytic_score(self, x):
        device = self.dist.sample((1,)).device # hacky way to get device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device)
        return self.dist.log_prob(x)


class SimpleNormal2D(torch.utils.data.Dataset):
    def __init__(self, n, std):
        self.n = n
        self.noise = torch.randn(self.n, 2) * std
        self.std = std

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.noise[idx]

    def analytic_score(self, x):
        """
        shape(x): [B, D]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        device = x.device

        dist = torch.distributions.Normal(torch.zeros_like(x, device=device), torch.ones_like(x, device=device) * self.std)
        log_prob = dist.log_prob(x).sum(dim=-1)
        return log_prob


class GMM2D(torch.utils.data.Dataset):
    def __init__(self, each, std):
        self.clusters = torch.tensor([
            [0.0, 1.0],
            [1.0, -0.5],
            [-1.0, -0.5]
        ])
        self.each = each
        self.std = std
        self.noise = torch.randn(self.__len__(), 2) * std

    def __len__(self):
        return self.each * len(self.clusters)

    def __getitem__(self, idx):
        return self.clusters[idx % len(self.clusters)] + self.noise[idx]

    def analytic_score(self, x):
        """
        shape(x): [B, D]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        device = x.device
        dist = torch.distributions.Normal(self.clusters, torch.ones_like(self.clusters, device=device) * self.std)
        x_ = x.unsqueeze(1).repeat(1, len(self.clusters), 1)
        log_prob = dist.log_prob(x_).sum(dim=-1)
        log_prob_sum = torch.logsumexp(log_prob, dim=-1)
        log_prob_mean = log_prob_sum - torch.log(torch.tensor(len(self.clusters), device=device))
        return log_prob_mean


class Spiral2D(torch.utils.data.Dataset):
    def __init__(self, start_angle=0.0, end_angle=torch.pi * 3, start_radius=0.1, end_radius=1.0, n=1000):
        theta = torch.linspace(start_angle, end_angle, n)
        radius = torch.linspace(start_radius, end_radius, n)
        self.x = radius * torch.cos(theta)
        self.y = radius * torch.sin(theta)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor([self.x[idx], self.y[idx]])
    
    def analytic_score(self, x):
        """
        shape(x): [B, D]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        device = x.device
        pts = torch.stack([self.x, self.y], dim=-1).to(device)
        distances = (pts[None, :, :] - x[:, None, :]).pow(2).sum(dim=-1).sqrt()
        distances = distances.min(dim=-1).values
        log_prob = -torch.where(distances < 0.1, (distances**2) * 100, distances * 30)
        assert len(log_prob.shape) == 1
        log_prob = log_prob - torch.logsumexp(log_prob, dim=0)
        return log_prob


def demo_gmm():
    n = 10
    theta = torch.arange(n) * torch.pi * 2 / n
    means = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    stds = torch.eye(2).repeat(n, 1, 1) * 0.03

    dist = GMMDistributionMultivariate(
        means=means,
        stds=stds,
        weights=torch.ones(n) / n,
    )

    samples = dist.sample((10000,))

    plot_scatter(samples)
    show_plot("gmm_dist")

    x = torch.linspace(-1.5, 1.5, 50)
    y = torch.linspace(-1.5, 1.5, 50)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    pts = torch.stack([X, Y], dim=-1)
    Z = dist.log_prob(pts)
    plot_image(Z.exp(), nx=50, ny=50, lx=-1.5, ly=-1.5, rx=1.5, ry=1.5)
    show_plot("gmm_dist_log_prob")
