import torch
from src.utils import bb


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
