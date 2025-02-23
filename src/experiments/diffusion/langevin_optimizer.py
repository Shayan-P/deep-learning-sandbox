### experimenting with Langeving Dynamics

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.visualization import colorful_curve
from src.utils import show_plot

def experiment(distribution, starting_point, lr, num_steps, lamda=1.0):
    x = starting_point
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(lr, torch.Tensor):
        lr = torch.tensor(lr)

    x = x.detach()
    path = [x]

    for i in range(num_steps):
        x = x.clone().detach().requires_grad_(True)
        score = distribution.log_prob(x)
        grad = torch.autograd.grad(score.sum(), x)[0]
        nxt = x + (lr/2) * grad * lamda + torch.randn_like(x) * torch.sqrt(lr)
        x = nxt.detach()
        path.append(x)

    path = torch.stack(path)
    return path


def plot_distribution(distribution, l, r, nx, device):
    xs = torch.linspace(l, r, nx, device=device)
    ys = distribution.log_prob(xs).exp()
    plt.plot(xs.cpu(), ys.cpu())

def visualize_path(path, distribution):
    fig = plt.figure(figsize=(10, 10))

    l = path.min().item()
    r = path.max().item()
    nx = 100
    plot_distribution(distribution, l=l, r=r, nx=nx, device=path.device)

    path_x = path
    path_y = distribution.log_prob(path_x).exp()
    lc = colorful_curve(path_x.cpu(), path_y.cpu())
    last = 5
    plt.scatter(path_x[-last:].cpu(), path_y[-last:].cpu(), c=np.linspace(0, 1, last), cmap='cool', norm=plt.Normalize(0, 1), alpha=1)
    plt.colorbar(lc)
    show_plot("langevin")

    return path

def get_bimodal_distribution(device, mu1=-2.0, mu2=2.0, sigma1=0.5, sigma2=0.5, mix=0.5):
    # Create a mixture of two Gaussians
    dist1 = torch.distributions.Normal(
        torch.tensor(mu1, device=device),
        torch.tensor(sigma1, device=device)
    )
    dist2 = torch.distributions.Normal(
        torch.tensor(mu2, device=device), 
        torch.tensor(sigma2, device=device)
    )
    
    class BimodalDistribution:
        def __init__(self, dist1, dist2, mixing_coef):
            self.dist1 = dist1
            self.dist2 = dist2
            self.mixing_coef = mixing_coef
            
        def log_prob(self, value):
            # Compute log probability for mixture model
            # p(x) = α*p1(x) + (1-α)*p2(x)
            # log p(x) = log(α*p1(x) + (1-α)*p2(x))
            log_prob1 = self.dist1.log_prob(value)
            log_prob2 = self.dist2.log_prob(value)
            log_mix_coef = torch.log(torch.tensor(self.mixing_coef, device=value.device))
            log_1_mix_coef = torch.log(torch.tensor(1 - self.mixing_coef, device=value.device))
            log_prob = torch.logsumexp(torch.stack([
                log_mix_coef + log_prob1,
                log_1_mix_coef + log_prob2
            ]), dim=0)
            return log_prob
            
    return BimodalDistribution(dist1, dist2, mix)


def get_distribution(device, lamda):
    # return torch.distributions.normal.Normal(
    #     torch.tensor(0.0, device=device),
    #     torch.sqrt(torch.tensor(1.0/lamda, device=device))
    # )
    dist = get_bimodal_distribution(device,
                                    mu1=-1.0, mu2=1.0,
                                    sigma1=0.2, sigma2=0.2,
                                    mix=0.5)
    return dist


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lamda = 100.0
    distribution_lambda = get_distribution(device, lamda)
    distribution = get_distribution(device, 1.0)

    lr = 0.001
    # todo why do we get Nan when lambda is too high?!

    num_steps = 10000
    starting_point = torch.tensor(-2.0)
    # path = experiment(distribution, starting_point, lr, num_steps)
    # visualize_path(path, distribution)

    starting_point = torch.tensor(-1.0).repeat(1000).to(device)
    path = experiment(distribution, starting_point, lr, num_steps, lamda=lamda)
    final_x = path[-1].cpu()

    fig = plt.figure(figsize=(10, 10))
    plt.hist(final_x.numpy(), bins=100, density=True)
    plot_distribution(distribution_lambda, l=-3, r=3, nx=100, device=device)
    show_plot("distribution")
    visualize_path(path[:, 0], distribution_lambda)


if __name__ == "__main__":
    main()
