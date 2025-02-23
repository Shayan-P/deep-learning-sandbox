import torch
import torch.nn as nn
import einops as eo
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataset import SimpleNormal2D, GMM2D, Spiral2D
from src.visualization import plot_vector_field, plot_scatter, plot_line, colorful_curve
from torch.utils.data import DataLoader
import numpy as np
from ml_collections import ConfigDict
from src.utils import bb, show_plot, read_plot, save_gif


class FlowModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.net = nn.Sequential(
            nn.Linear(dim+1, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, dim)
        )

    def forward(self, x, t):
        assert len(x.shape) == len(t.shape)
        shape = [max(x_s, t_s) for x_s, t_s in zip(x.shape, t.shape)]
        x = x.expand([*shape[:-1], x.shape[-1]])
        t = t.expand([*shape[:-1], t.shape[-1]])
        inp = torch.cat([x, t], dim=-1)
        out = self.net(inp)
        return out

    @torch.no_grad()
    def sample(self, x, include_path=False, num_steps=500):
        ts = torch.linspace(0.0, 1.0, num_steps, device=x.device)
        dt = ts[1] - ts[0]
        if include_path:
            path = [x]
        for t in ts:
            t = torch.tensor([t], device=x.device).repeat(x.shape[0], 1)
            dir = self.forward(x, t)
            x = x + dir * dt
            if include_path:
                path.append(x)
        if include_path:
            return x, torch.stack(path, dim=0)
        else:
            return x


def eval_model(model):
    device = next(model.parameters()).device
    ts = torch.linspace(0.0, 1.0, 9, device=device)
    dt = ts[1] - ts[0]
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    axs = axs.reshape(-1)
    for t, ax in zip(ts, axs):
        x = torch.linspace(-2, 2, 20, device=device)
        y = torch.linspace(-2, 2, 20, device=device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack([X, Y], dim=-1) # flow pos
        flow_pos = grid
        with torch.no_grad():
            flow_dir = model(flow_pos, t.view(1, 1, 1))
        plot_vector_field(pos=flow_pos.view(-1, 2), vec=dt * flow_dir.view(-1, 2), scale=1, ax=ax)
        ax.set_title(f"t={t.item()}")
    show_plot("flow_eval")
    return read_plot("flow_eval")


def sample_model(model, x_init):
    x, path = model.sample(x_init, include_path=True, num_steps=100)
    plot_scatter(x, color="red")
    for i in range(min(100, path.shape[1])):
        lc = colorful_curve(path[:, i, 0], path[:, i, 1])
    plt.colorbar(lc)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    show_plot("flow_sample")
    return read_plot("flow_sample")


class MSE_MINUS_VAR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        assert x.shape == y.shape
        loss = ((x - y) ** 2).sum(dim=-1).view(-1)
        var = torch.var(x.detach()).detach()
        return loss.mean() - var


def train(dist1_sampler, dist2_sampler, model, t_samples=20, iters=10000, plot_every=100):
    loss_fn = MSE_MINUS_VAR()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    device = next(model.parameters()).device

    losses = []
    grad_norms = []

    dist1_sample_for_plot = next(iter(dist1_sampler))

    eval_frames = []
    sample_frames = []

    with tqdm(zip(range(iters), dist1_sampler, dist2_sampler)) as pbar:
        for iter_cnt, pts1, pts2 in pbar:
            ts = torch.rand(t_samples, device=device)

            # b t d
            pts1 = eo.rearrange(pts1, "b d -> b 1 d")
            pts2 = eo.rearrange(pts2, "b d -> b 1 d")
            ts = eo.rearrange(ts, "t -> 1 t 1")

            flow_pos = pts1 + ts * (pts2 - pts1)
            flow_dir = (pts2 - pts1) * torch.ones_like(ts, device=device)
            flow_dir_pred = model(flow_pos, ts)
            loss = loss_fn(flow_dir, flow_dir_pred)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

            losses.append(loss.item())
            grad_norms.append(torch.linalg.norm(
                torch.stack([
                    torch.linalg.norm(p.grad) for p in model.parameters() if p.grad is not None
                ])
            ).item())

            if iter_cnt % plot_every == 0:
                _, ax = plt.subplots(1, 1, figsize=(5, 5))
                plot_scatter(pts1[:, 0, :].view(-1, 2), ax=ax, color="red")
                plot_scatter(pts2[:, 0, :].view(-1, 2), ax=ax, color="blue")
                plot_scatter(flow_pos[:, 0, :].view(-1, 2), ax=ax, color="green")
                ax.legend(["pts1", "pts2", "flow_pos"])
                plot_vector_field(pos=flow_pos[:, 0, :].view(-1, 2), vec=0.1 * flow_dir[:, 0, :].view(-1, 2), scale=1, ax=ax)
                ax.set_title(f"t={ts[0, 0, 0].item()}")
                show_plot("sample_train_signal")

                frame = eval_model(model)
                eval_frames.append(frame)
                frame = sample_model(model, dist1_sample_for_plot)
                sample_frames.append(frame)

                # Compute smoothed losses with sliding window of 10 using numpy
                window_size = 10
                if len(losses) >= window_size:
                    smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                else:
                    smoothed_losses = losses
                plot_line(losses)
                plot_line(smoothed_losses)
                plt.ylim(-0.5, 1.5)
                plt.legend(["loss", "smoothed_loss"])
                plt.axhline(y=0.0, color='black', linestyle='--')
                show_plot("loss")
                plot_line(grad_norms)
                show_plot("grad_norm")
    save_gif(eval_frames, "eval_frames")
    save_gif(sample_frames, "sample_frames")
    return model


def get_sampler_from_dataset(dataset, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    while True:
        for pts in dataloader:
            yield pts.to(device)


def get_sampler_from_distribution(dist: torch.distributions.Distribution, batch_size, device):
    while True:
        res = dist.sample((batch_size,)).to(device)
        yield res


def main():
    config = ConfigDict(dict(
        batch_size=512, 
        t_samples=20,
        iters=2000,
        plot_every=100,
    ))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlowModel(dim=2).to(device)

    dataset = Spiral2D(n=10000)
    goal_dist = torch.distributions.Normal(loc=torch.tensor([0.0, 0.0]), scale=torch.tensor([0.4, 0.4]))

    dist1_sampler = get_sampler_from_distribution(goal_dist, config.batch_size, device=device)
    dist2_sampler = get_sampler_from_dataset(dataset, config.batch_size, device=device)

    train(dist1_sampler, dist2_sampler, model, t_samples=config.t_samples, iters=config.iters, plot_every=config.plot_every)


def main_straight_experiment():
    config = ConfigDict(dict(
        batch_size=512, 
        t_samples=20,
        iters=2000,
        plot_every=100,
    ))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlowModel(dim=2).to(device)

    dataset_1 = torch.randn(10000, 2, device=device)
    dataset_2 = torch.randn(10000, 2, device=device)
    dataset_1[:, 0] = -0.5
    dataset_2[:, 0] = +0.5

    dist1_sampler = get_sampler_from_dataset(dataset_1, config.batch_size, device=device)
    dist2_sampler = get_sampler_from_dataset(dataset_2, config.batch_size, device=device)

    train(dist1_sampler, dist2_sampler, model, t_samples=config.t_samples, iters=config.iters, plot_every=config.plot_every)


if __name__ == "__main__":
    main()
