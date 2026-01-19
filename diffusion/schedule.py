import torch

def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

class Diffusion:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1, 1)
        sqrt_one_minus = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise