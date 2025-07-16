import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class ConditionalDDPM(nn.Module):
    def __init__(self, unet_model, betas, noise_steps, device, noise_type='linear'):
        super(ConditionalDDPM, self).__init__()

        self.n_T = noise_steps
        self.device = device
        self.unet_model = unet_model
        self.noise_type = noise_type

        if noise_type == 'linear':
            self.betas = torch.linspace(betas[0], betas[1], noise_steps).to(device)
        elif noise_type == 'cosine':
            timesteps = torch.linspace(0, 1, noise_steps)
            self.betas = (betas[0] + (betas[1] - betas[0]) * (1 + torch.cos(np.pi * timesteps)) / 2).to(device)
        else:
            raise ValueError("Unsupported noise type. Choose 'linear' or 'cosine'.")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)
        self.one_minus_alphas_cumprod = (1 - self.alphas_cumprod).to(device)

        self.mab_over_sqrtmab = (self.betas / self.sqrt_one_minus_alphas_cumprod).to(device)

        self.mse_loss = nn.MSELoss()

        self.device = device

    def forward(self, x, cond):
        """training ddpm, sample time and noise randomly (return loss)"""
        batch_size = x.shape[0]
        timestep = torch.randint(0, self.n_T, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timestep].view(batch_size, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep].view(batch_size, 1, 1, 1)
        x_t = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        predict_noise = self.unet_model(x_t, cond, timestep / self.n_T)
        mse_loss = self.mse_loss(noise, predict_noise)

        return mse_loss

    def sample(self, cond, size, device):
        """sample initial noise and generate images based on conditions"""
        n_sample = len(cond)
        x_i = torch.randn(n_sample, *size).to(device)
        sqrt_beta_t = torch.sqrt(self.betas).to(device)
        samples = []

        for idx in range(self.n_T-1, -1, -1):
            timestep = torch.tensor([idx / self.n_T]).to(device)
            z = torch.randn(n_sample, *size).to(device) if idx > 1 else 0
            eps = self.unet_model(x_i, cond, timestep)
            
            oneover_sqrta = self.sqrt_recip_alphas[idx] * (x_i - eps * self.mab_over_sqrtmab[idx])
            beta_t = sqrt_beta_t[idx] * z
            x_i = oneover_sqrta + beta_t
            
            if idx < 100 and idx % 10 == 0:
                samples.append(x_i[9].clone())

        return x_i, samples