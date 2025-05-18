import torch
import torch.functional as F
import numpy as np

class DDIMSampler:
    def __init__(self, model, betas, device='cuda', solver='ddim'):
        self.model = model
        self.device = device
        self.solver = solver
        self.set_noise_schedule(betas)

    def set_noise_schedule(self, betas):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(torch.tensor(self.alphas_cumprod)).to(self.device)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - torch.tensor(self.alphas_cumprod)).to(self.device)

    @torch.no_grad()
    def predict_noise(self, x, t):
        eps = self.model(
            x,
            torch.tensor([t], device=x.device, dtype=torch.float32),
        )
        return eps

    def ddim_step(self, x_t, t, t_prev):
        eps = self.predict_noise(x_t, t)

        alpha_t = self.sqrt_alphas_cumprod[t]
        alpha_prev = self.sqrt_alphas_cumprod[t_prev]

        x0_pred = (x_t - self.sqrt_one_minus_alpha_cumprod[t] * eps) / alpha_t
        x_prev = alpha_prev * x0_pred + torch.sqrt(1 - alpha_prev ** 2) * eps
        return x_prev

    def euler_step(self, x, t, t_prev):
        dt = t_prev - t
        f = lambda x_, t_: -self.predict_noise(x_, t_)
        return x + dt * f(x, t)

    def midpoint_step(self, x, t, t_prev):
        dt = t_prev - t
        f = lambda x_, t_: -self.predict_noise(x_, t_)
        k1 = f(x, t)
        k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
        return x + dt * k2

    def rk4_step(self, x, t, t_prev):
        dt = t_prev - t
        f = lambda x_, t_: -self.predict_noise(x_, t_)
        k1 = f(x, t)
        k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(x + dt * k3, t + dt)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def heun_step(self, x, t, t_prev):
        dt = t_prev - t
        f = lambda x_, t_: -self.predict_noise(x_, t_)
        k1 = f(x, t)
        k2 = f(x + dt * k1, t + dt)
        return x + 0.5 * dt * (k1 + k2)

    def solve_step(self, x, t, t_prev):
        if self.solver == 'ddim':
            return self.ddim_step(x, t, t_prev)
        elif self.solver == 'euler':
            return self.euler_step(x, t, t_prev)
        elif self.solver == 'midpoint':
            return self.midpoint_step(x, t, t_prev)
        elif self.solver == 'rk4':
            return self.rk4_step(x, t, t_prev)
        elif self.solver == 'heun':
            return self.heun_step(x, t, t_prev)
        else:
            raise NotImplementedError('Solver not implemented')

    def sample(self, shape, num_steps=50):
        x = torch.randn(shape, device=self.device)
        ts = np.linspace(len(self.betas) - 1, 0, num_steps, dtype=int)

        for i in range(num_steps - 1):
            t, t_prev = ts[i], ts[i + 1]
            x = self.solve_step(x, t, t_prev)
        return x