"""Noise distribution classes and samplers for GAN training."""

import torch
import torch.distributions as dist
import torch.nn.functional as F


class NoiseDistribution:
    """Base class for noise distributions."""

    def __init__(self, noise_dim: int, device: str = "cpu"):
        self.noise_dim = noise_dim
        self.device = device
        self.distribution = None

    def sample(self, batch_size: int) -> torch.Tensor:
        if self.distribution is None:
            raise ValueError("Distribution not initialized.")
        return self.distribution.sample([batch_size])


class GaussianNoise(NoiseDistribution):
    """Multivariate normal noise."""
    def __init__(self, noise_dim, loc=0.0, scale=1.0, device="cpu", **kwargs):
        super().__init__(noise_dim, device)
        self.distribution = dist.multivariate_normal.MultivariateNormal(
            loc=torch.full((self.noise_dim,), loc, device=self.device),
            covariance_matrix=torch.eye(self.noise_dim, device=self.device) * (scale ** 2),
        )


class UniformNoise(NoiseDistribution):
    """Uniform continuous noise."""
    def __init__(self, noise_dim, low=0.0, high=1.0, device="cpu", **kwargs):
        super().__init__(noise_dim, device)
        self.distribution = dist.Uniform(
            low=torch.full((self.noise_dim,), float(low), device=self.device),
            high=torch.full((self.noise_dim,), float(high), device=self.device),
        )


class LaplaceNoise(NoiseDistribution):
    """Laplace noise."""
    def __init__(self, noise_dim, loc=0.0, scale=1.0, device="cpu", **kwargs):
        super().__init__(noise_dim, device)
        self.distribution = dist.Laplace(
            loc=torch.full((self.noise_dim,), loc, device=self.device),
            scale=torch.full((self.noise_dim,), scale, device=self.device),
        )


class ExponentialNoise(NoiseDistribution):
    """Exponential noise."""
    def __init__(self, noise_dim, rate=1.0, device="cpu", **kwargs):
        super().__init__(noise_dim, device)
        self.distribution = dist.Exponential(
            rate=torch.full((self.noise_dim,), rate, device=self.device)
        )


class UniformInteger(NoiseDistribution):
    """Uniform integer sampler."""
    def __init__(self, low=0, high=10, device="cpu", **kwargs):
        super().__init__(noise_dim=None, device=device)
        if high <= low:
            raise ValueError("high must be greater than low.")
        self.low = low
        self.high = high
        probs = torch.ones(high - low, device=self.device) / (high - low)
        self.distribution = dist.Categorical(probs=probs)

    def sample(self, batch_size: int) -> torch.Tensor:
        return self.distribution.sample((batch_size,)) + self.low


class ClassificationLabelSampler(NoiseDistribution):
    """Soft one-hot label sampler for GAN conditioning."""
    def __init__(self, num_classes: int, temperature: float=0.5, device="cpu", **kwargs):
        super().__init__(noise_dim=num_classes, device=device)
        self.num_classes = num_classes
        self.temperature = temperature

    def sample(self, batch_size: int, return_condition=False) -> torch.Tensor:
        hard_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)

        u = torch.rand(batch_size, self.num_classes, device=self.device)
        e_k = F.one_hot(hard_labels, self.num_classes).float()
        v = u + e_k
        soft_targets = v / v.sum(dim=1, keepdim=True)

        if return_condition:
            return soft_targets, e_k

        return soft_targets

class RegressionLabelSampler(NoiseDistribution):
    """Soft one-hot label sampler for GAN conditioning."""
    def __init__(self, low: int, high, device="cpu", **kwargs):
        super().__init__(noise_dim=1, device=device)
        self.distribution = dist.Uniform(
            low=torch.full((self.noise_dim,), float(low), device=self.device),
            high=torch.full((self.noise_dim,), float(high), device=self.device),
        )

    def sample(self, batch_size: int, return_condition=False) -> torch.Tensor:
        targets = self.distribution.sample([batch_size])
        if return_condition:
            return targets, targets
        return targets

def get_noise_sampler(distribution: str, **kwargs) -> NoiseDistribution:
    """Return a noise sampler by name."""
    samplers = {
        "GAUSSIAN": GaussianNoise,
        "UNIFORM": UniformNoise,
        "LAPLACE": LaplaceNoise,
        "EXPONENTIAL": ExponentialNoise,
        "UINTEGER": UniformInteger,
        "CLSLABEL": ClassificationLabelSampler,
        "REGLABEL": RegressionLabelSampler,
    }
    key = distribution.upper()
    if key not in samplers:
        raise ValueError(f"Unknown noise distribution '{distribution}'.")
    return samplers[key](**kwargs)


def sample_noise(batch_size: int, distribution: str = "GAUSSIAN", **kwargs) -> torch.Tensor:
    """Convenience function to sample noise directly."""
    return get_noise_sampler(distribution, **kwargs).sample(batch_size)