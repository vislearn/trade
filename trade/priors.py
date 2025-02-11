import torch.nn as nn
import torch
from torch.distributions import Distribution, Normal, Uniform

class Prior(nn.Module, Distribution):
    def __init__(self, base_distribution_class, trainable=False, **kwargs):
        super().__init__()
        self.parameter_keys = kwargs.keys()
        for key, value in kwargs.items():
            setattr(self, key, nn.Parameter(value, requires_grad=trainable))
        self.base_distribution_class = base_distribution_class

    def to(self, device):
        for key in self.parameter_keys:
            setattr(self, key, getattr(self, key).to(device))
        return self

    def log_prob(self, *args, **kwargs):
        distribution_parameters = {key: getattr(self, key) for key in self.parameter_keys}
        return self.base_distribution_class(**distribution_parameters).log_prob(*args, **kwargs)
    
    def sample(self, *args, **kwargs):
        distribution_parameters = {key: getattr(self, key) for key in self.parameter_keys}
        return self.base_distribution_class(**distribution_parameters).sample(*args, **kwargs)

GaussianPrior = Normal

class GaussianPriorParameterScaling(GaussianPrior):
    def log_prob(self, x, parameter: float | torch.Tensor = 1.0):
        if not isinstance(parameter, torch.Tensor):
            parameter = torch.ones(x.shape[0], 1, device=x.device)*parameter
        return super().log_prob(x / torch.sqrt(parameter).reshape(-1, 1)) - torch.log(torch.sqrt(parameter).reshape(-1, 1))
    
    def sample(self, sample_shape=torch.Size(), parameter: float | torch.Tensor = 1.0):
        if not isinstance(parameter, torch.Tensor):
            parameter = torch.ones(sample_shape[0], 1, device=self.loc.device)*parameter
        return super().sample(sample_shape) * torch.sqrt(parameter).reshape(-1, 1)

class GaussianMixturePrior(Distribution):
    def __init__(self, locs: torch.Tensor = torch.ones(1), scales: torch.Tensor = torch.ones(1), weights: torch.Tensor = torch.ones(1)):
        super().__init__()
        self.locs = locs
        self.scales = scales
        self.weights = weights
        assert len(locs) == len(scales) == len(weights)

    def log_prob(self, x):
        weights = self.weights / self.weights.sum()
        log_probs = torch.stack([Normal(loc, scale).log_prob(x) for loc, scale in zip(self.locs, self.scales)])
        return torch.logsumexp(log_probs + torch.log(weights), dim=0)
    
    def sample(self, sample_shape=torch.Size()):
        weights = self.weights / self.weights.sum()
        component_indices = torch.multinomial(weights, sample_shape[0], replacement=True)
        samples = torch.stack([Normal(self.locs[i], self.scales[i]).sample() for i in component_indices])
        return samples
    
UniformPrior = Uniform

class UniformPriorParameterScaling(UniformPrior):
    def log_prob(self, x, parameter: float | torch.Tensor = 1.0):
        return super().log_prob(x)
    
    def sample(self, sample_shape=torch.Size(), parameter: float | torch.Tensor = 1.0):
        return super().sample(sample_shape)