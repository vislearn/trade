from .base import EnergyModel, DataSet
import numpy as np
import os
import torch

class GMMDataset(DataSet):
    def __init__(self, dim, temperature=1.0, read=True, **kwargs):
        self.dataset_temperature = temperature
        self.dataset_dim = dim
        super(GMMDataset, self).__init__(read=read, **kwargs)
        self._temperature = temperature
        if not hasattr(self, "_energy_model"):
            data_T1 = np.load(os.path.join(self.root, f"data/gmm_{self.dataset_dim}d_1.00.npz"), allow_pickle=True)
            self._energy_model = GMMEnergy(dim=int(self.dim), locs=data_T1["locs"], scales=data_T1["scales"], mixture_weights=data_T1["mixture_weights"], temperature=temperature)


    def read(self, **kwargs):
        data = np.load(os.path.join(self.root, f"data/gmm_{self.dataset_dim}d_{self.dataset_temperature:.2f}.npz"), allow_pickle=True)
        self._xyz = data["data"]
        self._energy_model = GMMEnergy(dim=int(self.dim), locs=data["locs"], scales=data["scales"], mixture_weights=data["mixture_weights"])
        self.num_frames = len(self._xyz)

    @property
    def dim(self):
        return self.dataset_dim
    
    @property
    def temperature(self):
        return self.dataset_temperature
    
    @property
    def locs(self):
        return self._energy_model.locs
    
    @property
    def scales(self):
        return self._energy_model.scales
    
    @property
    def mixture_weights(self):
        return self._energy_model.mixture_weights

    def get_energy_model(self, **kwargs):
        return self._energy_model

class GMMEnergy(EnergyModel):
    def __init__(self, dim, locs, scales, mixture_weights, temperature=1.0):
        super().__init__(dim)
        self.locs = torch.from_numpy(locs)
        self.scales = torch.from_numpy(scales)
        self.mixture_weights = torch.from_numpy(mixture_weights)
        self.temperature = temperature

    def _energy(self, x, **kwargs):
        locs = kwargs.pop("locs", self.locs).to(x.device)
        scales = kwargs.pop("scales", self.scales).to(x.device)
        mixture_weights = kwargs.pop("mixture_weights", self.mixture_weights).to(x.device)

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}.\n" +
                             "GMMEnergy only accepts locs, scales, mixture_weights as keyword arguments.")

        mix = torch.distributions.Categorical(mixture_weights)
        com = torch.distributions.MultivariateNormal(locs,
                                                     scale_tril=scales,
                                                     validate_args=False)
        gmm = torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                         component_distribution=com,
                                                         validate_args=False)
        return - gmm.log_prob(x).flatten()/self.temperature