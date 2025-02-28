from .base import EnergyModel, DataSet
import numpy as np
import os
import torch


class TwoMoonsDataset(DataSet):
    def __init__(self, dim, likelihood_power=1.0, read=True, **kwargs):
        self.beta = likelihood_power
        self.dataset_dim = dim
        super(TwoMoonsDataset, self).__init__(read=read, **kwargs)

    def read(self, **kwargs):
        data = np.load(os.path.join(self.root, f"data/two_moons_{self.beta:.1f}.npz"), allow_pickle=True)
        self._xyz = data["coordinates"]
        self.conditions = data["conditions"]
        self.num_frames = len(self._xyz)

    @property
    def dim(self):
        return self.dataset_dim
    
    @property
    def likelihood_power(self):
        return self.beta

    def get_energy_model(self, **kwargs):
        return TwoMoonsEnergy(dim=2)


class TwoMoonsEnergy(EnergyModel):
    def __init__(self, dim=2, likelihood_power=1.0):
        super().__init__(dim)
        self.beta = likelihood_power

    def _energy(self, x):
        raise NotImplementedError("TwoMoons does not have a tractable posterior.")

    def derivative(self, x, parameter, wrt=None, computed_energy=None):
        if computed_energy is None:
            raise NotImplementedError("TwoMoons does not have a tractable posterior. Please provide the energy.")
        prior_energy = - torch.distributions.Normal(0, 0.3).log_prob(x).sum(dim=1)

        if wrt == "likelihood_power":
            return computed_energy - prior_energy
        else:
            return super().derivative(x, parameter, wrt=wrt, computed_energy=computed_energy)