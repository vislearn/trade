from .base import EnergyModel, DataSet
import numpy as np
import os
import torch

class MultiWellDataset(DataSet):
    def __init__(self, dim, temperature=1.0, read=True, **kwargs):
        self.dataset_temperature = temperature
        self.dataset_dim = dim
        super(MultiWellDataset, self).__init__(read=read, **kwargs)
        self._temperature = temperature

    def read(self, **kwargs):
        data = np.load(os.path.join(self.root, f"data/multi_well_{self.dataset_dim}d_{self.dataset_temperature:.1f}.npz"), allow_pickle=True)
        self._xyz, self._energies = data["coordinates"], data["energies"]
        self.num_frames = len(self._xyz)

    @property
    def dim(self):
        return self.dataset_dim
    
    @property
    def temperature(self):
        return self.dataset_temperature
    
    @property
    def a(self):
        return self.get_energy_model()._a
    
    @property
    def b(self):
        return self.get_energy_model()._b
    
    @property
    def c(self):
        return self.get_energy_model()._c

    def get_energy_model(self, **kwargs):
        return MultiWellEnergy(dim=int(self.dim))


class MultiWellEnergy(EnergyModel):
    def __init__(self, dim, a=0.0, b=-4.0, c=1.0, transformer=None):
        super().__init__(dim)
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c)
        self.register_buffer("_a", a)
        self.register_buffer("_b", b)
        self.register_buffer("_c", c)
        if transformer is not None:
            self.register_buffer("_transformer", transformer)
        else:
            self._transformer = None

    def _energy(self, x, **kwargs):
        a, b, c = kwargs.pop("a", self._a), kwargs.pop("b", self._b), kwargs.pop("c", self._c)

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}.\n" +
                             "MultiWellEnergy only accepts a, b, c as keyword arguments.")

        if self._transformer is not None:
            x = torch.matmul(x, self._transformer)
        e1 = a * x + b * x.pow(2) + c * x.pow(4)
        return e1.sum(dim=1).squeeze()

    def derivative(self, x, parameter, wrt=None, computed_energy=None):
        if wrt == "a":
            return x
        elif wrt == "b":
            return x.pow(2)
        elif wrt == "c":
            return x.pow(4)
        else:
            return super().derivative(x, parameter, wrt=wrt, computed_energy=computed_energy)
