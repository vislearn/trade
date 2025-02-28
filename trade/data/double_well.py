from .base import EnergyModel, DataSet
import numpy as np
import os

class DoubleWellDataset(DataSet):
    def __init__(self, dim, temperature=1.0, read=True, **kwargs):
        self.dataset_temperature = temperature
        self.dataset_dim = dim
        super(DoubleWellDataset, self).__init__(read=read, **kwargs)
        self._temperature = temperature

    def read(self, **kwargs):
        data = np.load(os.path.join(self.root, f"data/double_well_{self.dataset_dim}d_{self.dataset_temperature:.1f}.npz"), allow_pickle=True)
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
        return  DoubleWellEnergy(dim=int(self.dim))



class DoubleWellEnergy(EnergyModel):
    def __init__(self,  dim, a=0, b=-4.0, c=1.0):
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x, **kwargs):

        a, b, c = kwargs.pop("a", self._a), kwargs.pop("b", self._b), kwargs.pop("c", self._c)

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}.\n" +
                             "MultiWellEnergy only accepts a, b, c as keyword arguments.")
        d = x[..., [0]]
        v = x[..., 1:]
        e1 = a * d + b * d.pow(2) + c * d.pow(4)
        e2 = 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return (e1 + e2).squeeze()

    def derivative(self, x, parameter, wrt=None, computed_energy=None):
        d = x[..., [0]]

        if wrt == "a":
            return d
        elif wrt == "b":
            return d.pow(2)
        elif wrt == "c":
            return d.pow(4)
        else:
            return super().derivative(x, parameter, wrt=wrt, computed_energy=computed_energy)

