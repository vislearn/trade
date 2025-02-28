import torch
import types
from bgmol.datasets import Ala2TSF600, Ala2TSF300, Ala2TSF1000


def derivative_dynamic(self, x, parameter, wrt="temperature", computed_energy=None, **kwargs):
    if computed_energy is None:
        computed_energy = self._energy(x)

    if wrt == "temperature":
        return - computed_energy/(parameter**2)
    elif wrt == "inverse_temperature":
        return computed_energy
    elif wrt == "coordinates":
        return torch.autograd.grad(computed_energy.sum(), x, create_graph=True)[0]
    else:
        raise(NotImplementedError(f"Unknown parameter {wrt}"))

def get_energy_dynamic(energy_unconditional):
    def energy_dynamic(self, x, c=None, **kwargs):
        if "temperature" in kwargs:
            kwargs["temperature"] = torch.tensor(kwargs["temperature"], device=x.device).reshape(-1, 1)
        return energy_unconditional(x, **kwargs)
    return energy_dynamic

class Ala2_300(Ala2TSF300):

    @property
    def has_energy(self):
        return True
    
    def get_energy_model(self, **kwargs):
        self.system.reinitialize_energy_model(temperature=self.temperature, **kwargs)
        energy_model = self.system.energy_model
        energy_model.derivative = types.MethodType(derivative_dynamic, energy_model)
        energy_unconditional = energy_model.energy
        energy_model.energy = types.MethodType(get_energy_dynamic(energy_unconditional), energy_model)
        return energy_model

class Ala2_600(Ala2TSF600):

    @property
    def has_energy(self):
        return True

    def get_energy_model(self, **kwargs):
        self.system.reinitialize_energy_model(temperature=self.temperature, **kwargs)
        energy_model = self.system.energy_model
        energy_model.derivative = types.MethodType(derivative_dynamic, energy_model)
        energy_unconditional = energy_model.energy
        energy_model.energy = types.MethodType(get_energy_dynamic(energy_unconditional), energy_model)
        return energy_model

class Ala2_1000(Ala2TSF1000):

    @property
    def has_energy(self):
        return True

    def get_energy_model(self, **kwargs):
        self.system.reinitialize_energy_model(temperature=self.temperature, **kwargs)
        energy_model = self.system.energy_model
        energy_model.derivative = types.MethodType(derivative_dynamic, energy_model)
        energy_unconditional = energy_model.energy
        energy_model.energy = types.MethodType(get_energy_dynamic(energy_unconditional), energy_model)

        return energy_model


ALA2BYTEMPERATURE = {
    300.0: (Ala2_300, "Ala2TSF300.npy"),
    600.0: (Ala2_600, "Ala2TSF600.npy"),
    1000.0: (Ala2_1000, "Ala2TSF1000.npy"),
}
