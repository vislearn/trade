from temperature_scaling.config import BGFlowHParams
import torch
from typing import Iterable, Tuple, List
from temperature_scaling.bgflow_wrapper.generators import create_generator
from bgflow import BoltzmannGenerator
import torch.nn as nn

class BGFlowFlow(nn.Module):
    def __init__(self, hparams: BGFlowHParams, energy_model, system, parameter_name="temperature"):
        super().__init__()


        self.parameter_name = parameter_name
        self.bgflow_bg: BoltzmannGenerator
        self.bgflow_bg, self.target_energy = create_generator(hparams, energy_model, system)
        self.hparams = hparams

    def _convert_param_to_context(self, parameter: float | torch.Tensor, shape_like: torch.Tensor):

        if isinstance(parameter, float) or parameter.numel() == 1:
            parameter = torch.ones(shape_like.shape[0], 1, device=shape_like.device) * parameter
        return parameter.reshape(-1, 1)

    def forward(self, x: torch.Tensor, c: Iterable[torch.Tensor] = None,
                rev: bool = False, jac: bool = True, parameter: float | torch.Tensor = 1.0) -> Tuple[torch.Tensor | List[torch.Tensor], torch.Tensor]:
        assert c is None or len(c) == 0, "Currently, bgflow model does not support conditioning"

        if self.hparams.parameter_aware:
            context = self._convert_param_to_context(parameter, x)
        else:
            context = None
        return self.bgflow_bg.flow(x, inverse=True, context=context)

    def energy(self, x: torch.Tensor, c: Iterable[torch.Tensor] = None, parameter:float | torch.Tensor = 1.0):
        
        assert c is None or len(c) == 0, "Currently, we do not support conditioning"

        if self.hparams.parameter_aware:
            context = self._convert_param_to_context(parameter, x)
        else:
            context = None
        return self.bgflow_bg.energy(x, context=context).squeeze()

    def log_prob(self, x: torch.Tensor, c: Iterable[torch.Tensor] = None, parameter:float | torch.Tensor = 1.0):

        return -self.energy(x, c, parameter=parameter)

    def kldiv(self, batch_size: int, c: Iterable[torch.Tensor] = None, parameter:float | torch.Tensor = 1.0):

        assert c is None or len(c) == 0, "Currently, we do not support conditioning"

        if self.hparams.parameter_aware:
            context = self._convert_param_to_context(parameter, torch.empty(batch_size, 1))
        else:
            context = None
        return self.bgflow_bg.kldiv(batch_size, context=context, **{self.parameter_name:parameter}).squeeze()

    def energy_ratio_from_latent(self, z: torch.Tensor, c: Iterable[torch.Tensor] = None, parameter:float | torch.Tensor = 1.0):
        raise NotImplementedError

    def sample(self, batch_size: int, c: Iterable[torch.Tensor], parameter:float | torch.Tensor = 1.0, **kwargs):

        assert c is None or len(c) == 0, "Currently, we do not support conditioning"


        if self.hparams.parameter_aware:
            context = self._convert_param_to_context(parameter, torch.empty(batch_size, 1, device=next(self.parameters()).device))
        else:
            context = None
        return self.bgflow_bg.sample(batch_size, context=context, **{self.parameter_name:parameter}, **kwargs)
    
    def get_energy_model(self):
        try:
            return self.target_energy
        except AttributeError:
            return None