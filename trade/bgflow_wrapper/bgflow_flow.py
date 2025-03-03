from trade.config import BGFlowHParams
import torch
from typing import Iterable, Tuple, List
from trade.bgflow_wrapper.generators import create_generator
from bgflow import BoltzmannGenerator
import torch.nn as nn
from bgflow.utils.types import pack_tensor_in_tuple

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

        pass_temperature_to_flow = self.parameter_name == "temperature" and self.hparams.temperature_steerable
        
        if not pass_temperature_to_flow:
            *z, neg_dlogp = self.bgflow_bg.flow(x, inverse=True, context=context)
        else:
            *z, neg_dlogp = self.bgflow_bg.flow(x, inverse=True, context=context, temperature=parameter)

        if self.hparams.scale_latent_with_parameter:
            return (self.bgflow_bg.prior.energy(*z, temperature=parameter) - neg_dlogp).squeeze()
        else:
            return (self.bgflow_bg.prior.energy(*z) - neg_dlogp).squeeze()


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

    def sample(self, batch_size: int, c: Iterable[torch.Tensor], parameter:float | torch.Tensor = 1.0, 
               with_latent: bool = False, with_dlogp: bool = False, with_energy: bool = False,
               with_log_weights: bool = False, with_weights: bool = False):

        assert c is None or len(c) == 0, "Currently, we do not support conditioning"


        if self.hparams.parameter_aware:
            context = self._convert_param_to_context(parameter, torch.empty(batch_size, 1, device=next(self.parameters()).device))
        else:
            context = None

        pass_temperature_to_flow = self.parameter_name == "temperature" and self.hparams.temperature_steerable

        z = self.bgflow_bg._prior.sample(batch_size)
        z = pack_tensor_in_tuple(z)
        if not pass_temperature_to_flow:
            *x, dlogp = self.bgflow_bg._flow(*z, context=context)
        else:
            *x, dlogp = self.bgflow_bg._flow(*z, context=context, temperature=parameter)
        results = list(x)

        if with_latent:
            results.append(*z)
        if with_dlogp:
            results.append(dlogp)
        if with_energy or with_log_weights or with_weights:
            if self.hparams.scale_latent_with_parameter:
                bg_energy = self.bgflow_bg._prior.energy(*z, temperature=parameter) + dlogp
            else:
                bg_energy = self.bgflow_bg._prior.energy(*z) + dlogp
            if with_energy:
                results.append(bg_energy)
            if with_log_weights or with_weights:
                target_energy = self.bgflow_bg._target.energy(*x, **{self.parameter_name:parameter})
                log_weights = bg_energy - target_energy
                if with_log_weights:
                    results.append(log_weights)
                if with_weights:
                    weights = torch.softmax(log_weights, dim=0).view(-1)
                    results.append(weights)
        if len(results) > 1:
            return (*results,)
        else:
            return results[0]
            
    def get_energy_model(self):
        try:
            return self.target_energy
        except AttributeError:
            return None