import math
import numpy as np
import torch
from .config import (BoltzmannGeneratorHParams, FlowHParams, LossHParams, 
                     CouplingBlockHParams, RQSplineHParams, AffineHParams, 
                     ParameterPriorHParams, BGFlowHParams, DataAugmentorHParams)
from inspect import isclass

def get_clipped_loss_from_hparams(hparams: LossHParams | None, loss):
    if hparams is None:
        # This can happen during validation
        clip = 1e9
        loss[loss > clip] = clip
        loss[loss.isnan()] = 0.0
        return loss
    loss = torch.clamp(loss, None, hparams.clip)
    loss[loss > hparams.log_clip] = torch.log(1 + loss[loss > hparams.log_clip] - hparams.log_clip) + hparams.log_clip
    loss[loss.isnan()] = hparams.clip
    return loss

def get_loss_lambda_from_hparams(hparams: LossHParams | None, total_steps, current_step):
    if hparams is None:
        # This can happen during validation
        return 0.0
    shift = math.floor(hparams.pct_start*total_steps)
    current, total = current_step - shift, total_steps - shift
    if current < 0:
        lam =  hparams.start_value
    else:
        lam = hparams.end_value - (hparams.end_value - hparams.start_value) * (1 + np.cos(np.pi * current / total)) / 2
    if hparams.adaptive:
        lam = lam * hparams.adaptive_weight
    return lam

def parameter_schedule(total_steps, hparams):
    if isinstance(hparams.parameters, (float, int, np.int64)):
        def parameter_prior(step, batchsize):
            return torch.ones(batchsize)*hparams.parameters
    else:
        def parameter_prior(step, batchsize):
            if hparams.sample_parameter_per_batch:
                requested_batchsize = batchsize
                batchsize = 1
            t_min, t_max = hparams.parameters
            t_min, t_max = np.log(t_min), np.log(t_max)
            c = torch.bernoulli(torch.ones(batchsize)*t_max/(t_max - t_min))

            assert hparams.s_min is not None and hparams.s_max is not None, "s_min and s_max must be set if a parameter range is used \
                (hint: set both to 1 for a log-uniform distribution)"
            assert hparams.s_min > 0 and hparams.s_max > 0
            exp_start = np.log(hparams.s_min)
            exp_end = np.log(hparams.s_max)
            exp = np.exp(exp_start + (exp_end - exp_start) * step/ total_steps)

            samples = torch.empty(batchsize)
            samples[c == 0] = torch.exp(t_min*(1-torch.rand(torch.sum(c == 0))**exp))
            samples[c == 1] = torch.exp(t_max*(1-torch.rand(torch.sum(c == 1))**exp))
            if hparams.sample_parameter_per_batch:
                samples = samples.repeat(requested_batchsize)
            return samples
    return parameter_prior

def weight_by_temperature(nll_base, temperature, min_nll):
    logp = (-nll_base + min_nll + 2) * (1/temperature.squeeze() - 1)
    return torch.exp(logp)

def convert_dict_to_hparams(hparams, dim, dims_c=None):
    if not isinstance(hparams, BoltzmannGeneratorHParams):
        hparams = BoltzmannGeneratorHParams(**hparams)
    if not isinstance(hparams.flow_hparams, (FlowHParams, BGFlowHParams)):
        hparams.flow_hparams["dim"] = dim
        if dims_c is not None:
            hparams.flow_hparams["dims_c"] = dims_c
        if "flow_type" in hparams and hparams.flow_type == "bgflow":
            hparams.flow_hparams = BGFlowHParams(**hparams.flow_hparams)
        else:
            hparams.flow_hparams = FlowHParams(**hparams.flow_hparams)
    if not isinstance(hparams.nll_loss, (LossHParams, type(None))):
        hparams.nll_loss = LossHParams(**hparams.nll_loss)
    if not isinstance(hparams.kl_loss, (LossHParams, type(None))):
        hparams.kl_loss = LossHParams(**hparams.kl_loss)
    if not isinstance(hparams.temperature_weighted_loss, (LossHParams, type(None))):
        hparams.temperature_weighted_loss = LossHParams(**hparams.temperature_weighted_loss)
    if not isinstance(hparams.trade_loss, (LossHParams, type(None))):
        hparams.trade_loss = LossHParams(**hparams.trade_loss)
    if not isinstance(hparams.parameter_prior_hparams, ParameterPriorHParams):
        hparams.parameter_prior_hparams = ParameterPriorHParams(**hparams.parameter_prior_hparams)
    if not isinstance(hparams.data_augmentation_hparams, DataAugmentorHParams):
        hparams.data_augmentation_hparams = DataAugmentorHParams(**hparams.data_augmentation_hparams)                      
    if hparams.flow_type != "bgflow" and not isinstance(hparams.flow_hparams.coupling_hparams, CouplingBlockHParams):
        coupling_type = hparams.flow_hparams.coupling_hparams.pop("coupling_type", "affine")
        if coupling_type.lower() == "affine":
            hparams.flow_hparams.coupling_hparams = AffineHParams(**hparams.flow_hparams.coupling_hparams)
        elif coupling_type.lower() == "rqspline":
            hparams.flow_hparams.coupling_hparams = RQSplineHParams(**hparams.flow_hparams.coupling_hparams)
        else:
            raise ValueError(f"Unknown coupling type {coupling_type}")
    return hparams


def get_module(name, library="torch"):
    """ Get a nn.Module in a case-insensitive way """
    if library.lower() == "torch":
        base_lib = torch.nn
        module_parent_class = torch.nn.Module
    elif library.lower() == "freia":
        import FrEIA.modules as Fm
        base_lib = Fm
        module_parent_class = Fm.InvertibleModule
    modules = base_lib.__dict__
    modules = {
        key.lower(): value for key, value in modules.items()
        if isclass(value) and issubclass(value, module_parent_class)
    }

    return modules[name.lower()]

def factors_of_2(x):
     return sum([int(x / (2**i)) == x/(2**i) for i in range(int(np.log2(x))+1)]) - 1