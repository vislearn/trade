import torch
import torch.nn as nn
import numpy as np
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from functools import partial
from typing import Iterable,Tuple
from .config import FlowHParams
import warnings
from .util import get_module, factors_of_2
from .priors import (GaussianPrior,
                    GaussianPriorParameterScaling,
                    GaussianMixturePrior,
                    Prior,
                    UniformPrior,
                    UniformPriorParameterScaling)
from .coordinate_transform import CoordinateTransform, LegacyCoordinateTransform, ConstrainedCoordinateTransform
from math import prod
from .freia_modules import InvertibleSigmoidReverse, AugmentedFlow, ParameterAwareCouplingBlock, ParameterAwareInvertibleModule

class Flow(Ff.SequenceINN):
    def __init__(self, hparams: FlowHParams, energy_model, system, training_coordinates, parameter_name="temperature"):
        if isinstance(hparams.dim, (int, np.int64, np.int32)):
            hparams.dim = (hparams.dim,)

        super().__init__(*hparams.dim)
        self.hparams = hparams
        if energy_model is not None:
            self.target_energy = energy_model

        self.parameter_name = parameter_name

        # Parameter preprocessing
        self.t_embed_dim = 1
        if hparams.parameter_preprocessing is None:
            self.parameter_preprocessing = lambda x: x
        elif hparams.parameter_preprocessing == "log":
            self.parameter_preprocessing = torch.log
        elif "nn" in hparams.parameter_preprocessing:
            parameter_preprocessing = hparams.parameter_preprocessing
            use_log = False
            if hparams.parameter_preprocessing.startswith("log_"):
                parameter_preprocessing = parameter_preprocessing[len("log_"):]
                use_log = True
            if parameter_preprocessing == "nn":
                self.t_embed_dim = 1
            else:
                self.t_embed_dim = int(parameter_preprocessing[len("nn"):])
            layers = [nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, self.t_embed_dim)]
            if use_log:
                layers = [LogLayer()] + layers
            self.parameter_preprocessing = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown parameter preprocessing {hparams.parameter_preprocessing}")

        if hparams.coordinate_transform:
            if hparams.coordinate_transform == "legacy":
                self.append(LegacyCoordinateTransform, system=system, coordinates=training_coordinates)
            elif hparams.coordinate_transform == "constrained":
                self.append(ConstrainedCoordinateTransform, system=system, coordinates=training_coordinates)
            else:
                if isinstance(hparams.coordinate_transform, dict):
                    self.append(CoordinateTransform, system=system, coordinates=training_coordinates, **hparams.coordinate_transform)
                elif isinstance(hparams.coordinate_transform, bool):
                    self.append(CoordinateTransform, system=system, coordinates=training_coordinates)
                else:
                    raise(ValueError(f"Unknown coordinate transform {hparams.coordinate_transform}. \
                                     Hint: pass True for default settings, or False for no coordinate transform"))

        if hparams.sigmoid_layer:
            self.append(InvertibleSigmoidReverse)

        if hparams.augmentation_dim > 0:
            self.append(AugmentedFlow, dim_augment=hparams.augmentation_dim)

        self.build_network()

        if hparams.prior == "normal":
            trainable_parameters = {"loc": torch.zeros(self.shapes[-1]), "scale": torch.ones(self.shapes[-1])}
            prior_class = GaussianPriorParameterScaling if hparams.scale_latent_with_parameter else GaussianPrior
        elif hparams.prior == "uniform":
            trainable_parameters = {"low": torch.zeros(self.shapes[-1]), "high": torch.ones(self.shapes[-1])}
            prior_class = UniformPriorParameterScaling if hparams.scale_latent_with_parameter else UniformPrior
        elif hparams.prior.startswith("mixture"):
            if hparams.prior == "mixture":
                n_components = 10
            else:
                n_components = int(hparams.prior[len("mixture"):])
            trainable_parameters = {"locs": torch.zeros(n_components, self.shapes[-1]), "scales": torch.ones(n_components, self.shapes[-1]), "weights": torch.ones(n_components, self.shapes[-1])}
            if hparams.scale_latent_with_parameter:
                raise NotImplementedError("Mixture priors are not yet implemented with parameter scaling")
            prior_class = GaussianMixturePrior
        else:
            raise ValueError(f"Unknown prior {hparams.prior}")

        self.prior = Prior(prior_class, trainable=hparams.trainable_prior, **trainable_parameters)


    def build_network(self):
        total_dim = current_dim = self.shapes[0]

        if len(total_dim) > 1:
            n_stages = factors_of_2(total_dim[1]) + 1
            blocks_per_stage = self.hparams.n_transforms // n_stages
            if blocks_per_stage == 0:
                raise ValueError("Requested less transforms ({hparams.n_transforms}) than the number of stages ({n_stages})")
            last_stage_blocks = self.hparams.n_transforms - blocks_per_stage * (n_stages - 1)
            if total_dim == 1 and n_stages > 1:
                self.append(Fm.IRevNetDownsampling)
                n_stages -= 1
                current_dim = (total_dim[0] * 4, total_dim[1] // 2, total_dim[2] // 2)

        else:
            blocks_per_stage = self.hparams.n_transforms
            n_stages = 1
            last_stage_blocks = blocks_per_stage

        network_spec = self.hparams.coupling_hparams.hidden
        if self.hparams.coupling_hparams.resnet:
            if not isinstance(network_spec[0][0], list):
                if n_stages > 1:
                    warnings.warn("You set one hidden layer specification with multiple stages. Copying it to all stages.")
                network_spec = [network_spec] * n_stages
        else:
            if not isinstance(network_spec[0], list):
                if n_stages > 1:
                    warnings.warn("You set one hidden layer specification with multiple stages. Copying it to all stages.")
                network_spec = [network_spec] * n_stages

        assert len(network_spec) == n_stages, "If providing hidden layers per stage, the length of the list must be equal to the number of stages"

        if self.hparams.coupling_hparams.module_params.get("clamp", 2.0) is None:
            self.hparams.coupling_hparams.module_params["clamp_activation"] = lambda x: x
            self.hparams.coupling_hparams.module_params["clamp"] = 1.0

        for stage, stage_hidden in enumerate(network_spec):
            transform = get_module(self.hparams.coupling_hparams.coupling_transform, "freia")
            if current_dim[0] > 1:
                if transform == "AllInOneBlock":
                    if "permute_soft" in self.hparams.coupling_hparams.module_params.keys():
                        warnings.warn("The parameter 'permute_soft' of the AllInOneBlock is being \
                                    overwritten by the 'permutation_type' parameter of the FlowHParams")
                    self.hparams.coupling_hparams.module_params["permute_soft"] = self.hparams.permutation_type is not None and self.hparams.permutation_type == "soft"
                    self.hparams.permutation_type = None
                if self.hparams.coupling_hparams.parameter_aware:
                    transform = partial(ParameterAwareCouplingBlock, internal_coupling_class=transform, t_emb_dim=self.t_embed_dim)
            else:
                if self.hparams.coupling_hparams.parameter_aware:
                    transform = partial(ParameterAwareInvertibleModule, internal_module_class=transform, t_emb_dim=self.t_embed_dim)

            for block in range(blocks_per_stage if stage < n_stages - 1 else last_stage_blocks):
                subnet_constructor = build_resnet if self.hparams.coupling_hparams.resnet else build_subnet
                subnet_constructor = partial(subnet_constructor,
                                        hidden=stage_hidden,
                                        activation_type=self.hparams.coupling_hparams.activation,
                                        dropout=self.hparams.coupling_hparams.dropout,
                                        conv=len(current_dim) > 1,
                                        zero_init=self.hparams.coupling_hparams.zero_init)
                self.append(transform,
                            cond=None if not len(self.hparams.dims_c) else 0,
                            cond_shape=None if not len(self.hparams.dims_c) else self.hparams.dims_c[0],
                            subnet_constructor=subnet_constructor,
                            **self.hparams.coupling_hparams.module_params)

                if self.hparams.permutation_type is not None:
                    if self.hparams.permutation_type == "soft":
                        M = torch.linalg.qr(torch.randn(total_dim[0], total_dim[0]))[0]
                        if len(total_dim) > 1:
                            self.append(Fm.Fixed1x1Conv, M=M)
                        else:
                            self.append(Fm.FixedLinearTransform, M=M)
                    elif self.hparams.permutation_type == "hard":
                        self.append(Fm.PermuteRandom)
                    else:
                        raise ValueError("Unknown permutation type {self.hparams.permutation_type}")

                if self.hparams.use_actnorm:
                    self.append(Fm.ActNorm)

            if stage < n_stages - 2:
                self.append(Fm.IRevNetDownsampling)
                current_dim = (current_dim[0] * 4, current_dim[1] // 2, current_dim[2] // 2)
            elif stage == n_stages - 2:
                self.append(Fm.Flatten)
                current_dim = (prod(current_dim),)

        if self.hparams.coupling_hparams.parameter_aware:
            self.append(partial(ParameterAwareInvertibleModule, internal_module_class=Fm.ConditionalAffineTransform, t_emb_dim=1),
                        cond=None if not len(self.hparams.dims_c) else 0,
                        cond_shape=None if not len(self.hparams.dims_c) else self.hparams.dims_c[0],
                        subnet_constructor=subnet_constructor)
        else:
            self.append(Fm.ActNorm)

    def forward(self, x_or_z: torch.Tensor, c: Iterable[torch.Tensor] = None,
                rev: bool = False, jac: bool = True, parameter: float | torch.Tensor = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(parameter, float) or parameter.numel() == 1:
            parameter = torch.ones(x_or_z.shape[0], 1, device=x_or_z.device)*parameter
        parameter = parameter.reshape(-1, 1)

        iterator = range(len(self.module_list))
        log_det_jac = torch.zeros(x_or_z.shape[0], device=x_or_z.device)

        if rev:
            iterator = reversed(iterator)

        if torch.is_tensor(x_or_z):
            x_or_z = (x_or_z,)
        for i in iterator:
            if isinstance(self.module_list[i], (ParameterAwareCouplingBlock)):
                kwargs = {"parameter": self.parameter_preprocessing(parameter)}
            elif isinstance(self.module_list[i], (ParameterAwareInvertibleModule, AugmentedFlow)):
                kwargs = {"parameter": parameter}
            else:
                kwargs = {}
            if self.conditions[i] is None:
                x_or_z, j = self.module_list[i](x_or_z, jac=jac, rev=rev, **kwargs)
            else:
                x_or_z, j = self.module_list[i](x_or_z,
                                                c=[c[self.conditions[i]]],
                                                jac=jac, rev=rev, **kwargs)

            log_det_jac = j + log_det_jac
            if prod(log_det_jac.shape) != x_or_z[0].shape[0]:
                raise ValueError(f"Offending module {self.module_list[i]} produced a log_det_jac of shape {j.shape} ")

        return x_or_z if self.force_tuple_output else x_or_z[0], log_det_jac

    def energy(self, x, c, parameter:float | torch.Tensor = 1.0):
        z, log_det_jac = self.forward(x, c, rev=False, jac=True, parameter=parameter)
        if self.hparams.scale_latent_with_parameter:
            return -self.prior.log_prob(z, parameter).sum(dim=1) - log_det_jac
        else:
            return -self.prior.log_prob(z).sum(dim=1) - log_det_jac

    def log_prob(self, x, c, parameter:float | torch.Tensor = 1.0):
        return - self.energy(x, c, parameter=parameter)

    def kldiv(self, batch_size, c, parameter:float | torch.Tensor = 1.0):
        if self.hparams.scale_latent_with_parameter:
            z = self.prior.sample([batch_size], parameter)
        else:
            z = self.prior.sample([batch_size])
        return self.energy_ratio_from_latent(z, c, parameter)

    def energy_ratio_from_latent(self, z, c, parameter:float | torch.Tensor = 1.0):
        energy_model = self.get_energy_model()
        if energy_model is None:
            raise ValueError("The target energy model is not defined")

        if self.hparams.scale_latent_with_parameter:
            latent_energy = - self.prior.log_prob(z, parameter).sum(dim=1)
        else:
            latent_energy = - self.prior.log_prob(z).sum(dim=1)
        x, log_det_jac = self.forward(z, c, rev=True, jac=True, parameter=parameter)
        return energy_model.energy(x, **{self.parameter_name:parameter}).squeeze() - log_det_jac - latent_energy

    def get_energy_model(self):
        try:
            return self.target_energy
        except AttributeError:
            return None

    def sample(self, batch_size, c, parameter:float | torch.Tensor = 1.0):
        if self.hparams.scale_latent_with_parameter:
            z = self.prior.sample([batch_size], parameter)
        else:
            z = self.prior.sample([batch_size])
        return self.forward(z, c, rev=True, jac=False, parameter=parameter)[0]

def build_subnet(c_in, c_out, hidden=None, activation_type=None, conv=False, dropout=0.0, zero_init=False):
    assert type(hidden[0]) == int, "For a feed forward network, the hidden parameter must be a list of integers."
    weight_layer = partial(nn.Conv2d, kernel_size=3, padding=1) if conv else nn.Linear
    activation_layer = get_module(activation_type, "torch")
    dropout_layer = nn.Dropout2d if conv else nn.Dropout
    channels = [c_in] + hidden + [c_out]
    layers = []

    for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
        if dropout > 0.0:
            layers.append(dropout_layer(dropout))
        layers.append(weight_layer(in_channels, out_channels))
        if i < len(channels) - 2:
            layers.append(activation_layer())
    # layers[-1].weight.data.zero_()
    layers[-1].bias.data.zero_()
    if zero_init:
        layers[-1].weight.data.zero_()
    return nn.Sequential(*layers)

class SkipConnection(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return x + self.inner(x)

def build_resnet(c_in, c_out, hidden=None, activation_type=None, conv=False, dropout=0.0, zero_init=False):
    assert type(hidden[0]) == list, "For a resnet block, the hidden parameter must be a list of lists."
    weight_layer = partial(nn.Conv2d, kernel_size=3, padding=1) if conv else nn.Linear

    channels = [[c_in]] + hidden + [[c_out]]
    layers = []

    for i, (last_channels, current_channels) in enumerate(zip(channels[:-1], channels[1:])):
        if last_channels[-1] != current_channels[0] or i == len(channels) - 2:
            layers.append(weight_layer(last_channels[-1], current_channels[0]))
        assert current_channels[0] == current_channels[-1], "The first and last layer of a resnet block must have the same number of channels"
        if i < len(channels) - 2:
            layers.append(SkipConnection(build_subnet(current_channels[0], current_channels[-1], current_channels[1:], activation_type, conv, dropout)))
            layers.append(get_module(activation_type, "torch")())
    # layers[-1].weight.data.zero_()
    layers[-1].bias.data.zero_()
    if zero_init:
        layers[-1].weight.data.zero_()
    return nn.Sequential(*layers)

class LogLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.log(x)
