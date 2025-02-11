from functools import partial
import FrEIA.modules as Fm
import torch
import numpy as np


class ParameterAwareCouplingBlock(Fm.coupling_layers._BaseCouplingBlock):
    def __init__(self, dims_in, dims_c=[], internal_coupling_class=Fm.GLOWCouplingBlock, t_emb_dim=1, **kwargs):
        super().__init__(dims_in, dims_c)
        dims_c_coupling = dims_c + [(t_emb_dim, *dims_in[0][1:])]
        self.coupling = internal_coupling_class(dims_in, dims_c=dims_c_coupling, **kwargs)
        self.t_emb_dim = t_emb_dim

    def forward(self, x, parameter=None, c=[], rev=False, jac=True):
        self._coupling1 = partial(self._coupling1_parameter_aware, parameter=parameter)
        self._coupling2 = partial(self._coupling2_parameter_aware, parameter=parameter)
        return super().forward(x, c, rev, jac)

    def _coupling1_parameter_aware(self, x1, u2, parameter: torch.Tensor, rev=False):
        if parameter.numel() == self.t_emb_dim:
            parameter = parameter.reshape(1, self.t_emb_dim) * torch.ones(u2.shape[0], 1, device=u2.device)
        else:
            parameter = parameter.reshape(u2.shape[0], self.t_emb_dim)

        if u2.ndim == 4:
            parameter = parameter.unsqueeze(-1).unsqueeze(-1) * torch.ones(u2.shape[0], 1, u2.shape[2], u2.shape[3], device=u2.device)
        u2 = torch.cat([u2, parameter], dim=1)
        return self.coupling._coupling1(x1, u2, rev=rev)

    def _coupling2_parameter_aware(self, x2, u1, parameter: torch.Tensor, rev=False):
        if parameter.numel() == self.t_emb_dim:
            parameter = parameter.reshape(1, self.t_emb_dim) * torch.ones(u1.shape[0], 1, device=u1.device)
        else:
            parameter = parameter.reshape(u1.shape[0], self.t_emb_dim)

        if u1.ndim == 4:
            parameter = parameter.unsqueeze(-1).unsqueeze(-1) * torch.ones(u1.shape[0], 1, u1.shape[2], u1.shape[3], device=u1.device)

        u1 = torch.cat([u1, parameter], dim=1)
        return self.coupling._coupling2(x2, u1, rev=rev)

class ParameterAwareInvertibleModule(Fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, internal_module_class=Fm.ActNorm, t_emb_dim=1, **kwargs):
        super().__init__(dims_in, dims_c)
        if dims_c is None:
            dims_c = []
        dims_c_module = dims_c + [(t_emb_dim, *dims_in[0][1:])]
        self.internal_module = internal_module_class(dims_in, dims_c=dims_c_module, **kwargs)
        self.t_emb_dim = t_emb_dim

    def forward(self, x_or_z, c = None, rev=False, jac=False, parameter: torch.Tensor = 1.0):
        x_or_z = x_or_z[0]

        if parameter.numel() == self.t_emb_dim:
            parameter = parameter.reshape(1, self.t_emb_dim) * torch.ones(x_or_z.shape[0], 1, device=x_or_z.device)
        else:
            parameter = parameter.reshape(x_or_z.shape[0], self.t_emb_dim)

        if x_or_z.ndim == 4:
            parameter = parameter.unsqueeze(-1).unsqueeze(-1) * torch.ones(x_or_z.shape[0], 1, x_or_z.shape[2], x_or_z.shape[3], device=x_or_z.device)
        
        # print(parameter[:5])
        if c is None:
            c = [parameter]
        else:
            c.append(parameter)
        return self.internal_module((x_or_z, ), c, rev=rev, jac=jac)

    def output_dims(self, input_dims):
        return self.internal_module.output_dims(input_dims)

class AugmentedFlow(Fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, dim_augment=0):
        super().__init__(dims_in, dims_c)
        self.dim_augment = dim_augment

    def forward(self, x_or_z, c = None, rev=False, jac=False, parameter: torch.Tensor = 1.0):
        x_or_z = x_or_z[0]
        if not rev:
            x_or_z_augment = torch.randn(x_or_z.shape[0], self.dim_augment, *x_or_z.shape[2:], device=x_or_z.device) * torch.sqrt(parameter)
            x_or_z = torch.cat([x_or_z, x_or_z_augment], dim=1)
            log_jac_det = 0.5 * torch.sum(x_or_z_augment**2 / parameter, dim=tuple(range(1, x_or_z_augment.ndim))) + 0.5 * torch.log(2*np.pi*parameter.squeeze()) * self.dim_augment
        else:
            x_or_z, x_or_z_augment = x_or_z.split([x_or_z.shape[1] - self.dim_augment, self.dim_augment], dim=1)
            log_jac_det = -0.5 * torch.sum(x_or_z_augment**2 / parameter, dim=tuple(range(1, x_or_z_augment.ndim))) - 0.5 * torch.log(2*np.pi*parameter.squeeze()) * self.dim_augment
        return (x_or_z,), log_jac_det

    def output_dims(self, input_dims):
        return [(input_dims[0][0] + self.dim_augment, *input_dims[0][1:])]

class InvertibleSigmoidReverse(Fm.InvertibleSigmoid):
    def forward(self, *args, rev=False, **kwargs):
        return super().forward(*args, rev=not rev, **kwargs)
