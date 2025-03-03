import torch
import numpy as np
from warnings import warn
from torch.distributions import Distribution
from typing import Callable, Dict, Any
from functools import partial
from tqdm.auto import tqdm
from math import ceil
from typing import Tuple

from trade.models import INN_Model

from trade.datasets import (
    log_p_target_dict,
    S_dict,
    dS_dparam_dict
)

class ModelDistribution(Distribution):
    def __init__(self, flow, condition_sampler, parameter=1.0, n_repeats=1):
        super().__init__(validate_args=False)
        self.flow = flow
        self.parameter = parameter
        self.n_repeats = n_repeats
        self.condition_sampler = condition_sampler

    def sample(self, sample_shape=torch.Size([]), return_condition=False):
        c = self.condition_sampler(*sample_shape)
        if return_condition:
            return self.flow.sample(*sample_shape, c=c, parameter=self.parameter), c
        return self.flow.sample(*sample_shape, c=c, parameter=self.parameter)

    def log_prob(self, x, c):
        log_probs = []
        for i in range(self.n_repeats):
            log_probs.append(-self.flow.energy(x, c, parameter=self.parameter))
        log_probs = torch.stack(log_probs, dim=0)
        return torch.logsumexp(log_probs, dim=0) - np.log(self.n_repeats)

def consistency_check_KL(log_probs_beta, log_probs_beta_0, correction, beta, beta_0):
    KL = torch.mean(log_probs_beta - log_probs_beta_0 * beta / beta_0)
    return torch.clip(KL + correction(beta, beta_0), 0, None)

def get_correction_function(p_beta_0: ModelDistribution, q: ModelDistribution, n_proposal_samples=100000):
    batch_size = 10000
    log_probs_reference_beta_0 = []
    log_probs_reference_proposal = []
    for _ in range(n_proposal_samples//batch_size):
        x, c = q.sample(torch.Size([batch_size]), return_condition=True)
        log_probs_reference_beta_0.append(p_beta_0.log_prob(x, c).reshape(batch_size, -1).sum(dim=-1))
        log_probs_reference_proposal.append(q.log_prob(x, c).reshape(batch_size, -1).sum(dim=-1))
    log_probs_reference_beta_0 = torch.cat(log_probs_reference_beta_0, dim=0)
    log_probs_reference_proposal = torch.cat(log_probs_reference_proposal, dim=0)

    def correction(beta, beta_0):
        return torch.logsumexp(log_probs_reference_beta_0 * beta/beta_0 - log_probs_reference_proposal, dim=0) - np.log(len(log_probs_reference_beta_0))
    return correction

def compute_KL(p_beta: ModelDistribution, p_beta_0: ModelDistribution, correction, T_star, T_0, n_KL_samples=10000):
    batch_size = 10000
    beta = 1/T_star
    beta_0 = 1/T_0
    log_probs_beta = []
    log_probs_beta_0 = []
    for _ in range(n_KL_samples//batch_size):
        x, c = p_beta.sample(torch.Size([batch_size]), return_condition=True)
        log_probs_beta.append(p_beta.log_prob(x, c).reshape(batch_size, -1).sum(dim=-1))
        log_probs_beta_0.append(p_beta_0.log_prob(x, c).reshape(batch_size, -1).sum(dim=-1))
    log_probs_beta = torch.cat(log_probs_beta, dim=0)
    log_probs_beta_0 = torch.cat(log_probs_beta_0, dim=0)
    return consistency_check_KL(log_probs_beta, log_probs_beta_0, correction, beta, beta_0)

def reconstruction_loss(residuals: torch.Tensor, mode="MSE", **kwargs) -> torch.Tensor:
    """
    Compute the reconstruction loss for the target distribution.

    parameters:

        residuals: The residuals of the target and the evaluation points.

    returns:

        loss: The reconstruction loss.
    """
    if mode == "MSE":
        loss = residuals.pow(2)
    elif mode == "MAE":
        loss = residuals.abs()
    elif mode == "Huber":
        loss = torch.nn.SmoothL1Loss(beta=kwargs["delta_huber"], reduction="none")(residuals, torch.zeros_like(residuals))
    return loss

class TRADE_loss_legacy():
    def __init__(self,
                 check_consistency_with="self",
                 evaluation_mode="raw_beta_0_samples",
                 evaluation_stddev=0.1,
                 loss_mode="MSE",
                 **kwargs):
        self.check_consistency_with = check_consistency_with
        if self.check_consistency_with == "ground_truth":
            raise(RuntimeError("Ground truth consistency check is bugged in the legacy implementation. Use the new implementation instead."))
        self.evaluation_mode = evaluation_mode
        self.evaluation_stddev = evaluation_stddev
        self.loss_mode = loss_mode
        self.reconstruction_kwargs = kwargs


    def __call__(self, flow, target_temperature, reference_temperature=None, **kwargs) -> torch.Tensor:

        if reference_temperature is None and self.check_consistency_with == "ground_truth":
            warn("Reference temperature is None, but samples are taken from the dataset. \
                  This assumes that the given samples are already at the target temperature.")

        target_energy = self.get_target_energy(flow)

        samples, evaluation_points, conditions = self.sample_evaluation_points(flow, target_temperature, **kwargs)

        return self.objective_function_TS(flow = flow, target_energy = target_energy, reference_samples = samples, conditions=conditions,
                                        target_temperature = target_temperature, reference_temperature = reference_temperature,
                                        evaluation_points = evaluation_points)

    @torch.enable_grad()
    def objective_function_TS(self, flow, target_energy, reference_samples: torch.Tensor, conditions: list | tuple, target_temperature:float | torch.Tensor,
                            reference_temperature: float | None, evaluation_points: torch.Tensor)-> torch.Tensor:
        """
        This objective function computes the target using the distribution at the base temperature 1 / beta_star where
        training data is available.

        parameters:

            INN: Wrapper class for INN distributions. Used for evaluation of the loglikelihoods and for sampling.
            beta_star: The inverse temperature at which the training data used to compute the nll contribution is available.
            training_iteration: The current training iteration.
            batch_size: The batch size used for the computation of the objective
            evaluation_points_params: Dictionary containing parameters for sampling of the evaluation points.
            sample_beta_0_params: Dictionary containing parameters for sampling of the exponent beta_0 at which the gradient is evaluated.
            device: The device on which the computation is performed.

        returns:

            loss: Objective
            beta_0: The value of the exponent beta_0 at which the gradient is evaluated.

        """

        batch_size = reference_samples.shape[0]

        #1) Sample beta at which the gradient is evaluated
        beta_0_tensor = 1/target_temperature
        if not isinstance(beta_0_tensor,torch.Tensor):
            beta_0_tensor = torch.ones([batch_size,1]) * beta_0_tensor
        beta_0_tensor.requires_grad = True

        if reference_temperature is not None:
            beta_star = 1 / reference_temperature
            beta_star_tensor = torch.ones_like(beta_0_tensor) * beta_star
        else:
            beta_star = 1 / target_temperature
            beta_star_tensor = beta_0_tensor

        with torch.no_grad():

            #2) Compute the log_likelihoods of the samples at beta_0
            log_prob_beta_0 = - target_energy.energy(x = reference_samples, c = conditions, temperature = target_temperature)

            #3) Compute the expectation value of the energy at beta_star using importance sampling
            if reference_temperature is not None:
                #3a) Compute the log_likelihoods of the samples at beta_star
                log_prob_beta_star = - target_energy.energy(x = reference_samples, c = conditions, temperature = reference_temperature)

                #3b) Compute partition function using importance sampling
                log_w_i = (beta_0_tensor / beta_star) * log_prob_beta_star - log_prob_beta_0
                log_Z = torch.logsumexp(log_w_i, dim=0) - np.log(batch_size)

                #3c) Compute iportance weights for the expectation values of the energies
                log_omega_i = log_w_i - log_Z

                #3d) Compute the expectation value of the energy
                EX_energy = - (log_omega_i.exp() * log_prob_beta_star).mean()
            else:
                EX_energy = - log_prob_beta_0.mean()

            #4) Compute the energies of the evaluation points at beta_star
            eval_energies = target_energy.energy(x = evaluation_points, c = conditions, temperature = 1/beta_star_tensor)

            #5) Compute the target
            target = (EX_energy - eval_energies) / beta_star

        #6) Compute the gradient with respect to beta in beta_0
        log_prob_beta_0_eval = - flow.energy(x = evaluation_points, c = conditions, temperature = 1/beta_0_tensor)
        grad = torch.autograd.grad(log_prob_beta_0_eval.sum(), beta_0_tensor, create_graph = True)[0].reshape(-1)

        #7) Comput the objective
        loss = reconstruction_loss(grad - target.detach(), mode = self.loss_mode, **self.reconstruction_kwargs)
        return loss

    def sample_evaluation_points(self, flow, target_temperature, **kwargs):
        if self.check_consistency_with == "dataset":
            samples = kwargs.get("samples")
            assert samples is not None, "Samples must be provided if samples are taken from the dataset."
            conditions = kwargs.get("conditions")
            assert conditions is not None, "Conditions must be provided if samples are taken from the dataset."
        elif self.check_consistency_with == "flow":
            condition_sampler = kwargs.get("condition_sampler")
            assert condition_sampler is not None, "Condition sampler must be provided if samples are taken from the flow."
            batch_size = kwargs.get("batch_size")
            assert batch_size is not None, "Batch size must be provided if samples are taken from the flow."
            conditions = condition_sampler(batch_size)
            samples = flow.sample(batch_size, conditions, temperature = target_temperature)


        if self.evaluation_mode == "raw_beta_0_samples":
            evaluation_points = (samples).detach()
        elif self.evaluation_mode == "beta_0_samples_gaussian_noise":
            evaluation_points = (samples + torch.randn_like(samples) * self.evaluation_stddev).detach()
        elif self.evaluation_mode == "beta_0_samples_gaussian_noise_temperature_scaling":
            sigma = self.evaluation_stddev * np.sqrt(target_temperature)
            evaluation_points = (samples + torch.randn_like(samples) * sigma).detach()
        else:
            raise ValueError(f"Unknown evaluation mode {self.evaluation_mode}")

        return samples, evaluation_points, conditions

    def get_target_energy(self, flow):
        if self.check_consistency_with == "ground_truth":
            target_energy = flow.get_energy_model()
        elif self.check_consistency_with == "self":
            target_energy = flow
        else:
            raise ValueError(f"Unknown consistency check mode {self.check_consistency_with}")
        return target_energy


class TRADE_loss_base(torch.nn.Module):
    def __init__(self,
                 check_consistency_with="self",
                 take_samples_from="flow",
                 condition_sampler=None,
                 target_parameter_name="temperature",
                 evaluation_mode="raw_beta_0_samples",
                 evaluation_stddev=0.1,
                 loss_mode="MSE",
                 **kwargs):
        super().__init__()
        self.target_parameter_name = target_parameter_name
        self.check_consistency_with = check_consistency_with
        self.take_samples_from = take_samples_from
        self.evaluation_mode = evaluation_mode
        self.evaluation_stddev = evaluation_stddev
        self.loss_mode = loss_mode
        self.reconstruction_kwargs = kwargs
        print("Initialized TRADE loss for target parameter", self.target_parameter_name)
        if condition_sampler is None:
            def default_condition_sampler(*args, **kwargs):
                return []
            self.condition_sampler = default_condition_sampler
        else:
            self.condition_sampler = condition_sampler


    def __call__(self, flow, base_parameter=None, target_parameter_proposals=None, sample_parameter=None, **kwargs) -> torch.Tensor:

        energy_model = self.get_target_energy(flow)

        energy_derivative = self.target_energy_derivative(flow)

        target_parameter = self.sample_target_parameter(target_parameter_proposals, base_parameter, **kwargs)

        if base_parameter is None:
            base_parameter = target_parameter
        base_energy = self.get_target_energy(flow, parameter=base_parameter)
        target_energy = self.get_target_energy(flow, parameter=target_parameter)

        if sample_parameter is None:
            if self.take_samples_from == "dataset":
                warn("Given sample parameter is None, but samples are taken from the dataset. \
                    This assumes that the given samples are already at the target parameter.")
            sample_parameter = target_parameter

        reference_samples, reference_conditions, evaluation_samples, evaluation_conditions, sample_energy = self.sample_evaluation_points(flow, sample_parameter, energy_model=energy_model, **kwargs)

        if isinstance(target_parameter, float):
            target_parameter = torch.ones([reference_samples.shape[0]], device=reference_samples.device) * target_parameter
        if isinstance(base_parameter, float):
            base_parameter = torch.ones([reference_samples.shape[0]], device=reference_samples.device) * base_parameter
        if isinstance(sample_parameter, float):
            sample_parameter = torch.ones([reference_samples.shape[0]], device=reference_samples.device) * sample_parameter

        return self.compute_trade_loss(flow=flow,
                                      target_energy=target_energy,
                                      base_energy=base_energy,
                                      sample_energy=sample_energy,
                                      energy_derivative=energy_derivative,
                                      reference_samples=reference_samples,
                                      reference_conditions=reference_conditions,
                                      target_parameter=target_parameter,
                                      base_parameter=base_parameter,
                                      sample_parameter=sample_parameter,
                                      evaluation_samples=evaluation_samples,
                                      evaluation_conditions=evaluation_conditions)

    def sample_target_parameter(self, target_parameter_proposals, base_parameter):
        raise(NotImplementedError("This method must be implemented in the derived class."))

    def compute_trade_loss(self,
                          flow,
                          target_energy: Callable,
                          base_energy: Callable,
                          sample_energy: Callable,
                          energy_derivative: Callable,
                          reference_samples: torch.Tensor,
                          reference_conditions: list | tuple,
                          target_parameter: torch.Tensor,
                          base_parameter: torch.Tensor,
                          sample_parameter: torch.Tensor,
                          evaluation_samples: torch.Tensor,
                          evaluation_conditions: torch.Tensor)-> torch.Tensor:
        raise(NotImplementedError("This method must be implemented in the derived class."))

    @torch.no_grad()
    def compute_mean_energy_derivative(self,
                                       reference_samples,
                                       reference_conditions,
                                       target_energy,
                                       base_energy,
                                       log_prob_samples_sample_parameter,
                                       energy_derivative,
                                       target_parameter,
                                       base_parameter,
                                       sample_parameter):
        """
        target_parameter: The target parameter at which the derivative of the energy is computed.
        sample_parameter: The parameter at which the samples are drawn.
        base_parameter: The parameter at which the energies are computed. In most cases this is the parameter of the training data.
        """

        if (base_parameter is not None) and (not torch.allclose(target_parameter, base_parameter)) and (self.check_consistency_with == "ground_truth"):
            warn("A base parameter is set, to reweight the energies from, which incurs addiional cost but checking against the ground truth, \
                 which should already be accurate at the target parameter. Is this really what you want?.")

        # Option 1: Check consistency with self and reweight the energies from the base parameter where the model is accurate to the target parameter
        if base_parameter is not None and not torch.allclose(target_parameter, base_parameter):
            log_prob_samples_base_parameter = - base_energy(x = reference_samples, c = reference_conditions).squeeze()
            log_prob_samples_target_parameter = - target_energy(x = reference_samples, c = reference_conditions).squeeze()


            # Compute partition function using importance sampling
            log_w_i = self.get_log_weight(target_parameter, base_parameter, log_prob_samples_base_parameter, log_prob_samples_sample_parameter).squeeze()
            log_Z = torch.logsumexp(log_w_i, dim=0) - np.log(len(reference_samples))

            # Compute importance weights for the expectation values of the energies
            log_omega_i = log_w_i - log_Z
            samples_energy_derivative = energy_derivative(x = reference_samples, parameter = base_parameter, computed_energy=-log_prob_samples_base_parameter).squeeze()
            # Compute the expectation value of the energy
            EX_energy = (log_omega_i.exp() * samples_energy_derivative)
            EX_energy = EX_energy.mean()
        # Option 2: Check consistency with self and reweight directly from sample parameter to target parameter or reweight from flow energy to ground truth energy
        elif not torch.allclose(sample_parameter, target_parameter) or (self.check_consistency_with == "ground_truth" and self.take_samples_from == "flow"):
            log_prob_samples_target_parameter = - target_energy(x = reference_samples, c = reference_conditions).squeeze()

            log_w_i = self.get_log_weight(target_parameter, target_parameter, log_prob_samples_target_parameter, log_prob_samples_sample_parameter).squeeze()
            log_Z = torch.logsumexp(log_w_i, dim=0) - np.log(len(reference_samples))
            # Compute importance weights for the expectation values of the energies
            log_omega_i = log_w_i - log_Z
            samples_energy_derivative = energy_derivative(x = reference_samples, parameter = target_parameter, computed_energy=-log_prob_samples_target_parameter).squeeze()

            # Compute the expectation value of the energy
            EX_energy = (log_omega_i.exp() * samples_energy_derivative).mean()
        # Option 3: Samples are assumed to be from the ground truth distribution at the target parameter
        else:
            log_prob_samples_target_parameter = log_prob_samples_sample_parameter
            EX_energy = energy_derivative(x = reference_samples, parameter = target_parameter, computed_energy=-log_prob_samples_target_parameter).mean()
        return EX_energy

    @torch.no_grad()
    def sample_evaluation_points(self, flow, sample_parameter, **kwargs):
        if self.take_samples_from == "dataset":
            reference_samples = kwargs.get("samples")
            assert reference_samples is not None, "Samples must be provided if samples are taken from the dataset."
            reference_conditions = kwargs.get("conditions")
            assert reference_conditions is not None, "Conditions must be provided if samples are taken from the dataset."
            assert "energy_model" in kwargs, "Energy model must be provided if samples are taken from the dataset."
        elif self.take_samples_from == "flow":
            batch_size = kwargs.get("batch_size")
            assert batch_size is not None, "Batch size must be provided if samples are taken from the flow."
            reference_conditions = self.condition_sampler(batch_size)
            reference_samples = flow.sample(batch_size, reference_conditions, parameter = sample_parameter)

        if self.check_consistency_with == "ground_truth":
            sample_energy = partial(kwargs.get("energy_model").energy, **{self.target_parameter_name:sample_parameter})
        elif self.check_consistency_with == "self":
            sample_energy = partial(flow.energy, parameter=sample_parameter)

        if self.evaluation_mode == "raw_beta_0_samples":
            evaluation_points = (reference_samples).detach()
        elif self.evaluation_mode == "beta_0_samples_gaussian_noise":
            evaluation_points = (reference_samples + torch.randn_like(reference_samples) * self.evaluation_stddev).detach()
        elif self.evaluation_mode == "beta_0_samples_gaussian_noise_parameter_scaling":
            try:
                sigma = self.evaluation_stddev * torch.sqrt(sample_parameter)
                if sigma.numel() > 1:
                    sigma = sigma.reshape(-1, 1)
            except:
                sigma = self.evaluation_stddev * np.sqrt(sample_parameter)
            evaluation_points = (reference_samples + torch.randn_like(reference_samples) * sigma).detach()
        else:
            raise ValueError(f"Unknown evaluation mode {self.evaluation_mode}")

        return reference_samples, reference_conditions, evaluation_points, reference_conditions, sample_energy

    def get_target_energy(self, flow, parameter=None):
        if self.check_consistency_with == "ground_truth":
            target_energy = flow.get_energy_model()
            if parameter is not None:
                return partial(target_energy.energy, **{self.target_parameter_name:parameter})
            else:
                return target_energy
        elif self.check_consistency_with == "self":
            target_energy = flow
            if parameter is not None:
                return partial(target_energy.energy, **{"parameter":parameter})
            else:
                return target_energy
        else:
            raise ValueError(f"Unknown consistency check mode {self.check_consistency_with}")

    def target_energy_derivative(self, flow):
        if self.check_consistency_with == "self" and not self.target_parameter_name in ["temperature", "coordinates", "inverse_temperature", "likelihood_power"]:
            raise(NotImplementedError(f"Only temperature and coordinates support checking self-consistency for now. \
                                This is because flows only provide the full energy, but other parameter derivatives might require additional information. \
                                Implementation hint: Find out the subterms of the energy function via autograd, then subtract them to get the desired term."))
        ground_truth_energy = flow.get_energy_model()
        return partial(ground_truth_energy.derivative, wrt=self.target_parameter_name)

    def get_log_weight(self, target_parameter, base_parameter, log_prob_samples_base_parameter, log_prob_samples_sample_parameter):
        if torch.allclose(target_parameter, base_parameter):
            return log_prob_samples_base_parameter - log_prob_samples_sample_parameter
        elif self.target_parameter_name == "temperature":
            return (base_parameter / target_parameter) * log_prob_samples_base_parameter - log_prob_samples_sample_parameter
        elif self.target_parameter_name == "inverse_temperature":
            return (target_parameter / base_parameter) * log_prob_samples_base_parameter - log_prob_samples_sample_parameter
        else:
            raise RuntimeError(f"Cannot compute the log weights for the target parameter {self.target_parameter_name} from a base parameter to a different target parameter.")

    def reweight_target(self, target, base_parameter, target_parameter):
        if torch.allclose(base_parameter, target_parameter):
            return target
        elif self.target_parameter_name == "inverse_temperature":
            return target
        elif self.target_parameter_name == "temperature":
            return target * (base_parameter ** 2) / (target_parameter ** 2)
        else:
            raise RuntimeError(f"Cannot reweight the target gradient for target parameter {self.target_parameter_name} from a base parameter to a different target parameter.")


class EnergyBuffer():
    def __init__(self, max_energy_buffer_size=10000):
        self.energies = None
        self.parameters = None
        self.samples = None
        self.conditions = None
        self.max_energy_buffer_size = max_energy_buffer_size

    def update(self, reference_samples, reference_conditions, sample_energy, sample_parameter):
        computed_energy_sample_parameter = sample_energy(x = reference_samples, c = reference_conditions)

        if self.energies is None:
            # initialize buffer
            self.energies = computed_energy_sample_parameter
            self.parameters = sample_parameter
            self.samples = reference_samples
            self.conditions = reference_conditions
        else:
            self.energies = torch.cat([self.energies, computed_energy_sample_parameter], dim=0)[-self.max_energy_buffer_size:]
            self.parameters = torch.cat([self.parameters, sample_parameter], dim=0)[-self.max_energy_buffer_size:]
            self.samples = torch.cat([self.samples, reference_samples], dim=0)[-self.max_energy_buffer_size:]
            self.conditions = [torch.cat([c_0, c_1], dim=0)[-self.max_energy_buffer_size:] for c_0, c_1 in zip(self.conditions, reference_conditions)]


class TRADE_loss_continuous(TRADE_loss_base):
    def __init__(self, *args, use_energy_buffer=False, max_energy_buffer_size=10000, **kwargs):
        self.use_energy_buffer = use_energy_buffer
        if use_energy_buffer:
            self.energy_buffer = EnergyBuffer(max_energy_buffer_size)
            raise(NotImplementedError("Energy Buffer not yet fully implemented. Missing broadcasting of target/base parameter to all samples"))
        super().__init__(*args, **kwargs)

    def sample_target_parameter(self, target_parameter_proposals, base_parameter, **kwargs):
        return target_parameter_proposals

    @torch.enable_grad()
    def compute_trade_loss(self,
                            flow,
                            target_energy: Callable,
                            base_energy: Callable,
                            sample_energy: Callable,
                            energy_derivative: Callable,
                            reference_samples: torch.Tensor,
                            reference_conditions: list | tuple,
                            target_parameter: torch.Tensor,
                            base_parameter: torch.Tensor,
                            sample_parameter: torch.Tensor,
                            evaluation_samples: torch.Tensor,
                            evaluation_conditions: torch.Tensor)-> torch.Tensor:
        """
        Compute the TRADE loss for the target paramter.

        parameters:

            flow: The flow model.
            target_energy: The energy model at a given paramter.
            target_energy_derivative: The derivative of the energy model at a given parameter.
            reference_samples: The samples at the base paramter used for the computation of the expectation value of the energy derived with respect to the parameter.
            reference_conditions: The conditions of the reference samples.
            target_parameter: The target paramter.
            base_parameter: The base paramter at which the reference samples are drawn.
            evaluation_samples: The evaluation points, where the derivative of the flow is evaluated.
            evaluation_conditions: The conditions of the evaluation points.

        returns:

            loss: The TRADE loss.
        """
        target_parameter = target_parameter.squeeze()
        if base_parameter is not None:
            base_parameter = base_parameter.squeeze()
        sample_parameter = sample_parameter.squeeze()

        with torch.no_grad():

            if self.use_energy_buffer:
                self.energy_buffer.update(reference_samples, reference_conditions, sample_energy, sample_parameter)
                reference_samples = self.energy_buffer.samples
                reference_conditions = self.energy_buffer.conditions
                sample_parameter = self.energy_buffer.parameters
                log_prob_samples_sample_parameter = - self.energy_buffer.energies.squeeze()
            else:
                log_prob_samples_sample_parameter = - sample_energy(x = reference_samples, c = reference_conditions).squeeze()
            # Compute the mean energy
            EX_energy_derivative = self.compute_mean_energy_derivative(reference_samples=reference_samples,
                                                                       reference_conditions=reference_conditions,
                                                                       target_energy=target_energy,
                                                                       base_energy=base_energy,
                                                                       log_prob_samples_sample_parameter=log_prob_samples_sample_parameter,
                                                                       energy_derivative=energy_derivative,
                                                                       target_parameter=target_parameter,
                                                                       base_parameter=base_parameter,
                                                                       sample_parameter=sample_parameter)
            eval_energies = base_energy(x = evaluation_samples, c = evaluation_conditions).squeeze()
            eval_energies_derivative = energy_derivative(x = evaluation_samples, parameter = base_parameter, computed_energy=eval_energies).squeeze()
            # Compute the target
            target = EX_energy_derivative - eval_energies_derivative
            self.reweight_target(target, base_parameter, target_parameter)

        # Compute the gradient with respect to beta in beta_0
        target_parameter.requires_grad = True
        log_prob_eval_target = - flow.energy(x = evaluation_samples, c = evaluation_conditions, parameter = target_parameter).squeeze()
        grad = torch.autograd.grad(log_prob_eval_target.sum(), target_parameter, create_graph = True)[0].reshape(-1)

        # Comput the objective
        loss = reconstruction_loss(grad - target.detach(), mode = self.loss_mode, **self.reconstruction_kwargs)
        return loss


class TRADE_loss_grid(TRADE_loss_base):
    def __init__(self,
                 param_min:float,
                 param_max:float,
                 base_params: torch.tensor,
                 epsilon_causality_weight:float,
                 n_points_param_grid:int,
                 alpha_running_EX_A:float,
                 average_importance_weights:bool,
                 update_freq_running_average_EX_A:int,
                 alpha_running_loss:float,
                 n_samples_expectation_computation:int,
                 bs_expectation_computation:int,
                 init_param_grid_in_log_space:bool,
                 use_target_proposals:bool = False,
                 **kwargs)->None:

        """
        parameters:
            param_min:                          The minimal parameter value
            param_max:                          The maximal parameter value
            n_reps:                             The number of repetitions for the loss computation
            base_params:                        The base parameters at which the nll loss is computed
            residual_processing_parameters:     Parametersto compute the loss from the residuals
            epsilon_causality_weight:           The weight of the causality term in the loss
            n_points_param_grid:int,            The number of points in the parameter grid
            alpha_running_EX_A:                 The running average parameter for the expectation value of the derivate of the ground truth energy function with respect to the parameter
            average_importance_weights:         If True, the importance weights are averaged if they are inbetween of two base parameters
            update_freq_running_average_EX_A:   The frequency with which the running average of the expectation value is updated
            alpha_running_loss:                 The running average parameter for the loss
            n_samples_expectation_computation:  The number of samples used to compute the expectation values
            bs_expectation_computation:         The batch size for the computation of the expectation values
            init_param_grid_in_log_space:       If True, the parameter grid is initialized in log space
            use_target_proposals:               If True, the target parameter proposals are used for the computation of the loss
        """

        super().__init__(**kwargs)

        if self.take_samples_from == "dataset":
            raise(NotImplementedError("Taking samples from the dataset for computing the expected energy is not supported yet."))
        #Store settings
        self.epsilon_causality_weight = epsilon_causality_weight
        self.update_freq_running_average_EX_A = update_freq_running_average_EX_A
        self.n_samples_expectation_computation = n_samples_expectation_computation
        self.bs_expectation_computation = bs_expectation_computation

        #Set the parameter grid
        if init_param_grid_in_log_space:
            log_param_grid = torch.linspace(np.log(param_min),np.log(param_max),n_points_param_grid)
            self.param_grid = torch.exp(log_param_grid)

        else:
            self.param_grid = torch.linspace(param_min,param_max,n_points_param_grid)
        if base_params not in self.param_grid.to(base_params.device):
            self.param_grid = torch.cat((base_params, self.param_grid.to(base_params.device)), dim=0)

        #Ensure, that the parameters are sorted in ascending order
        self.param_grid = torch.nn.Parameter(torch.sort(self.param_grid,dim = 0)[0], requires_grad=False)

        #Get the indices of the base parameters
        self.n_points_param_grid = n_points_param_grid
        self.base_params = torch.nn.Parameter(base_params, requires_grad=False)

        #Intialize the running average stores for the expectation values
        self.EX_A = None
        self.alpha_running_EX_A = alpha_running_EX_A
        self.average_importance_weights = average_importance_weights

        #Counter for the number of calls of the loss function
        self.iteration = 0

        assert(self.alpha_running_EX_A >= 0.0)
        assert(self.alpha_running_EX_A <= 1.0)
        assert(self.epsilon_causality_weight >= 0.0)
        #Statistics for the loss in the individual bins
        self.loss_statistics = torch.nn.Parameter(torch.zeros_like(self.param_grid), requires_grad=False)
        self.alpha_running_loss = alpha_running_loss
        self.freq_update_causality_weights = 50
        self.log_causality_weights = None
        self.use_target_proposals = use_target_proposals

    def sample_target_parameter(self, target_parameter_proposals, base_parameter, **kwargs):
        if not self.use_target_proposals:
            batch_size = target_parameter_proposals.shape[0] if "batch_size" not in kwargs else kwargs["batch_size"]
            #Initial call: Uniformly sample the points
            if self.log_causality_weights is None:
                idx = torch.randint(low = 0,high = len(self.param_grid), size = (batch_size,))

            #Follow up calls: Sample the indices based on the distribution defined by the causality weights
            else:
                m = torch.distributions.categorical.Categorical(logits = self.log_causality_weights)
                idx = m.sample([batch_size])

            param_tensor = self.param_grid[idx]
        else:
            idx = self.parameter_to_idx(target_parameter_proposals.squeeze())
            param_tensor = self.param_grid[idx]
        return param_tensor

    def parameter_to_idx(self,parameter:torch.tensor)->torch.tensor:
        '''
        Convert a parameter to the closest index in the parameter grid

        parameters:
            parameter:  The parameter to be converted

        returns:
            idx:        The index of the closest parameter in the parameter grid
        '''
        assert(len(parameter.shape) == 1)

        return torch.argmin((self.param_grid.reshape(-1, 1) - parameter).abs(),dim = 0)

    @torch.enable_grad()
    def compute_trade_loss(self,
                          flow,
                          target_energy: Callable,
                          base_energy: Callable,
                          sample_energy: Callable,
                          energy_derivative: Callable,
                          reference_samples: torch.Tensor,
                          reference_conditions: list | tuple,
                          target_parameter: torch.Tensor,
                          base_parameter: torch.Tensor,
                          sample_parameter: torch.Tensor,
                          evaluation_samples: torch.Tensor,
                          evaluation_conditions: torch.Tensor)-> torch.Tensor:

        """
        This loss uses the stored expectation values for the derivative of the ground truth energy function with respect to the parameter to compute the loss. To do this, sampels are randomly drawn
        from a discrete distribution defined by the causality weights which are based on the running averages of the loss on the individual parameter grid points. The expectation of the derivative of
        the ground truth energy function with respect to the parameter is not updated in this method. However, the running averages of the loss are updated.

        parameters:
            INN:                The INN model
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor.

        returns:
            loss:               The parameter gradient at the given parameter
        """
        idx = self.parameter_to_idx(target_parameter.squeeze())

        with torch.no_grad():
            eval_energies = target_energy(x = evaluation_samples, c = evaluation_conditions).squeeze()
            eval_energies_derivative = energy_derivative(x = evaluation_samples, parameter = target_parameter, computed_energy=eval_energies).squeeze()
            target = (self.EX_A[idx].squeeze() - eval_energies_derivative)
        target_parameter.requires_grad_(True)


        # Compute the gradient with respect to beta in beta_0
        log_prob_eval_target = - flow.energy(x = evaluation_samples, c = evaluation_conditions, parameter = target_parameter).squeeze()
        grad = torch.autograd.grad(log_prob_eval_target.sum(), target_parameter, create_graph = True)[0].reshape(-1)

        loss = reconstruction_loss(grad - target.detach(), mode = self.loss_mode, **self.reconstruction_kwargs)

        with torch.no_grad():
            # Average the losses if there are multiple samples for the same parameter
            unique_idx, position_uniques, counts_uniques = torch.unique(idx, return_counts = True, return_inverse = True)
            target_tensor_loss = torch.zeros(unique_idx.shape[0], device=loss.device)
            target_tensor_loss.scatter_add_(0, position_uniques, loss)
            target_tensor_loss = target_tensor_loss / counts_uniques
            self.loss_statistics[unique_idx] = self.alpha_running_loss * self.loss_statistics[unique_idx] + (1.0 - self.alpha_running_loss) * target_tensor_loss

            if (self.log_causality_weights is None) or ((self.iteration % self.freq_update_causality_weights) == 0):
                self.log_causality_weights = self.compute_causality_weights_exponents(self.loss_statistics.detach(), self.param_grid)

        loss = loss.mean()
        return loss

    @torch.no_grad()
    def compute_causality_weights_exponents(self,loss:torch.tensor,param_tensor:torch.tensor)->torch.tensor:
        '''

        Compute the logarithms of the causality weights for the individual losses.

        Parameters:
            loss:               Tensor of shape [K] contaiing the loss values at the evaluated parameter values
            param_tensor:       Tensor of shape [K,self.bs] containing the parameter values at which the loss is evaluated. has to be sorted in ascending order

        returns:
            causality_weights:  Tensor of shape [K] containing the logarithms of the causality weights for the individual losses

        '''

        with torch.no_grad():
            causality_weights = torch.zeros_like(param_tensor)

            for i in range(len(param_tensor)):

                param_i = param_tensor[i]

                #get the index of the closest base parameter
                a = torch.argmin((self.base_params - param_i).abs()).item()
                param_base = self.base_params[a]

                idx_base = torch.where(param_tensor == param_base)[0].item()
                idx_parameter = i

                #Get the second closest base parameter
                if len(self.base_params) > 1:
                    mask = self.base_params != param_base
                    base_params_masked = self.base_params[mask]

                    b = torch.argmin((base_params_masked - param_i).abs()).item()
                    param_base_second = base_params_masked[b]
                    idx_base_second = torch.where(param_tensor == param_base_second)[0].item()

                #Get the loss weights based on the closest base parameter
                if idx_base < idx_parameter:
                    s1 = loss[idx_base:idx_parameter]

                elif idx_base > idx_parameter:
                    s1 = loss[idx_parameter+1:idx_base+1]

                else:
                    s1 = torch.zeros(1).to(param_i.device)

                exponent_closest = (-s1.sum() * self.epsilon_causality_weight).detach()

                if len(self.base_params) > 1:
                    #If applicable, get the loss weights based on the second closest base parameter, i.e. if the sample is between two base parameters
                    is_between = (param_base < param_i < param_base_second) or (param_base > param_i > param_base_second)

                    if is_between and self.average_importance_weights:
                        #Get the loss weights based on the closest base parameter
                        if idx_base_second < idx_parameter:
                            s2 = loss[idx_base_second:idx_parameter]

                            d_base_idx = idx_parameter - (idx_base_second + 1)

                        elif idx_base_second > idx_parameter:
                            s2 = loss[idx_parameter+1:idx_base_second+1]

                            d_base_idx = idx_base_second - 1 - idx_parameter

                        else:
                            raise ValueError("Not suported case for causality weights")

                        exponent_second = (-s2.sum() * self.epsilon_causality_weight).detach()

                        #Get the relative weigting based on the distance to the base parameter
                        d_base_base = abs(idx_base_second - idx_base) - 2
                        assert(d_base_base >= 0)
                        assert(d_base_idx >= 0)
                        assert(d_base_base >= d_base_idx)

                        k = d_base_idx / d_base_base

                        #Catch edge cases where the weights ignore one of the two contributions
                        if k == 1.0:
                            exponent = exponent_closest

                        elif k == 0.0:
                            exponent = exponent_second

                        else:
                            exponent = torch.logsumexp(torch.tensor([exponent_closest + np.log(k),exponent_second + np.log(1-k)]),0)

                        assert(exponent.shape == torch.Size([]))

                    else:
                        exponent = exponent_closest

                else:
                    exponent = exponent_closest

                causality_weights[i] = exponent

            causality_weights = causality_weights.detach()

            return causality_weights


    @torch.no_grad()
    def compute_causality_weights_exponents_vectorized(self,loss:torch.tensor, param_tensor:torch.tensor)->torch.tensor:
        '''

        Compute the logarithms of the causality weights for the individual losses.

        Parameters:
            loss:               Tensor of shape [K] contaiing the loss values at the evaluated parameter values
            param_tensor:       Tensor of shape [K,self.bs] containing the parameter values at which the loss is evaluated. has to be sorted in ascending order

        returns:
            causality_weights:  Tensor of shape [K] containing the logarithms of the causality weights for the individual losses

        '''

        causality_weights = torch.zeros(param_tensor.shape[0], device=param_tensor.device)

        a = torch.argmin(self.base_params.reshape(-1) - param_tensor.reshape(-1, 1), dim=1)
        param_base = self.base_params[a]
        idx_base = torch.where(param_tensor == param_base)[0]
        idx_parameter = torch.arange(len(param_tensor), device=param_tensor.device)

        if len(self.base_params) > 1:
            mask = self.base_params != param_base
            base_params_masked = self.base_params[mask]

            b = torch.argmin(self.base_params_masked.reshape(-1) - param_tensor.reshape(-1, 1), dim=1)
            param_base_second = base_params_masked[b]
            idx_base_second = torch.where(param_tensor == param_base_second)[0]

        s1 = torch.zeros_like(idx_base)

        mask_less = idx_base < idx_parameter
        mask_greater = idx_base > idx_parameter
        mask_equal = idx_base == idx_parameter

        if mask_less.any():
            s1[mask_less] = torch.stack([loss[idx_base[i]:idx_parameter[i]].sum() for i in torch.nonzero(mask_less)])

        if mask_greater.any():
            s1[mask_greater] = torch.stack([loss[idx_parameter[i]+1:idx_base[i]+1].sum() for i in torch.nonzero(mask_greater)])

        s1[mask_equal] = torch.zeros(mask_equal.sum(), device=loss.device)

        exponent_closest = (-s1 * self.epsilon_causality_weight).detach()

        if len(self.base_params) == 1 or not self.average_importance_weights:
            return exponent_closest

        is_between = torch.logical_or(param_base < param_tensor < param_base_second, param_base > param_tensor > param_base_second)

        idx_base_second_between = idx_base_second[is_between]
        idx_parameter_between = idx_parameter[is_between]
        idx_base_between = idx_base[is_between]

        s2 = torch.zeros_like(idx_base_second_between)
        d_base_idx = torch.zeros_like(idx_base_second_between)

        mask_less = idx_base_second_between < idx_parameter_between
        mask_greater = idx_base_second_between > idx_parameter_between
        mask_equal = idx_base_second_between == idx_parameter_between
        assert mask_equal.sum() == 0, "Parameter lands exactly on second closest base parameter. Duplicate base parameters?"

        if mask_less.any():
            s2[mask_less] = torch.stack([loss[idx_base_second_between[i]:idx_parameter_between[i]].sum() for i in torch.nonzero(mask_less)])
            d_base_idx[mask_less] = idx_parameter_between[mask_less] - (idx_base_second_between[mask_less] + 1)

        if mask_greater.any():
            s2[mask_greater] = torch.stack([loss[idx_parameter_between[i]+1:idx_base_second_between[i]+1].sum() for i in torch.nonzero(mask_greater)])
            d_base_idx[mask_greater] = idx_base_second_between[mask_less] - (idx_parameter_between[mask_less] + 1)

        exponent_second = (-s2 * self.epsilon_causality_weight).detach()

        #Get the relative weigting based on the distance to the base parameter
        d_base_base = abs(idx_base_second_between - idx_base_between) - 2
        assert torch.all(d_base_base >= 0)
        assert torch.all(d_base_idx >= 0)
        assert torch.all(d_base_base >= d_base_idx)

        k = d_base_idx / d_base_base

        exponent = torch.zeros_like(idx_base)

        exponent[is_between][k == 1.0] = exponent_closest[is_between][k == 1.0]
        exponent[is_between][k == 0.0] = exponent_second[k == 0.0]
        exponent[~is_between] = exponent_closest[~is_between]

        idx_interpolate = torch.where(torch.logical_and(k != 1.0, k!= 0.0))[0]

        exponent[is_between][idx_interpolate] = torch.logsumexp(torch.stack(
        (exponent_closest[is_between][idx_interpolate] + np.log(k[idx_interpolate]), \
        exponent_second[is_between][idx_interpolate] + np.log(1-k[idx_interpolate])), dim=0), dim=0)

        return exponent.detach()

    @torch.no_grad()
    def get_expectation_values(self, flow, target_energy, target_energy_derivative)->torch.tensor:
        """
        Compute the expectation values of the derivative of the ground truth energy function with respect to the parameter at the parameter grid points

        parameters:
            INN:                The INN model
            n_samples:          The number of samples used to compute the expectation values

        returns:
            EX_A_temporary:    The approximated expectation values of the derivative of the ground truth energy function with respect to the parameter at the parameter grid points
        """

        EX_A_temporary = torch.zeros_like(self.param_grid)
        n_batches = ceil(self.n_samples_expectation_computation / self.bs_expectation_computation)

        for i in tqdm(range(len(self.param_grid)), desc = "Computing expectation values", leave=False):

            param = self.param_grid[i]

            log_p_x_proposal_INN = torch.zeros(self.bs_expectation_computation * n_batches, device = param.device)
            log_p_x_proposal_GT = torch.zeros(self.bs_expectation_computation* n_batches, device = param.device)
            A_proposal = torch.zeros(self.bs_expectation_computation* n_batches, device = param.device)

            for j in range(n_batches):
                c_proposal_i = self.condition_sampler(self.bs_expectation_computation)
                x_proposal_i = flow.sample(self.bs_expectation_computation, c_proposal_i, parameter = param)

                #2) Compute the derivative of the ground truth energy function with respect to the parameter at the evaluation points
                A_proposal_i = target_energy_derivative(x = x_proposal_i, parameter = param).squeeze()
                assert(A_proposal_i.shape == torch.Size([self.bs_expectation_computation])), f"{A_proposal_i.shape} is not the same shape as {torch.Size([self.bs_expectation_computation])}"

                #3) Compute the log likelihood of the samples under the INN distribution and the ground truth distribution
                log_p_x_proposal_INN_i    = - flow.energy(x = x_proposal_i, c = c_proposal_i, parameter = param).squeeze()
                if self.check_consistency_with == "self":
                    log_p_x_proposal_GT_i     = - target_energy.energy(x = x_proposal_i, c = c_proposal_i, parameter = param).squeeze()
                else:
                    log_p_x_proposal_GT_i     = - target_energy.energy(x = x_proposal_i, c = c_proposal_i, **{self.target_parameter_name:param}).squeeze()

                log_p_x_proposal_GT[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = log_p_x_proposal_GT_i
                log_p_x_proposal_INN[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = log_p_x_proposal_INN_i
                A_proposal[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = A_proposal_i


            #4) compute the log likelihood ratios
            log_w = log_p_x_proposal_GT - log_p_x_proposal_INN

            #5) compute the log parition function
            log_Z = torch.logsumexp(log_w,dim = 0) - np.log(self.bs_expectation_computation * n_batches)

            #6) Compute the importance weights
            log_omega = log_w - log_Z

            #7) Compute the sample based expectation value of the energy
            EX_A_temporary[i] = (A_proposal * log_omega.exp()).mean()

        return EX_A_temporary

    def __call__(self, flow, **kwargs) -> torch.Tensor:
        #Initialize the running average store for the expectation values at the first call
        target_energy = self.get_target_energy(flow)
        target_energy_derivative = self.target_energy_derivative(flow)

        if self.EX_A is None:
            self.iter_start = self.iteration
            self.EX_A =  self.get_expectation_values(flow, target_energy, target_energy_derivative)
        elif (self.iteration + 1 - self.iter_start) % self.update_freq_running_average_EX_A == 0 and self.training:
            update_EX_A = self.get_expectation_values(flow, target_energy, target_energy_derivative)
            self.EX_A = self.alpha_running_EX_A * self.EX_A + (1.0 - self.alpha_running_EX_A) * update_EX_A
        if self.training:
            self.iteration += 1

        return super().__call__(flow, **kwargs)

##############################################################################################
# Interface to initialize the data free loss functions
##############################################################################################

class DataFreeLossFactory():
    def __init__(self):
        pass

    def create(self,key,config):

        #Check consistency
        assert(key == config["config_training"]["data_free_loss_mode"])

        if key == "reverse_KL":
            loss_model = Objective_reverse_KL(
                log_p_target=log_p_target_dict[config["config_training"]["log_p_target_name"]],
                log_p_target_kwargs=config["config_training"]["log_p_target_kwargs"],
                device = config["device"],
                **config["config_training"]["loss_model_params"]
            )
            print("Initialize loss model of type <Objective_reverse_KL>")

        elif key == "PINF_local_Ground_Truth_one_param_V2":
            loss_model = Objective_PINF_local_Ground_Truth_one_param_V2(
                S = S_dict[config["config_data"]["data_set_name"]],
                dSdparam = dS_dparam_dict[config["config_data"]["data_set_name"]],
                device = config["device"],
                **config["config_training"]["loss_model_params"]               
            )
            print("Initialize loss model of type <Objective_PINF_local_Ground_Truth_one_param_V2>")

        elif key == "PINF_parallel_Ground_Truth_one_param_V2":

            loss_model = Objective_PINF_parallel_Ground_Truth_one_param_V2(
                S = S_dict[config["config_data"]["data_set_name"]],
                dSdparam = dS_dparam_dict[config["config_data"]["data_set_name"]],
                device = config["device"],
                **config["config_training"]["loss_model_params"]
            )

            print("Initialize loss model of type <Objective_PINF_parallel_Ground_Truth_one_param_V2>")
        
        elif key == "PINF_parallel_Ground_Truth_one_param":

            loss_model = Objective_PINF_parallel_Ground_Truth_one_param(
                S = S_dict[config["config_data"]["data_set_name"]],
                dSdparam = dS_dparam_dict[config["config_data"]["data_set_name"]],
                device = config["device"],
                **config["config_training"]["loss_model_params"]
            )

            print("Initialize loss model of type <Objective_PINF_parallel_Ground_Truth_one_param>")           
        
        else: 
            loss_model = None
            print("No data free loss model is used")

        return loss_model

##############################################################################################
# Utils for loss computation
##############################################################################################

def get_beta(t:int,t_burn_in:int,t_full:int,beta_star:float,beta_min:float,beta_max:float,**kwargs)->Tuple[float,float,float]:
        """
        Sample the inverse temperature from an interval depending on the current time step

        parameters:
            t:              The current time step
            t_burn_in:      The lenght of the burin in phase
            t_full:         The number of time steps it takes to reach the full range
            beta_star:      The inverse base temperature
            beta_min:       The minimal inverse temperature
            beta_max:       The maximal inverse temperature

        returns:
            beta:           The sampled inverse temperature
            left:           The lower boundary of the interval from which the inverse temperature is sampled
            right:          The upper boundary of the interval from which the inverse temperature is sampled
        """

        assert(isinstance(beta_star,float))
        assert(isinstance(beta_min,float))
        assert(isinstance(beta_max,float))
        assert(t_burn_in <= t_full)
        assert(beta_star >= beta_min)
        assert(beta_star <= beta_max)
        assert(beta_star > 0.0)
        assert(beta_max > 0.0)
        assert(beta_min > 0.0)

        #Burn in Phase return the inverse base temperature
        if t < t_burn_in:
            beta = beta_star
            left = beta_star
            right = beta_star

        #Randomly sample the inverse temperature from the interval
        else:
            
            #Compute the linear interpolation factor
            #use full range from the begining or directly after the burn in phase
            if (t_full == 0) or (t_full == t_burn_in): l = 1.0
            
            #There is a finite ramp up phase
            else:
                l = (min(t,t_full)-t_burn_in) / (t_full-t_burn_in)

            #Sample the inverse temperature uniformly from the interval [beta_star - l * beta_min, beta_star + l * beta_max]
            if kwargs["mode"] == "linear":

                left = beta_star - (beta_star -beta_min) * l
                right = beta_star + (beta_max -beta_star) * l

                beta = (right - left) * torch.rand(1).item() + left

            #Sample the logarithm of the inverse temperature uniformly from the interval [log(beta_star) - l * log(beta_min), log(beta_star) + l * log(beta_max)]
            elif kwargs["mode"] == "log-linear":

                beta_min_log = np.log(beta_min)
                beta_max_log = np.log(beta_max)
                beta_0_log = np.log(beta_star)

                log_left = beta_0_log - (beta_0_log - beta_min_log) * l
                log_right = beta_0_log + (beta_max_log - beta_0_log) * l

                log_beta_t = (log_right - log_left) * torch.rand(1).item() + log_left

                beta = np.exp(log_beta_t)
                left = np.exp(log_left)
                right = np.exp(log_right)

            else:
                raise ValueError("Sampling mode for inverse temperature not recognized")
            
        return beta,left,right

def get_loss_from_residuals(residuals:torch.Tensor,dim:int = 0,weight_tensor:torch.tensor = None,**kwargs) -> torch.Tensor:
    """
    Compute the loss from the residuals

    parameters:
        residuals:      One dimensional tensor containing the residuals
        kwargs:         Additional parameters for the loss computation, must contain the key "mode" which specifies the loss function used to compute the loss
        weight_tensor:  Weights for each loss contributions.

    returns:
        loss:       The loss
    """

    #Mean squared error
    if kwargs["mode"] == "MSE":
        loss = residuals.pow(2)

    #Mean absolute error
    elif kwargs["mode"] == "MAE":
        loss = residuals.abs()

    #Huber loss
    elif kwargs["mode"] == "Huber":
        loss = huber_loss(residuals,delta = kwargs["delta"])
    
    else:
        raise ValueError("Loss computation mode not recognized")
    
    if weight_tensor is None:
        return loss.mean(dim = dim)

    else:
        assert(loss.shape == weight_tensor.shape)

        return (loss * weight_tensor).mean(dim = dim)

def huber_loss(x:torch.Tensor,delta:float) -> torch.Tensor:
    """
    Compute the Huber loss

    parameters:
        x       The residuals
        delta   The threshold for the Huber loss

    returns:
        h: The Huber loss
    """

    assert(isinstance(delta,float))

    h = torch.where(x.abs() < delta,0.5 * x.pow(2),delta * (x.abs() - 0.5 * delta))
    
    return h

##############################################################################################
# Grid-based TRADE
##############################################################################################

class Objective_PINF_parallel_Ground_Truth_one_param():
    def __init__(self,
                 param_min:float,
                 param_max:float,
                 S:Callable,
                 S_kwargs:Dict,
                 dSdparam:Callable,
                 dSdparam_kwargs:Dict,
                 n_reps:int,
                 base_params:torch.tensor,
                 device:str,
                 epsilon_causality_weight:float,
                 n_points_param_grid:int,
                 residual_processing_parameters:Dict,
                 alpha_running_EX_A:float,
                 average_importance_weights:bool,
                 update_freq_running_average_EX_A:int,
                 bs:int,
                 alpha_running_loss:float,
                 n_samples_expectation_computation:int,
                 bs_expectation_computation:int,
                 init_param_grid_in_log_space:bool,
                 n_epochs:int,
                 n_batches_per_epoch:int,
                 epsilon_causality_decay_factor:float,
                 epsilon_reduction_factor_inbetween:float
                 )->None:

        """
        parameters:
            param_min:                          The minimal parameter value
            param_max:                          The maximal parameter value
            S:                                  The ground truth function in the exponent of the Gibbs distribution
            S_kwargs:                           Additional parameters for the ground truth energy function
            dSdparam:                           The derivative of the ground truth function with respect to the parameter
            dSdparam_kwargs:                    Additional parameters for the derivative of the ground truth energy function
            n_reps:                             The number of repetitions for the loss computation
            base_params:                        The base parameters at which the nll loss is computed
            device:                             The device on which the computation is performed
            residual_processing_parameters:     Parametersto compute the loss from the residuals
            bs:                                 The batch size for the computation of the loss
            epsilon_causality_weight:           The weight of the causality term in the loss
            n_points_param_grid:int,            The number of points in the parameter grid
            alpha_running_EX_A:                 The running average parameter for the expectation value of the derivate of the ground truth energy function with respect to the parameter
            average_importance_weights:         If True, the importance weights are averaged if they are inbetween of two base parameters
            update_freq_running_average_EX_A:   The frequency with which the running average of the expectation value is updated
            alpha_running_loss:                 The running average parameter for the loss
            n_samples_expectation_computation:  The number of samples used to compute the expectation values
            bs_expectation_computation:         The batch size for the computation of the expectation values
            init_param_grid_in_log_space:       If True, the parameter grid is initialized in log space
            n_epochs:                           Number of epochs for which this loss is applied
            n_batches_per_epoch:                Number of optimization steps per epoch
            epsilon_causality_decay_factor:     Final ratio of epsilon at the end of the training
            epsilon_reduction_factor_inbetween: Factor by which epsilon is changed inbetween two base parameters
        """

        print("*********************************************************************************************")
        print("Use class 'Objective_PINF_parallel_Ground_Truth_one_param'")
        print("*********************************************************************************************")


        T = n_epochs * n_batches_per_epoch
        self.tau_cw = - np.log(epsilon_causality_decay_factor) / T

        #Store settings
        self.device = device
        self.bs = bs
        self.residual_processing_parameters = residual_processing_parameters

        self.S = S
        self.dSdparam = dSdparam
        self.S_kwargs = S_kwargs
        self.dSdparam_kwargs = dSdparam_kwargs
        self.n_reps = n_reps
        self.epsilon_causality_weight = epsilon_causality_weight

        self.update_freq_running_average_EX_A = update_freq_running_average_EX_A
        self.n_samples_expectation_computation = n_samples_expectation_computation
        self.bs_expectation_computation = bs_expectation_computation

        #Set the parameter grid
        if init_param_grid_in_log_space:
            log_param_grid = torch.linspace(np.log(param_min),np.log(param_max),n_points_param_grid).reshape(-1,1)
            self.param_grid = torch.exp(log_param_grid)
        
        else:
            self.param_grid = torch.linspace(param_min,param_max,n_points_param_grid).reshape(-1,1)

        self.param_grid = torch.cat((base_params.reshape(-1,1),self.param_grid),0)

        #remove duplicates
        unique_elements = torch.unique(self.param_grid)
        self.param_grid = unique_elements.reshape(-1,1)

        #Ensure, that the parameters are sorted in ascending order
        self.param_grid,_ = torch.sort(self.param_grid,dim = 0)

        #Get the indices of the base parameters
        self.n_points_param_grid = n_points_param_grid
        self.base_params = base_params

        #Intialize the running average stores for the expectation values
        self.EX_A = None
        self.alpha_running_EX_A = alpha_running_EX_A
        self.average_importance_weights = average_importance_weights

        #Counter for the number of calls of the loss function
        self.iteration = 0

        assert(self.alpha_running_EX_A >= 0.0)
        assert(self.alpha_running_EX_A <= 1.0)
        assert(self.epsilon_causality_weight >= 0.0)
        #assert(self.param_grid.shape == torch.Size([len(base_params)+n_points_param_grid,1]))

        #Statistics for the loss in the individual bins
        self.loss_statistics = torch.zeros(len(self.param_grid))
        self.alpha_running_loss = alpha_running_loss
        self.freq_update_causality_weights = 50
        self.log_causality_weights = None

        #########
        self.multiplyer = torch.ones_like(self.param_grid)
        
        for i,param in enumerate(self.param_grid):

            #Check if it is inbetween to of the base parameters
            mask_1 = (param < self.base_params).sum().item()
            mask_2 = (param > self.base_params).sum().item()

            flag = mask_1 * mask_2

            if flag == 1:
                self.multiplyer[i] = epsilon_reduction_factor_inbetween

    def get_loss(self,INN:INN_Model,get_eval_points:Callable)->torch.Tensor:

        """
        This loss uses the stored expectation values for the derivative of the ground truth energy function with respect to the parameter to compute the loss. To do this, sampels are randomly drawn 
        from a discrete distribution defined by the causality weights which are based on the running averages of the loss on the individual parameter grid points. The expectation of the derivative of 
        the ground truth energy function with respect to the parameter is not updated in this method. However, the running averages of the loss are updated.

        parameters:
            INN:                The INN model
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor.

        returns:
            loss:               The temperature scaling loss at the given inverse temperature
        """

        #############################################################################################################
        #1) Get randomly select a batch of parameters for the evaluation
        #############################################################################################################

        #Initial call: Uniformly sample the points
        if self.log_causality_weights is None:
            print("Initial call")
            idx = torch.randint(low = 0,high = len(self.param_grid),size = (self.bs,))

        #Follow up calls: Sample the indices based on the distribution defined by the causality weights 
        else:
            m = torch.distributions.categorical.Categorical(logits = self.log_causality_weights)
            idx = m.sample([self.bs]).cpu()
        
        assert(idx.shape == torch.Size([self.bs]))

        param_tensor = self.param_grid[idx].to(self.device)
        assert(param_tensor.shape == torch.Size([self.bs,1]))

        #############################################################################################################
        #2) Compute the target for the logarithm of the INN distribution
        #############################################################################################################
        with torch.no_grad():

            INN.train(False)

            #2a) Get evaluation points at which the loss is evaluated
            x_eval = get_eval_points(beta_tensor = param_tensor)

            #2b) Compute the ground truth energies of the evaluation points
            A_eval = self.dSdparam(x_eval,**self.dSdparam_kwargs).reshape(-1,1)
            assert(A_eval.shape == param_tensor.shape)

            #2c) Compute the target
            target = (self.EX_A[idx] - A_eval).detach()
            assert(self.EX_A[idx].shape == torch.Size([self.bs,1]))
            assert(target.shape == torch.Size([self.bs,1]))

            INN.train(True)

        #############################################################################################################
        #3) Compute the gradient of the INN ditsribution with respect to the parameter
        #############################################################################################################
        param_tensor.requires_grad_(True)

        #3a) Compute the log likelihood of the evaluation points under the INN distribution
        log_p_x_eval = INN.log_prob(x_eval,param_tensor)

        #3b) Compute the gradient of the log likelihood of the evaluation points under the INN distribution with respect to the parameter
        grad = torch.autograd.grad(log_p_x_eval.sum(),param_tensor,create_graph=True)[0]
        assert(grad.shape == param_tensor.shape)
        assert(grad.shape == target.shape)

        #############################################################################################################
        #4)Compute loss for each of the evaluation points
        #############################################################################################################
        
        #4a) Get the residuals
        residuals = grad - target.detach()
        assert(residuals.shape == param_tensor.shape)

        #4b) Compute the loss based on the residuals
        loss = get_loss_from_residuals(residuals,dim = 1,**self.residual_processing_parameters)
        assert(loss.shape == torch.Size([param_tensor.shape[0]]))

        #############################################################################################################
        #5) Update the running averages of the loss on the evaluated grid points
        #############################################################################################################

        #5a) Average the losses if there are multiple samples for the same parameter
        unique_idx,position_uniques,counts_uniques = torch.unique(idx,return_counts = True,return_inverse = True)

        target_tensor_loss = torch.zeros(unique_idx.shape[0])
        target_tensor_loss.scatter_add_(0, position_uniques, loss.cpu().detach()) / counts_uniques
        target_tensor_loss = target_tensor_loss / counts_uniques

        #5b) Compute weights depending on the number of samples for the same parameter
        alphas_reweighted = self.alpha_running_loss**counts_uniques  

        #5c) Compute running average
        assert(self.loss_statistics[unique_idx].shape == target_tensor_loss.shape)
        assert(self.loss_statistics[unique_idx].shape == alphas_reweighted.shape)

        w_old = alphas_reweighted
        w_new = 1.0 - w_old

        assert(w_old.shape == alphas_reweighted.shape)
        assert(w_new.shape == w_old.shape)

        self.loss_statistics[unique_idx.cpu()] = w_old * self.loss_statistics[unique_idx] + w_new * target_tensor_loss.cpu()
        assert(self.loss_statistics.shape == torch.Size([len(self.param_grid)]))

        #############################################################################################################
        #6) If applicable, update the logarithms of the causality weights
        #############################################################################################################
        if (self.log_causality_weights is None) or ((self.iteration % self.freq_update_causality_weights) == 0):
            self.log_causality_weights = self.compute_causality_weights_exponents(self.loss_statistics.detach(),self.param_grid)

        #############################################################################################################
        #7) Aggregate the loss
        #############################################################################################################
        loss = loss.mean()
        assert(loss.shape == torch.Size([]))

        return loss
    
    def compute_causality_weights_exponents(self,loss:torch.tensor,param_tensor:torch.tensor)->torch.tensor:
        '''

        Compute the logarithms of the causality weights for the individual losses.

        Parameters:
            loss:               Tensor of shape [K] contaiing the loss values at the evaluated parameter values
            param_tensor:       Tensor of shape [K,self.bs] containing the parameter values at which the loss is evaluated. has to be sorted in ascending order

        returns:
            causality_weights:  Tensor of shape [K] containing the logarithms of the causality weights for the individual losses
        
        '''

        assert(len(loss.shape) == 1)
        assert(len(param_tensor.shape) == 2)
        assert(loss.shape[0] == param_tensor.shape[0])

        #Compute the epsilon used for the causality weights
        self.epsilon_t = self.epsilon_causality_weight * np.exp(- self.tau_cw * (self.iteration - self.iter_start))

        with torch.no_grad():
            causality_weights = torch.zeros(param_tensor.shape[0]).to(self.device)

            for i in range(len(param_tensor)):

                param_i = param_tensor[i][0]

                #get the index of the closest base parameter
                a = torch.argmin((self.base_params.cpu().detach() - param_i.cpu().detach()).abs()).item()
                param_base = self.base_params[a]

                idx_base = torch.where(param_tensor[:,0] == param_base)[0].item()
                idx_parameter = i

                #Get the second closest base parameter
                if len(self.base_params) > 1:
                    mask = self.base_params != param_base
                    base_params_masked = self.base_params[mask]

                    b = torch.argmin((base_params_masked.cpu().detach() - param_i.cpu().detach()).abs()).item()
                    param_base_second = base_params_masked[b]
                    idx_base_second = torch.where(param_tensor[:,0] == param_base_second)[0].item()

                #Get the loss weights based on the closest base parameter
                if idx_base < idx_parameter:
                    s1 = loss[idx_base:idx_parameter]

                elif idx_base > idx_parameter:
                    s1 = loss[idx_parameter+1:idx_base+1]

                else:
                    s1 = torch.zeros(1).to(self.device)

                exponent_closest = (-s1.sum() * self.epsilon_t).detach()

                if len(self.base_params) > 1:
                    #If applicable, get the loss weights based on the second closest base parameter, i.e. if the sample is between two base parameters
                    is_between = (param_base < param_i < param_base_second) or (param_base > param_i > param_base_second)

                    if is_between and self.average_importance_weights:
                        #Get the loss weights based on the closest base parameter
                        if idx_base_second < idx_parameter:
                            s2 = loss[idx_base_second:idx_parameter]

                            d_base_idx = idx_parameter - (idx_base_second + 1)

                        elif idx_base_second > idx_parameter:
                            s2 = loss[idx_parameter+1:idx_base_second+1]

                            d_base_idx = idx_base_second - 1 - idx_parameter

                        else:
                            raise ValueError("Not suported case for causality weights")

                        exponent_second = (-s2.sum() * self.epsilon_t).detach()

                        #Get the relative weigting based on the distance to the base temperature
                        d_base_base = abs(idx_base_second - idx_base) - 2
                        assert(d_base_base >= 0)
                        assert(d_base_idx >= 0)
                        assert(d_base_base >= d_base_idx)

                        k = d_base_idx / d_base_base

                        #Catch edge cases where the weights ignore one of the two contributions
                        if k == 1.0:
                            exponent = exponent_closest
                        
                        elif k == 0.0:
                            exponent = exponent_second
                        
                        else:
                            exponent = torch.logsumexp(torch.tensor([exponent_closest + np.log(k),exponent_second + np.log(1-k)]),0)

                        assert(exponent.shape == torch.Size([]))

                    else:
                        exponent = exponent_closest

                else:
                    exponent = exponent_closest
                
                causality_weights[i] = exponent

            causality_weights = causality_weights.detach()

            ###
            assert(causality_weights.shape == self.multiplyer.squeeze().shape)
            causality_weights = causality_weights * self.multiplyer.squeeze().to(causality_weights.device)

            return causality_weights
    
    def get_expectation_values(self,INN:INN_Model,n_samples:int)->torch.tensor:
        """
        Compute the expectation values of the derivative of the ground truth energy function with respect to the parameter at the parameter grid points

        parameters:
            INN:                The INN model
            n_samples:          The number of samples used to compute the expectation values

        returns:
            EX_A_temporaray:    The approximated expectation values of the derivative of the ground truth energy function with respect to the parameter at the parameter grid points
        """

        INN.train(False)
        

        with torch.no_grad():

            EX_A_temporaray = torch.zeros(len(self.param_grid),1).to(self.device)
            n_batches = int(self.n_samples_expectation_computation / self.bs_expectation_computation)

            for i in tqdm(range(len(self.param_grid))):

                param = self.param_grid[i]

                log_p_x_proposal_INN = torch.zeros(self.bs_expectation_computation * n_batches).to(self.device)
                log_p_x_proposal_GT = torch.zeros(self.bs_expectation_computation* n_batches).to(self.device)
                A_proposal = torch.zeros(self.bs_expectation_computation* n_batches).to(self.device)

                for j in range(n_batches):

                    #1) Get samples from the INN
                    x_proposal_i = INN.sample(n_samples = self.bs_expectation_computation,beta_tensor = param.item())

                    #2) Compute the derivative of the ground truth energy function with respect to the parameter at the evaluation points
                    A_proposal_i = self.dSdparam(x_proposal_i,**self.dSdparam_kwargs)
                    assert(A_proposal_i.shape == torch.Size([self.bs_expectation_computation]))

                    #3) Compute the log likelihood of the samples under the INN distribution and the ground truth distribution
                    log_p_x_proposal_INN_i    = INN.log_prob(x_proposal_i,param.item())
                    log_p_x_proposal_GT_i     = - self.S(x_proposal_i,param.item(),**self.S_kwargs)

                    assert(log_p_x_proposal_GT_i.shape == torch.Size([self.bs_expectation_computation]))
                    assert(log_p_x_proposal_INN_i.shape == torch.Size([self.bs_expectation_computation]))

                    log_p_x_proposal_GT[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = log_p_x_proposal_GT_i
                    log_p_x_proposal_INN[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = log_p_x_proposal_INN_i
                    A_proposal[j*self.bs_expectation_computation:(j+1)*self.bs_expectation_computation] = A_proposal_i

                assert(log_p_x_proposal_INN.shape == log_p_x_proposal_GT.shape)
                assert(log_p_x_proposal_INN.shape == torch.Size([n_samples]))

                #4) compute the log likelihood ratios
                log_w = log_p_x_proposal_GT - log_p_x_proposal_INN
                assert(log_w.shape == torch.Size([n_samples]))

                #5) compute the log parition function
                log_Z = torch.logsumexp(log_w,dim = 0) - np.log(n_samples)
                assert(log_Z.shape == torch.Size([]))

                #6) Compute the importance weights
                log_omega = log_w - log_Z
                assert(log_omega.shape == A_proposal.shape)

                #7) Compute the sample based expectation value of the energy
                EX_A_temporaray[i] = (A_proposal * log_omega.exp()).mean().item()
                
            assert(EX_A_temporaray.shape == torch.Size([len(self.param_grid),1]))
    
            INN.train(True)

            return EX_A_temporaray
        
    def __call__(self,INN,epoch,get_eval_points,logger = None)->torch.Tensor:
        """
        Compute the TS-PINF loss

        parameters:
            INN:                The INN model
            epoch:              The current epoch
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor
            logger:             The logger for the loss

        returns:
            loss:               The temperature scaling loss
        """

        #############################################################################################################
        #Update the running average store for the expectation values
        #############################################################################################################

        #Initialize the running average store for the expectation values at the first call
        if self.EX_A is None:

            self.iter_start = self.iteration

            print("Initialize running average store for expectation values")
            self.EX_A =  self.get_expectation_values(INN = INN,n_samples = self.n_samples_expectation_computation)
            print("done")

        #Update the running average store for the expectation values
        elif (self.iteration - self.iter_start) % self.update_freq_running_average_EX_A == 0:
            print("Update running average store for expectation values")
            update_EX_A = self.get_expectation_values(INN = INN,n_samples = self.n_samples_expectation_computation)
            self.EX_A = self.alpha_running_EX_A * self.EX_A + (1.0 - self.alpha_running_EX_A) * update_EX_A

        #############################################################################################################
        #Compute the loss
        #############################################################################################################
            
        #Count the number of evaluations
        counter = 0
        loss = torch.zeros(1).to(self.device)

        for i in range(self.n_reps):

            loss_i = self.get_loss(INN = INN,get_eval_points = get_eval_points)
            loss = loss + loss_i
            counter += 1

        loss = loss / counter

        logger.experiment.add_scalar(f"metadata/loss_model_internal_iteratoins",self.iteration,self.iteration)
        logger.experiment.add_scalar(f"parameters/epsilon_causality_weight",self.epsilon_t,self.iteration)

        return loss

class Objective_PINF_parallel_Ground_Truth_one_param_V2(Objective_PINF_parallel_Ground_Truth_one_param):
    """
    Basically teh same as the default implementation, but the distance between the grid points is now considered in the computation of the causality weights.
    """

    def compute_causality_weights_exponents(self,loss:torch.tensor,param_tensor:torch.tensor)->torch.tensor:
        '''

        Compute the logarithms of the causality weights for the individual losses.

        Parameters:
            loss:               Tensor of shape [K] contaiing the loss values at the evaluated parameter values
            param_tensor:       Tensor of shape [K,self.bs] containing the parameter values at which the loss is evaluated. has to be sorted in ascending order

        returns:
            causality_weights:  Tensor of shape [K] containing the logarithms of the causality weights for the individual losses
        '''

        assert(len(loss.shape) == 1)
        assert(len(param_tensor.shape) == 2)
        assert(loss.shape[0] == param_tensor.shape[0])

        #Compute the epsilon used for the causality weights
        self.epsilon_t = self.epsilon_causality_weight * np.exp(- self.tau_cw * (self.iteration - self.iter_start))

        with torch.no_grad():
            causality_weights = torch.zeros(param_tensor.shape[0]).to(self.device)

            for i in range(len(param_tensor)):

                param_i = param_tensor[i][0]

                #get the index of the closest base parameter
                a = torch.argmin((self.base_params.cpu().detach() - param_i.cpu().detach()).abs()).item()
                param_base = self.base_params[a]

                idx_base = torch.where(param_tensor[:,0] == param_base)[0].item()
                idx_parameter = i

                #Get the second closest base parameter
                if len(self.base_params) > 1:
                    mask = self.base_params != param_base
                    base_params_masked = self.base_params[mask]

                    b = torch.argmin((base_params_masked.cpu().detach() - param_i.cpu().detach()).abs()).item()
                    param_base_second = base_params_masked[b]
                    idx_base_second = torch.where(param_tensor[:,0] == param_base_second)[0].item()

                #Get the loss weights based on the closest base parameter
                if idx_base < idx_parameter:
                    s1 = loss[idx_base:idx_parameter]
                    grid_distances1 = torch.abs(self.param_grid.squeeze()[idx_base+1:idx_parameter+1] - self.param_grid.squeeze()[idx_base:idx_parameter])

                elif idx_base > idx_parameter:
                    s1 = loss[idx_parameter+1:idx_base+1]
                    grid_distances1 = torch.abs(self.param_grid.squeeze()[idx_parameter:idx_base] - self.param_grid.squeeze()[idx_parameter+1:idx_base+1])

                else:
                    s1 = torch.zeros(1).to(self.device)
                    grid_distances1 = torch.zeros(1).to(self.device)

                assert(s1.shape == grid_distances1.shape)
                s1 = s1 * grid_distances1

                exponent_closest = (-s1.sum() * self.epsilon_t).detach()

                if len(self.base_params) > 1:
                    #If applicable, get the loss weights based on the second closest base parameter, i.e. if the sample is between two base parameters
                    is_between = (param_base < param_i < param_base_second) or (param_base > param_i > param_base_second)

                    if is_between and self.average_importance_weights:
                        #Get the loss weights based on the closest base parameter
                        if idx_base_second < idx_parameter:
                            s2 = loss[idx_base_second:idx_parameter]
                            grid_distances2 = torch.abs(self.param_grid.squeeze()[idx_base_second+1:idx_parameter+1] - self.param_grid.squeeze()[idx_base_second:idx_parameter])

                            d_base_idx = idx_parameter - (idx_base_second + 1)

                        elif idx_base_second > idx_parameter:
                            s2 = loss[idx_parameter+1:idx_base_second+1]
                            grid_distances2 = torch.abs(self.param_grid.squeeze()[idx_parameter:idx_base_second] - self.param_grid.squeeze()[idx_parameter+1:idx_base_second+1])

                            d_base_idx = idx_base_second - 1 - idx_parameter

                        else:
                            raise ValueError("Not suported case for causality weights")
                        
                        assert(s2.shape == grid_distances2.shape)
                        s2 = s2 * grid_distances2

                        exponent_second = (-s2.sum() * self.epsilon_t).detach()

                        #Get the relative weigting based on the distance to the base temperature
                        d_base_base = abs(idx_base_second - idx_base) - 2
                        assert(d_base_base >= 0)
                        assert(d_base_idx >= 0)
                        assert(d_base_base >= d_base_idx)

                        k = d_base_idx / d_base_base

                        #Catch edge cases where the weights ignore one of the two contributions
                        if k == 1.0:
                            exponent = exponent_closest
                        
                        elif k == 0.0:
                            exponent = exponent_second
                        
                        else:
                            exponent = torch.logsumexp(torch.tensor([exponent_closest + np.log(k),exponent_second + np.log(1-k)]),0)

                        assert(exponent.shape == torch.Size([]))

                    else:
                        exponent = exponent_closest

                else:
                    exponent = exponent_closest
                
                causality_weights[i] = exponent

            causality_weights = causality_weights.detach()

            ###
            assert(causality_weights.shape == self.multiplyer.squeeze().shape)
            causality_weights = causality_weights * self.multiplyer.squeeze().to(causality_weights.device)

            return causality_weights

##############################################################################################
# Grid-less TRADE
##############################################################################################

class Objective_PINF_local_Ground_Truth_one_param_V2():
    def __init__(self,
                 t_burn_in:int,
                 t_full:int,
                 param_min:float,
                 param_max:float,
                 S:Callable,
                 S_kwargs:Dict,
                 dSdparam:Callable,
                 dSdparam_kwargs:Dict,
                 n_samples_expectation_approx:int,
                 n_samples_evaluation_per_param:int,
                 n_evaluation_params:int,
                 param_sampler_mode:str,
                 include_base_params:bool,
                 sample_param_params:Dict,
                 base_params:torch.tensor,
                 device:str,
                 residual_processing_parameters:Dict,
                 n_bins_storage:int = 100,
                 **kwargs
                 )->None:

        """
        parameters:
            t_burn_in:                      The time step after which the burn in phase is finished
            t_full:                         The time step after which the full range is reached
            param_min:                      The minimal parameter value
            param_max:                      The maximal parameter value
            S:                              The ground truth function in the exponent of the Gibbs distribution
            S_kwargs:                       Additional parameters for the ground truth energy function
            dSdparam:                       The derivative of the ground truth function with respect to the parameter
            dSdparam_kwargs:                Additional parameters for the derivative of the ground truth energy function
            n_samples_expectation_approx:   Number of samples used to approximate the expectation value for each condition value
            n_samples_evaluation_per_param: Number of evaluataion points per condition value.
            n_evaluation_params:            Total number of evaluated condition values in each training step.
            include_base_params:            If True, the loss is computed at the base parameters all the time, in addition to the randomly sampled parameters
            sample_param_params:            Parameters for the sampling of the parameter at which the loss is computed
            base_params:                    The base parameters at which the nll loss is computed
            device:                         The device on which the computation is performed
            residual_processing_parameters: Parametersto compute the loss from the residuals
            param_sampler_mode:             Set the method used to sample condition values
            n_bins_storage:                 Nimber of bins for storing the approximated expectation values (for visualization only)
        """

        print("*********************************************************************************************")
        print("Use class 'Objective_PINF_local_Ground_Truth_one_param_V2'")
        print("*********************************************************************************************")
        
        #Store settings
        self.device = device
        self.residual_processing_parameters = residual_processing_parameters
        self.include_base_params = include_base_params
        self.t_full = t_full
        self.t_burn_in = t_burn_in

        self.n_samples_expectation_approx = n_samples_expectation_approx
        self.n_samples_evaluation_per_param = n_samples_evaluation_per_param
        self.n_evaluation_params = n_evaluation_params
        self.param_sampler_mode = param_sampler_mode

        self.sample_param_params = sample_param_params
        self.sample_param_params["t_full"] = self.t_full
        self.sample_param_params["t_burn_in"] = self.t_burn_in

        self.S = S
        self.dSdparam = dSdparam
        self.S_kwargs = S_kwargs
        self.dSdparam_kwargs = dSdparam_kwargs

        #Set the limits of the beta ranges for the individual base temperatures
        global_min = torch.ones([len(base_params),1]) * param_min
        global_max = torch.ones([len(base_params),1]) * param_max

        selected_min = global_min
        selected_max = global_max

        self.params_mins = []
        self.params_maxs = []
        self.params_stars = []

        #Check the consistency of the settings
        for i in range(len(base_params)):
            self.params_mins.append(selected_min[i].item())
            self.params_maxs.append(selected_max[i].item())
            self.params_stars.append(base_params[i].item())

        #Store the approximated expectation vlues (only for later evaluation and visualization)

        #Initialize the bins of the grid
        self.param_bin_edges = torch.linspace(np.log(param_min),np.log(param_max),n_bins_storage+1).exp()
        self.param_storage_grid = torch.zeros(n_bins_storage)
        self.EX_storage_grid = torch.zeros(n_bins_storage)

        #Counter for the number of calls of the loss function
        self.iteration = 0

        print("Initialize loss model of type <Objective_PINF_local_Ground_Truth_one_param>")

    def get_loss(self,INN:INN_Model,param_batch:torch.Tensor,EX_batch:torch.tensor,get_eval_points:Callable)->torch.Tensor:

        """
        Perform the actual loss computation for a given inverse temperature.

        parameters:
            INN:                The INN model
            param:              The parameter at which the loss is computed
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor

        returns:
            loss:               The temperature scaling loss at the given inverse temperature
            EX_A:               The approxiamted expectation value of the energy of the derivative of the ground truth energy function with respect to the parameter at the given parameter
        """

        #Check inputs
        assert(len(param_batch.shape) == 1)
        assert(EX_batch.shape == torch.Size([self.n_evaluation_params]))

        #Get condition values
        param_batch = param_batch.reshape(-1,1)
        param_tensor = torch.ones([self.n_evaluation_params,self.n_samples_evaluation_per_param]) * param_batch
        assert (param_tensor.shape == torch.Size([self.n_evaluation_params,self.n_samples_evaluation_per_param]))
        param_tensor_flat = param_tensor.reshape(-1,1).to(self.device)

        #Get expectation values
        EX_batch = EX_batch.reshape(-1,1)
        EX_tensor = torch.ones([self.n_evaluation_params,self.n_samples_evaluation_per_param]) * EX_batch
        assert (EX_tensor.shape == torch.Size([self.n_evaluation_params,self.n_samples_evaluation_per_param]))
        EX_tensor_flat = EX_tensor.reshape(-1).to(self.device)

        #Get the target for the gradient
        with torch.no_grad():
            INN.train(False)

            x_eval = get_eval_points(beta_tensor = param_tensor_flat)

            #10) Compute the ground truth energies of the evaluation points
            A_eval = self.dSdparam(x_eval,**self.dSdparam_kwargs)

            assert(EX_tensor_flat.shape == A_eval.shape)

            #11) Compute the target
            target = EX_tensor_flat - A_eval
            
            INN.train(True)

        #Compute the gradient of the log-likelihood with respect to the condition
        param_tensor_flat.requires_grad_(True)

        log_p_x_eval = INN.log_prob(x_eval,param_tensor_flat)

        grad = torch.autograd.grad(log_p_x_eval.sum(),param_tensor_flat,create_graph=True)[0].squeeze()

        #Compute the residuals
        assert(grad.shape == target.shape)

        residuals = grad - target.detach()
        assert(residuals.shape == torch.Size([self.n_evaluation_params * self.n_samples_evaluation_per_param]))

        #Copute the loss from the residuals
        loss = get_loss_from_residuals(residuals,**self.residual_processing_parameters)

        return loss

    def __sample_param_batch(self)->torch.Tensor:
        
        #In case of burn in phase or alwys inclusion of the base parameters add them first
        if (self.iteration <= self.t_burn_in) or self.include_base_params:

            param_batch = torch.tensor(self.params_stars)
            idx = torch.randperm(len(self.params_stars))
            param_batch = param_batch[idx][:min(len(self.params_stars),self.n_evaluation_params)]

        else:
            param_batch = torch.zeros(0)

        #If the max number of points is already reached return the batch of parameter values
        if (self.iteration <= self.t_burn_in) or (len(param_batch) == self.n_evaluation_params):
            return param_batch
        
        n_params_to_sample = int(self.n_evaluation_params - len(param_batch))
        assert(n_params_to_sample > 0)

        for i in range(n_params_to_sample):
            
            #Use linear uniform sampling in an increasing interval
            if self.param_sampler_mode == "simple":
                #Get a base temperature at random
                idx = np.random.randint(low = 0,high = len(self.params_stars))
                param_star_i = self.params_stars[idx]

                param_i,left,right = get_beta(
                        t = self.iteration,
                        beta_star=param_star_i,
                        beta_min=self.params_mins[idx],
                        beta_max=self.params_maxs[idx],
                        **self.sample_param_params
                        )
                
                param_batch = torch.cat((torch.Tensor([param_i]),param_batch),0)
            else:
                raise NotImplementedError()

        #print("param batch:     ", param_batch)
        assert(param_batch.shape == torch.Size([self.n_evaluation_params]))

        return param_batch

    def get_expectation_values(self,param_batch:torch.Tensor,INN:INN_Model)->torch.Tensor:

        assert(len(param_batch.shape) == 1)

        #Get one large parameter tensor
        param_batch = param_batch.reshape(-1,1)
        param_tensor = torch.ones([self.n_evaluation_params,self.n_samples_expectation_approx]) * param_batch
        assert (param_tensor.shape == torch.Size([self.n_evaluation_params,self.n_samples_expectation_approx]))
        
        param_tensor_flat = param_tensor.reshape(-1,1).to(self.device)
        
        #Approximate the expectation value for the given parameter
        with torch.no_grad():
            INN.train(False)

            #1) Get samples from the INN
            x_proposal = INN.sample(n_samples = len(param_tensor_flat),beta_tensor = param_tensor_flat)

            #2) Compute the derivative of the ground truth energy function with respect to the parameter at the evaluation points
            A_proposal = self.dSdparam(x_proposal,**self.dSdparam_kwargs).reshape([self.n_evaluation_params,self.n_samples_expectation_approx])

            #3) Compute the log likelihood of the samples under the INN distribution and the ground truth distribution
            log_p_x_proposal_INN    = INN.log_prob(x_proposal,param_tensor_flat)
            log_p_x_proposal_GT     = - self.S(x_proposal,param_tensor_flat,**self.S_kwargs)

            assert(log_p_x_proposal_INN.shape == log_p_x_proposal_GT.shape)

            #4) compute the log likelihood ratios
            log_w = log_p_x_proposal_GT - log_p_x_proposal_INN

            #reshape 
            log_w = log_w.reshape([self.n_evaluation_params,self.n_samples_expectation_approx])

            #5) compute the log parition function
            log_Z = torch.logsumexp(log_w,dim = 1,keepdim=True) - np.log(self.n_samples_expectation_approx)

            assert(log_Z.shape == torch.Size([self.n_evaluation_params,1]))

            #6) Compute the importance weights
            log_omega = log_w - log_Z

            assert(log_omega.shape == A_proposal.shape)

            #7) Compute the sample based expectation value of the energy
            EX_A = (A_proposal * log_omega.exp()).mean(-1)
            
            INN.train(True)

        assert(EX_A.shape == torch.Size([self.n_evaluation_params]))

        #print("EX_A:    ",EX_A)

        return EX_A.detach().cpu()

    def __call__(self,INN,epoch,get_eval_points,logger = None)->torch.Tensor:
        """
        Compute the TS-PINF loss

        parameters:
            INN:                The INN model
            epoch:              The current epoch
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor
            logger:             The logger for the loss

        returns:
            loss:               The temperature scaling loss
            
        """

        #Get a batch of parameters at which the loss is evluated
        param_batch = self.__sample_param_batch()

        #Get the expectation values for the given batch of parameters
        EX_batch = self.get_expectation_values(param_batch=param_batch,INN = INN)

        loss = self.get_loss(INN = INN,param_batch=param_batch,EX_batch = EX_batch,get_eval_points=get_eval_points)

        logger.experiment.add_scalar(f"metadata/loss_model_internal_iteratoins",self.iteration,self.iteration)

        #For evaluation only
        #Store the approximated expectation values for the given parameter batch
        bin_idx = torch.searchsorted(self.param_bin_edges, param_batch, right=False) - 1

        #If some indices are there multiple times only store one
        unique, idx, counts = torch.unique(bin_idx, dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]

        unique_bin_indices = bin_idx[first_indicies]
    
        self.param_storage_grid[unique_bin_indices] = param_batch.squeeze()[first_indicies]
        self.EX_storage_grid[unique_bin_indices] = EX_batch.squeeze()[first_indicies]

        return loss
    
##############################################################################################
# Reverse KL 
##############################################################################################

class Objective_reverse_KL():
    def __init__(self,
                 beta_min:float,
                 beta_max:float,
                 log_p_target:Callable,
                 log_p_target_kwargs:Dict,
                 device:str,
                 bs:int,
                 )->None:

        """
        parameters:
            beta_min:                       The minimal inverse temperature
            beta_max:                       The maximal inverse temperature
            log_p_target                    Log likelihood of the (unnormalized) ground truth distribution
            log_p_target_kwargs:            Additional arguments for the ground truth log likelihood
            device:                         Name of the device on which the experiment runs
            bs:                             Batch size
        """
    
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.log_p_target = log_p_target
        self.log_p_target_kwargs = log_p_target_kwargs
        self.bs = bs
        self.device = device
        self.iteration = 1

        print("*********************************************************************************************")
        print("Use class 'Objective_reverse_KL'")
        print("*********************************************************************************************")

    def __call__(self,INN,epoch,get_eval_points,logger = None)->torch.Tensor:
        """
        Compute the TS-PINF loss

        parameters:
            INN:                The INN model
            epoch:              The current epoch
            get_eval_points:    Function to get evaluation points for the gradient evaluation. Takes arguments beta_tensor
            logger:             The logger for the loss

        returns:
            loss:               The temperature scaling loss
        """

        #Get evaluation points
        #Get inverse temperatures uniformly from the log space
        log_beta_tensor = (np.log(self.beta_max) - np.log(self.beta_min)) * torch.rand([self.bs,1]).to(self.device) + np.log(self.beta_min)
        beta_tensor = log_beta_tensor.exp()

        #Get INN samples
        x = INN.sample(self.bs,beta_tensor = beta_tensor)

        #Evaluate the samples on the ground truth log likelihood
        log_p_target_x = self.log_p_target(x = x,beta_tensor = beta_tensor,device = self.device,**self.log_p_target_kwargs)

        #Get the log likelihood under the INN model
        log_p_theta_x = INN.log_prob(x,beta_tensor)

        #filter out invalid values
        mask_A = torch.isfinite(log_p_target_x)
        mask_B = torch.isfinite(log_p_theta_x)

        mask = mask_A * mask_B

        #Get the ratios
        r = log_p_theta_x[mask] - log_p_target_x[mask]
        rev_KL = r.mean()

        #log the ratio of valid samples
        valid_r = mask.sum() / len(mask)

        logger.experiment.add_scalar(f"metadata/valid_r",valid_r,self.iteration)
        self.iteration += 1

        return rev_KL