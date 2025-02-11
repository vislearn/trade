import torch
import numpy as np
from warnings import warn
from torch.distributions import Distribution
from typing import Callable, Dict, Any
from functools import partial
from tqdm.auto import tqdm
from math import ceil

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
