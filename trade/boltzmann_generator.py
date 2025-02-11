from torch import nn
import torch
from lightning_trainable import Trainable
from .config import BoltzmannGeneratorHParams
from .data import get_loader, split_data
from .plots import create_plots, plot_minor_mode, plot_latent_space, plot_consistency_check
from .flow import Flow
from .bgflow_wrapper.bgflow_flow import BGFlowFlow
from .util import get_loss_lambda_from_hparams, parameter_schedule, weight_by_temperature, convert_dict_to_hparams, get_clipped_loss_from_hparams
from .losses import get_correction_function, compute_KL, ModelDistribution, TRADE_loss_legacy, TRADE_loss_continuous, TRADE_loss_grid
from .data_augmentor import DataAugmentor
from warnings import warn
from functools import partial
import numpy as np
from collections.abc import Iterable
import optuna
import traceback


class BoltzmannGenerator(Trainable):
    hparams: BoltzmannGeneratorHParams

    def __init__(self, hparams: BoltzmannGeneratorHParams | dict, optuna_trial=None, criterion=None):
        self.parameter_name = hparams.get("target_parameter_name", "temperature")
        self.datasets = [get_loader(hparams["dataset"]["name"])(
            **{self.parameter_name:parameter},
            **hparams["dataset"].get("kwargs", {})) for parameter in hparams["dataset"]["training_data_parameters"]]
        
        hparams = convert_dict_to_hparams(hparams, 
                                          dim=self.datasets[0].dim, 
                                          dims_c=None if not hasattr(self.datasets[0], "conditions") else [self.datasets[0].conditions.shape[1:]])

        train_data, val_data, test_data = split_data(self.datasets, 
                                                     split=hparams.dataset["split"], 
                                                     parameter_name=self.parameter_name,
                                                     parameter_reference_value=hparams.parameter_reference_value)
        
        hparams.epoch_len = len(train_data)//hparams.batch_size
        hparams.n_steps = hparams.max_epochs * hparams.epoch_len
        super().__init__(hparams, train_data=train_data, val_data=val_data, test_data=test_data)
        if self.hparams.softflow and self.hparams.flow_type != "bgflow":
            hparams.flow_hparams.dims_c.append((1,))

        if self.hparams.flow_type == "freia":
            self.flow = Flow(hparams.flow_hparams,
                             self.datasets[0].get_energy_model(),
                             self.datasets[0].system,
                             train_data[:][0].cpu().numpy(),
                             parameter_name=self.parameter_name)
        elif self.hparams.flow_type == "bgflow":
            self.flow = BGFlowFlow(hparams.flow_hparams, 
                                   self.datasets[0].get_energy_model(),
                                   self.datasets[0].system,
                                   parameter_name=self.parameter_name)

        if self.hparams.trade_loss is not None:
            additional_kwargs = self.hparams.trade_loss.additional_kwargs
            trade_mode = additional_kwargs.pop("mode", "grid")
            if "target_parameter_name" in additional_kwargs:
                warn(f"Overwriting target_parameter_name in additional_kwargs from {additional_kwargs['target_parameter_name']} to {self.parameter_name}")

            additional_kwargs["target_parameter_name"] = self.parameter_name
            if trade_mode == "grid":
                parameters = hparams.parameter_prior_hparams.parameters
                if not isinstance(parameters, Iterable):
                    parameters = parameters, parameters
                self.trade_loss = TRADE_loss_grid(param_min = min(parameters),
                                                param_max = max(parameters),
                                                base_params = torch.ones(1),
                                                **additional_kwargs,
                                                condition_sampler = self.sample_condition)
            elif trade_mode == "continuous":
                self.trade_loss = TRADE_loss_continuous(**additional_kwargs, condition_sampler = self.sample_condition)
            elif trade_mode == "legacy":
                self.trade_loss = TRADE_loss_legacy(**additional_kwargs)


        self.data_augmentor = DataAugmentor(hparams.data_augmentation_hparams, self.datasets[0].dim)
        self.parameter_prior = parameter_schedule(hparams.n_steps, hparams.parameter_prior_hparams)
        self.min_nll = 0.0

        self.optuna_trial = optuna_trial
        self.criterion = criterion
        assert [self.hparams.nll_loss, self.hparams.kl_loss, self.hparams.temperature_weighted_loss, \
                self.hparams.trade_loss, self.hparams.var_loss].count(None) <= 4, "At least one loss must be specified"

    def forward(self, *args, **kwargs):
        return self.flow(*args, **kwargs)

    def dequantize(self, x, c):
        if not self.training:
            if self.hparams.softflow and isinstance(self.hparams.dequantization_noise, (list, tuple)):
                dequantization_noise = min(self.hparams.dequantization_noise)
            else:
                dequantization_noise = 0.0
        else:
            dequantization_noise = self.hparams.dequantization_noise

        if isinstance(dequantization_noise, float):
            noise_strength = dequantization_noise * torch.ones(x.shape[0], device=x.device)
            if self.hparams.softflow and self.training:
                warn("Softflow is enabled, but dequantization noise is not set to a range.")
        elif isinstance(dequantization_noise, (list, tuple)):
            assert len(dequantization_noise) == 2, "Dequantization noise must be a list or tuple of length 2"
            min_noise, max_noise = min(dequantization_noise), max(dequantization_noise)
            min_noise, max_noise = np.log(min_noise), np.log(max_noise)
            noise_strength = min_noise + (max_noise - min_noise) * torch.rand(x.shape[0], device=x.device)
            noise_strength = torch.exp(noise_strength)
        else:
            raise ValueError(f"Unknown dequantization noise format {type(dequantization_noise)}")

        noise_strength = noise_strength.unsqueeze(1)

        if self.hparams.softflow:
            c = (*c, torch.log(noise_strength))
        return x + noise_strength * torch.randn_like(x), c

    def data_augmentation(self, x, c):
        if self.training:
            x, c = self.data_augmentor(x, c)
        return x, c

    def compute_metrics(self, batch, batch_idx):
        metrics = {}

        x, data_parameter, c = batch[0].to(self.device), batch[1].to(self.device), list(batch[2:])
        c = [c_i.to(self.device) for c_i in c]
        x, c = self.dequantize(x, c)
        x, c = self.data_augmentation(x, c)

        current_step = batch_idx + self.hparams.epoch_len*self.current_epoch
        if self.hparams.update_min_nll_every is not None and current_step % self.hparams.update_min_nll_every == 0:
            self.update_min_nll()

        target_parameter = self.parameter_prior(current_step, x.shape[0]).unsqueeze(1).to(x.device)


        self.update_adaptive_lambdas(x, c, data_parameter, target_parameter, current_step)

        lam_nll = get_loss_lambda_from_hparams(self.hparams.nll_loss, self.hparams.n_steps, current_step)
        lam_kl = get_loss_lambda_from_hparams(self.hparams.kl_loss, self.hparams.n_steps, current_step)
        lam_temp = get_loss_lambda_from_hparams(self.hparams.temperature_weighted_loss, self.hparams.n_steps, current_step)
        lam_trade = get_loss_lambda_from_hparams(self.hparams.trade_loss, self.hparams.n_steps, current_step)
        lam_var = get_loss_lambda_from_hparams(self.hparams.var_loss, self.hparams.n_steps, current_step)

        metrics["lam_nll"] = lam_nll
        metrics["lam_kl"] = lam_kl
        metrics["lam_temp"] = lam_temp
        metrics["lam_trade"] = lam_trade
        metrics["lam_var"] = lam_var

        if (lam_var > 0 or lam_trade > 0) and not self.hparams.parameter_prior_hparams.sample_parameter_per_batch:
            warn("Parameter is not sampled per batch, but per sample. This can lead to unexpected behavior in the variance and trade loss, which rely on batch statistics.")


        metrics["loss"] = 0.0

        if lam_nll > 0 or not self.training:
            # Parameter for NLL is always data_parameter
            nll = self.get_nll_loss(x, c, target_parameter=data_parameter)
            metrics["nll"] = nll.mean()
            metrics["loss"] += lam_nll * get_clipped_loss_from_hparams(self.hparams.nll_loss, nll).mean()

        if lam_kl > 0 or (not self.training and self.datasets[0].has_energy):
            kl = self.get_kl_loss(x.shape[0], self.sample_condition(x.shape[0]), target_parameter)
            metrics["kl"] = kl.mean()
            metrics["loss"] += lam_kl * get_clipped_loss_from_hparams(self.hparams.kl_loss, kl).mean()

        if lam_var > 0 or (not self.training and self.datasets[0].has_energy):
            # check if kl already has been computed
            try:
                proposal_distribution = self.hparams.var_loss.additional_kwargs.get("proposal", "model")
            except AttributeError:
                proposal_distribution = "model"

            if proposal_distribution == "model" and "kl" in metrics:
                var_loss = kl.var()
            else:
                var_loss = self.get_variance_loss(x.shape[0], self.sample_condition(x.shape[0]), target_parameter, proposal_distribution)

            metrics["var_loss"] = var_loss
            metrics["loss"] += lam_var * get_clipped_loss_from_hparams(self.hparams.var_loss, var_loss).mean()

        if lam_temp > 0 or (not self.training and self.parameter_name == "temperature"):
            if "nll" in metrics:
                nll_base = nll
            else:
                nll_base = None
            temp_weighted_loss = self.get_temperature_weighted_loss(x, c, data_parameter, target_parameter, nll_base=nll_base)
            metrics["temperature_weighted_loss"] = (temp_weighted_loss).mean()
            metrics["loss"] += lam_temp * get_clipped_loss_from_hparams(self.hparams.temperature_weighted_loss, temp_weighted_loss).mean()

        if (lam_trade > 0 or not self.training) and hasattr(self, "trade_loss"):
            trade_loss = self.get_trade_loss(x, c, data_parameter, target_parameter)
            metrics["trade_loss"] = trade_loss.mean()
            metrics["loss"] += lam_trade * get_clipped_loss_from_hparams(self.hparams.trade_loss, trade_loss).mean()

        if not isinstance(metrics, dict):
            raise ValueError(f"Metrics must be a dictionary, but is {metrics}")
        return metrics

    def get_nll_loss(self, x, c, target_parameter):
        return self.flow.energy(x, c, parameter=target_parameter)

    def get_kl_loss(self, batch_size_or_x, c, target_parameter):
        if isinstance(batch_size_or_x, int):
            batch_size = batch_size_or_x
        else:
            batch_size = batch_size_or_x.shape[0]
        return self.flow.kldiv(batch_size, c, parameter=target_parameter)

    def get_temperature_weighted_loss(self, x, c, data_parameter, target_parameter, nll_base=None):
        assert self.parameter_name == "temperature", "Temperature weighted loss only works with temperature as parameter"
        try:
            self_supervised = self.hparams.temperature_weighted_loss.additional_kwargs.get("self_supervised", True)
            reference_temperature = self.hparams.temperature_weighted_loss.additional_kwargs.get("relative_temperature", "training")
        except AttributeError:
            self_supervised = True
            reference_temperature = "training"
        if self_supervised:
            if reference_temperature == "training":
                if nll_base is None:
                    nll_base = self.get_nll_loss(x, c, target_parameter=data_parameter)
                relative_temperature = target_parameter / data_parameter 
            elif reference_temperature == "uniform":
                new_parameter = torch.rand_like(target_parameter) * torch.abs(target_parameter - 1.0) + torch.where(target_parameter > 1.0, 1.0, target_parameter)
                x = self.flow.sample(x.shape[0], c, parameter=new_parameter)
                nll_base = self.get_nll_loss(x, c, target_parameter=new_parameter)
                relative_temperature = target_parameter / new_parameter
            elif reference_temperature == "centered":
                new_parameter = (target_parameter + 1.0)/2
                x = self.flow.sample(x.shape[0], c, parameter=new_parameter)
                nll_base = self.get_nll_loss(x, c, target_parameter=new_parameter)
                relative_temperature = target_parameter / new_parameter
            elif reference_temperature == "fixed":
                new_parameter = target_parameter + torch.where(target_parameter > 1.0, torch.clip(1.0 - target_parameter, -0.1, 0.0), torch.clip(1.0 - target_parameter, 0.0, 0.1))
                x = self.flow.sample(x.shape[0], c, temperature=new_parameter)
                nll_base = self.get_nll_loss(x, c, target_parameter=new_parameter)
                relative_temperature = target_parameter / new_parameter
            self.min_nll = 0.99 * self.min_nll + 0.01 * torch.min(nll_base)
        else:
            nll_base = self.flow.get_energy_model().energy(x)
        w_T = weight_by_temperature(nll_base, relative_temperature, self.min_nll)
        nll_at_T = self.get_nll_loss(x, c, target_parameter=target_parameter)
        return w_T.detach() * nll_at_T

    def get_trade_loss(self, x, c, data_parameter, target_parameter):
        if isinstance(self.trade_loss, TRADE_loss_legacy):
            # Hack to make the legacy trade loss work with multiple parameters in the training data
            reference_parameter = self.hparams.trade_loss.additional_kwargs.get("reference_parameter")
            if not self.hparams.trade_loss.additional_kwargs.get("self_supervised", True):
                reference_parameter = data_parameter
            return self.trade_loss(flow=self.flow,
                                  target_parameter=target_parameter,
                                  reference_parameter=reference_parameter,
                                  samples=x,
                                  conditions=c,
                                  condition_sampler=partial(self.sample_condition, train=self.training),
                                  batch_size=x.shape[0])
        else:
            return self.trade_loss(flow=self.flow,
                                base_parameter=self.hparams.trade_loss.additional_kwargs.get("reference_parameter"),
                                target_parameter_proposals=target_parameter,
                                sample_parameter=data_parameter,
                                samples=x,
                                conditions=c,
                                batch_size=x.shape[0])

    def get_variance_loss(self, batch_size, c, target_parameter, proposal="model"):
        if proposal == "model":
            return self.flow.kldiv(batch_size, c, parameter=target_parameter).var()
        elif proposal == "noised_model":
            if self.flow.hparams.scale_latent_with_parameter:
                z = self.flow.prior.sample([batch_size], c, target_parameter)
                z += torch.randn_like(z) * (0.1*target_parameter)
            else:
                z = self.flow.prior.sample([batch_size], c)
                z += torch.randn_like(z) * 0.1
            return self.flow.energy_ratio_from_latent(z, c, parameter=target_parameter).var()

    def get_grad_magnitude(self, loss_i):
        #Compute the magnitude of the gradient for the nll loss
        opt = self.optimizers()
        opt.zero_grad()

        loss_i.backward()

        grad_mag = 0
        num_params = sum([param.numel() for _, param in self.flow.named_parameters() if param.requires_grad])
        for name, param in self.flow.named_parameters():
            if param.requires_grad and (param.grad is not None):
                grad_mag += param.grad.pow(2).sum().detach().item()/ num_params**2
        grad_mag = np.sqrt(grad_mag)

        return grad_mag


    def update_adaptive_lambdas(self, x, c, data_parameter, target_parameter, current_step):

        configs = [self.hparams.nll_loss, self.hparams.kl_loss, self.hparams.temperature_weighted_loss, self.hparams.trade_loss, self.hparams.var_loss]

        if current_step % self.hparams.update_adaptive_weights_every != 0 or not any([config.adaptive and config.pct_start <= current_step/self.hparams.n_steps for config in configs if config is not None]) or not self.training:
            return

        loss_inputs = [
            (x, c, data_parameter),
            (x.shape[0], self.sample_condition(x.shape[0]), target_parameter),
            (x, c, data_parameter, target_parameter),
            (x, c, data_parameter, target_parameter),
            (x.shape[0], self.sample_condition(x.shape[0]), target_parameter)
        ]
        losses = [self.get_nll_loss, self.get_kl_loss, self.get_temperature_weighted_loss, self.get_trade_loss, self.get_variance_loss]

        grad_mags = np.zeros(len(configs))
        adaptive_configs = []

        for i, (config, loss, loss_input) in enumerate(zip(configs, losses, loss_inputs)):
            if config is not None and config.pct_start <= current_step/self.hparams.n_steps:
                loss_i = loss(*loss_input).mean()
                if config.adaptive:
                    grad_mags[i] = self.get_grad_magnitude(loss_i=loss_i)
                    adaptive_configs.append((i, config))
                else:
                    warn(f"Some loss weights are adaptive, while others are not. This can lead to problems in normalization.")
                    lam_loss_i = get_loss_lambda_from_hparams(config, self.hparams.n_steps, current_step)
                    grad_mags[i] = self.get_grad_magnitude(loss_i=lam_loss_i*loss_i)

        self.optimizers().zero_grad()

        grad_mags = grad_mags[np.array([ind for ind, _ in adaptive_configs])]
        grad_mags = np.clip(np.array(grad_mags), a_min=1e-5, a_max=1e9)
        grad_mags = np.sum(grad_mags) / grad_mags
        # Normalize the gradient magnitudes
        grad_mags = np.clip(grad_mags / np.sum(grad_mags), a_min=1e-4, a_max=1e3)

        for i, (_, config) in enumerate(adaptive_configs):
            config.adaptive_weight = config.alpha_adaptive_update * grad_mags[i] + (1 - config.alpha_adaptive_update) * config.adaptive_weight

    def update_min_nll(self):
        self.min_nll = 1e9
        with torch.no_grad():
            for i, batch in enumerate(self.train_dataloader()):
                x, data_parameter, c = batch[0].to(self.device), batch[1].to(self.device), list(batch[2:])
                c = [c_i.to(self.device) for c_i in c]
                x, c = self.dequantize(x, c)
                min_nll = torch.min(self.flow.energy(x, c, data_parameter).detach())
                self.min_nll = min(min_nll, self.min_nll)
                if i > 10:
                    break

    def sanity_check(self):
        # TODO: This fails because the coordinate system is not centered
        batch = next(iter(self.train_dataloader()))[0].to(self.device)
        with torch.no_grad():
            assert torch.all((batch - self.flow(self.flow(batch)[0], rev=True)[0])**2 < 1e-5), "Sanity check failed"

    def on_train_epoch_start(self):
        self.update_min_nll()
        # if self.current_epoch == 0:
        #     self.plot()

    def on_train_epoch_end(self):
        if self.current_epoch % self.hparams.plotting["interval"] == 0 or self.current_epoch == (self.hparams.max_epochs - 1):
            try:
                self.plot()
            except Exception as e:
                # No need to stop a run because of a plotting error
                print(f"Plotting failed in epoch {self.current_epoch} with error {traceback.format_exc()}")
        
        if self.optuna_trial is not None and self.criterion is not None:
            self.optuna_trial.report(self.criterion(self), self.current_epoch)
            if self.optuna_trial.should_prune():
                raise optuna.TrialPruned()

    def sample_condition(self, batch_size, train=True):
        if train:
            ind = torch.randint(len(self.train_data), (batch_size,))
            c = self.train_data[ind][2:]
            _, c = self.dequantize(torch.zeros(batch_size, self.datasets[0].dim, device=self.device), c)
        else:
            ind =  torch.randint(len(self.val_data), (batch_size,))
            c = self.val_data[ind][2:]
            _, c = self.dequantize(torch.zeros(batch_size, self.datasets[0].dim, device=self.device), c)
        return [c_i.to(self.device) for c_i in c]


    @torch.no_grad()
    def plot(self):
        parameter_samples = []
        for parameter in self.hparams.plotting["parameters"]:
            model_parameter = parameter/self.hparams.parameter_reference_value
            samples = []
            for _ in range(self.hparams.plotting["n_samples"]//10000):
                c = self.sample_condition(10000, train=False)
                samples.append(self.flow.sample(10000, c, parameter=model_parameter))
            samples = torch.cat(samples, dim=0)
            samples = samples[~torch.isnan(samples).any(dim=1)]
            parameter_samples.append((parameter, samples))
            create_plots(samples, 
                         self.flow,
                         parameter, 
                         self.parameter_name, 
                         self.hparams.parameter_reference_value,
                         self.logger.experiment,
                         self.hparams.dataset, 
                         self.current_epoch)
        if self.hparams.dataset.name == "ala2":
            plot_minor_mode(parameter_samples, self.logger.experiment, self.hparams.dataset.name, self.current_epoch)

        # if self.hparams.flow_type != "bgflow":
        latent = []
        for batch in self.val_dataloader():
            x, data_parameter, c = batch[0].to(self.device), batch[1].to(self.device), list(batch[2:])
            c = [c_i.to(self.device) for c_i in c]
            x, c = self.dequantize(x, c)
            latent.append(self.flow(x, c, parameter=data_parameter)[0].detach().cpu())
        latent = torch.cat(latent, dim=0)
        plot_latent_space(latent, self.logger.experiment, self.current_epoch)

        if self.parameter_name == "temperature":
            T_0 = 1.0
            p_T_0 = ModelDistribution(self.flow, self.sample_condition, parameter=T_0, n_repeats=self.hparams.plotting.get("n_repeats_log_prob", 1))
            q = ModelDistribution(self.flow, self.sample_condition, parameter=T_0, n_repeats=self.hparams.plotting.get("n_repeats_log_prob", 1))
            correction = get_correction_function(p_T_0, q, n_proposal_samples=self.hparams.plotting.get("n_proposal_samples", 100000))

            kl_divs = []

            for parameter in self.hparams.plotting["parameters"]:
                T_star = parameter/self.hparams.parameter_reference_value
                p_T_star = ModelDistribution(self.flow, self.sample_condition, parameter=T_star, n_repeats=self.hparams.plotting.get("n_repeats_log_prob", 1))
                kl_divs.append(compute_KL(p_T_star, p_T_0, correction, T_star, T_0, n_KL_samples=self.hparams.plotting.get("n_KL_samples", 10000)))
            kl_divs = torch.stack(kl_divs, dim=0).cpu().numpy()
            plot_consistency_check(kl_divs, self.hparams.plotting["parameters"], self.logger.experiment, self.current_epoch)

    def __del__(self):
        if self.hparams.flow_type == "bgflow":
            try:
                self.flow.bgflow_bg._target._bridge.context_wrapper.terminate()
            except:
                pass
