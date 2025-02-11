import optuna
from trade import BoltzmannGenerator
from yaml import safe_load
import torch
import argparse
from trade.data import get_loader
from os.path import join
import os
import importlib.util
from optuna.exceptions import TrialPruned

def import_function_from_file(module_path, function_name):
    module_name = os.path.splitext(os.path.basename(module_path))[0]  # Get the module name from the file name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

def get_objective(base_config, suggest_trial, criterion, study_name):
    # max_loss = -100

    def objective(trial):
        trial_config = suggest_trial(base_config, trial)
        
        model = BoltzmannGenerator(trial_config, trial, criterion)
        try:
            model.fit(
                logger_kwargs = {"save_dir": "lightning_logs", "name": f"optuna_study_{study_name}"},
            )
        except TrialPruned as e:
            raise e
        except Exception as e:
            print(f"Error during training: {e}.\nPruning Trial.")
            raise TrialPruned()

        loss = criterion(model)
        # if loss > max_loss:
        #    loss = max_loss
        return loss

    return objective

@torch.no_grad()
def alanine_dipeptide_criterion(model):
    loss = 0.0
    lam_relative = 0.9
    for batch in model.val_dataloader():
            x, c = batch[0].to(model.device), list(batch[1:])
            c = [c_i.to(model.device) for c_i in c]
            x, c = model.dequantize(x, c)
            loss += (1 - lam_relative) * model.get_nll_loss(x, c, parameter=1.0).mean().item()/len(model.val_dataloader())

    dataset = get_loader("ala2")(parameter=300)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(dataset.coordinates.reshape(-1, dataset.dim)).to(model.device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    for batch in loader:
        x, c = batch[0].to(model.device), list(batch[1:])
        c = [c_i.to(model.device) for c_i in c]
        x, c = model.dequantize(x, c)
        loss += lam_relative * model.get_nll_loss(x, c, parameter=300/model.hparams["parameter_reference_value"]).mean().item()/len(loader)
    return loss

@torch.no_grad()
def double_well_criterion(model):
    loss = 0.0
    for batch in model.val_dataloader():
            x, c = batch[0].to(model.device), list(batch[1:])
            c = [c_i.to(model.device) for c_i in c]
            x, c = model.dequantize(x, c)
            loss += model.get_nll_loss(x, c, parameter=1.0).mean().item()/len(model.val_dataloader())

    dataset = get_loader("double_well_2d")(parameter=0.5)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(dataset.coordinates.reshape(-1, dataset.dim)).to(model.device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    for batch in loader:
        x, c = batch[0].to(model.device), list(batch[1:])
        c = [c_i.to(model.device) for c_i in c]
        x, c = model.dequantize(x, c)
        loss += model.get_nll_loss(x, c, parameter=0.5/model.hparams["parameter_reference_value"]).mean().item()/len(loader)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser( )
    parser.add_argument(
        "--study-name",
        type=str,
        default="ala2_optimization",
        help="Name of the optuna study",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="alanine_dipeptide",
        help="Criterion to optimize",
    )
    parser.add_argument(
        "--study-config-path",
        type=str,
        help="Path to the study configuration file",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="TPESampler",
        help="Name of the sampler to use",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="NopPruner",
        help="Name of the pruner to use",
    )
    parser.add_argument(
        "--pruner-warmup-steps",
        type=int,
        default=10,
        help="Number of warmup epochs until the pruner starts",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Patience for the pruner"
    )

    args = parser.parse_args()
    base_config = safe_load(open(join(args.study_config_path, "base_params.yaml"), 'r'))
    if args.criterion == "alanine_dipeptide":
        criterion = alanine_dipeptide_criterion
    elif args.criterion == "double_well":
        criterion = double_well_criterion
    else:
        raise ValueError(f"Unknown criterion {args.criterion}")
    # import the sampler from the optuna.samplers module
    sampler = optuna.samplers.__dict__[args.sampler]()
    pruner = optuna.pruners.__dict__[args.pruner]
    pruner = optuna.pruners.PatientPruner(pruner(n_warmup_steps=args.pruner_warmup_steps), patience=args.patience) if args.pruner != "NopPruner" else pruner()
    study = optuna.create_study(storage=f"sqlite:///optuna_dbs/{args.study_name}.db", 
                                study_name=args.study_name, direction="minimize", 
                                sampler=sampler,
                                pruner=pruner,
                                load_if_exists=True)
    # import the suggest_trial function from the study_config_path
    suggest_trial = import_function_from_file(join(args.study_config_path, "study_config.py"), "suggest_trial")
    objective = get_objective(base_config, suggest_trial, criterion, args.study_name)
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)


