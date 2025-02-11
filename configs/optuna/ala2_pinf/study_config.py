from math import ceil

def suggest_trial(base_config, trial):
    trial_config = base_config.copy()
    trial_config["trade_loss"]["pct_start"] = trial.suggest_float("pct_start", 0.2, 0.9)
    trial_config["trade_loss"]["end_value"] = trial.suggest_float("end_value", 0.01, 1.0, log=True)
    trial_config["trade_loss"]["alpha_adaptive_update"] = trial.suggest_float("alpha_adaptive_update", 0.0001, 0.1, log=True)

    trial_config["optimizer"]["lr"] = trial.suggest_float("lr", 2.e-4, 8.e-4, log=True)
    trial_config["optimizer"]["weight_decay"] = trial.suggest_float("weight_decay", 2.e-6, 2.e-4, log=True)


    trial_config["trade_loss"]["additional_kwargs"]["epsilon_causality_weight"] = trial.suggest_float("epsilon_causality_weight", 0.003, 0.3, log=True)
    trial_config["trade_loss"]["additional_kwargs"]["n_points_param_grid"] = trial.suggest_int("n_points_param_grid", 10, 50)
    trial_config["trade_loss"]["additional_kwargs"]["alpha_running_EX_A"] = trial.suggest_float("alpha_running_EX_A", 0.0, 0.9)

    trial_config["flow_hparams"]["n_transforms"] = trial.suggest_int("n_transforms", 15, 35)
    trial_config["flow_hparams"]["coupling_hparams"]["zero_init"] = trial.suggest_categorical("zero_init", [True, False])

    trial_config["batch_size"] = trial.suggest_int("batch_size", 128, 768)
    trial_config["max_epochs"] = ceil(40 * (trial_config["batch_size"]/256))
    trial_config["trade_loss"]["adaptive"] = trial.suggest_categorical("adaptive", [True, False])




    return trial_config
