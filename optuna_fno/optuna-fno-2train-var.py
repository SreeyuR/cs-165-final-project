import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("PYTHONPATH:", sys.path)
import optuna
from optuna.trial import TrialState
from optuna.integration import WeightsAndBiasesCallback
import wandb
import argparse
from model_params import load_config
import torch
from model import get_model, train_model
import utils
import evaluation
from main import TrainingPipeline
from optuna_funcs import create_objective, run_optuna_study

# Increase the initialization timeout to 300 seconds (adjust as needed)
# os.environ['WANDB_TIMEOUT'] = '300'
# Set the start method to 'thread'
# os.environ['WANDB_START_METHOD'] = 'thread'
# os.environ['WANDB_DEBUG'] = 'true'
# os.environ['WANDB_CORE_DEBUG'] = 'true'

#config_file_path = "/Users/u235567/Desktop/cs-165-final-project/config/config_auc_optuna_all3.yaml"
config_file_path = "/Users/u235567/Desktop/cs-165-final-project/config/fno/config_fno_2train_var.yaml"
study_name = 'A-2TRAIN_VAR_FNO_AUC'

wandb_proj_name = "CORRECT-Fixed-FNO-Disruption-Tuning"
wandb_group_name = 'var-fno-tuning'
metrics = ['auc_score']
directions = ['maximize']     # For multi-objective, provide a list of directions (one for each returned metric), otherwise just have one thing in the list

def sample_hyperparameters(trial):
    # delta_t = trial.suggest_int('delta_t', 30, 60)
    # max_n_modes = delta_t // 2  # Dependent range for n_modes
    delta_t = trial.suggest_int('delta_t_factor', 1, 19) * 50 #trial.suggest_int('delta_t', 10, 70)
    # calculates based on sample rates for each machine (hard coaded values)
    machine_to_seq_lengths = utils.calculate_seq_length(delta_t)
    max_n_modes = min(machine_to_seq_lengths.values()) // 2  # Dependent range for n_modes, TODO: undersand what n_modes really us
    return {
        'training': {
            #'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]), #trial.suggest_int('batch_size', 17, 32),
            # 'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True),
            'learning_rate_scheduler_target': trial.suggest_float('learning_rate_scheduler_target', 1e-4, 2e-1, log=True), # higher initial learning rate does significantly better 
            'learning_rate_adam': trial.suggest_float('learning_rate_adam', 1e-3, 2e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-6, 1e-3, log=True),
            # This is the interval (number of epochs or iterations) after 
            # which the learning rate will be updated (decayed). 
            # For example, if your step size is 25, the learning rate will 
            # change every 25 epochs.
            'step_size': trial.suggest_int('step_size', 10, 40),
            # multiplicative factor by which the learning rate is decayed at 
            # each update. For instance, if gamma is set to 0.5, the 
            # learning rate will be halved every time the step size interval 
            # is reached. 
            'gamma': trial.suggest_float('gamma', 0.2, 0.9), # 0.1, 0.8
            'warmup_steps': trial.suggest_int('warmup_steps', 10, 40),
            # Adam
            'adam_weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True), #  40e-6, 1.5e-3
            # 'adam_beta1': trial.suggest_float('adam_beta1', 0.0, 1.0),
            # 'adam_beta2': trial.suggest_float('adam_beta2', 0.0, 1.0),
            # 'adam_eps': trial.suggest_float('adam_eps', 1e-8, 1e-1, log=True),
            'n_modes': trial.suggest_int('n_modes', 1, max_n_modes),
            'hidden_channels': trial.suggest_int('hidden_channels', 10, 50), # (30, 50)
            'mlp_dropout': trial.suggest_float('mlp_dropout', 0.0, 0.5), # (0.15, 0.4)
        },
        'data': {
            # cutoff of dataset, before truncation or anything
            'end_cutoff_timesteps': trial.suggest_int('end_cutoff_timesteps', 0, 8), 
            # timesteps from the end
            'delta_t': delta_t,
        },
        'balance_classes': {
            # number of times there are MORE non-disruptions than disruptions
            # TODO: try 4.6 for undersample ratio as that is the true amount
            'undersample_ratio': trial.suggest_float('undersample_ratio', 1.2, 4.6), # lower the number, higher % of disruptions in dataset
            # in addition to ratio d:nd weight in loss function, add'l emphasis multiplied to pos class
            'pos_weight_emphasis_factor': trial.suggest_float('pos_weight_emphasis_factor', 1.15, 4), # 1.6
            # ratio of augmented data to original data
            #'data_augmentation_ratio': trial.suggest_float('data_augmentation_ratio', 0.0, 2.0), 
        }
    }

############################################################################################################


assert len(metrics) == len(directions)

wandb_kwargs = {
    'project': wandb_proj_name,
    'group': wandb_group_name,
    'name': study_name
}

if len(metrics) > 1:
    wandbc = WeightsAndBiasesCallback(metric_name=metrics, wandb_kwargs=wandb_kwargs, as_multirun=True)
else:
    wandbc = WeightsAndBiasesCallback(metric_name=metrics[0], wandb_kwargs=wandb_kwargs, as_multirun=True)
    
# Create the objective function using the sampling function.
# Set multi_objective=False if you want to perform single-objective optimization.
objective_func = create_objective(config_file_path, sample_hyperparameters, metrics=metrics, study_name=study_name, wandbc=wandbc, )

# Load additional configuration (e.g., number of trials and parallel jobs) from your config.
config = load_config(config_folder="config", config_file=config_file_path, config_name="default", verbose=True)

# Run the study. internally handles multi vs single objective study
study = run_optuna_study(
    objective_func=objective_func,
    study_name=study_name,
    directions=directions,
    n_trials=config.optuna.num_trials,
    n_jobs=config.optuna.num_jobs,
    wandbc=wandbc,
    
)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics:")
print("  Number of finished trials:", len(study.trials))
print("  Number of pruned trials:", len(pruned_trials))
print("  Number of complete trials:", len(complete_trials))

if len(metrics) > 1 : # multi objective study
    print("Best trials (Pareto front):")
    for trial in study.best_trials:
        print(f"  Trial {trial.number}:")
        print(f"    Values: {trial.values}")
        print("    Hyperparameters:")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")
else: # single objective stidu
    print("Best trial:")
    trial = study.best_trial
    print("  AUPRC: ", trial.value)
    print("  Best Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
