import optuna
from optuna.trial import TrialState
import wandb
from optuna.integration import WeightsAndBiasesCallback
from model_params import load_config
from main import TrainingPipeline
from model import train_model
import utils
import evaluation

def create_objective(config_file_path, sample_hyperparameters_func, metrics=['auprc'], study_name='default', wandbc=None):
    """
    Creates an objective function for Optuna based on the provided configuration
    file and hyperparameter sampling function.

    Args:
        config_file_path (str): Path to the configuration file.
        sample_hyperparameters_func (callable): A function that accepts a trial object
            and returns a dictionary of hyperparameters to override.
        metrics (list): A list of metric names (strings) to be used as objective(s).
                        If the list contains more than one metric, the objective is considered multi-objective.

    Returns:
        callable: An objective function to be passed to study.optimize.
    """
    @wandbc.track_in_wandb()
    def objective(trial):
  
        # Sample hyperparameters using the provided sampling function.
        hyperparams = sample_hyperparameters_func(trial)
        
        # Create training pipeline and set the seed.
        pipeline = TrainingPipeline(config_file_path)
        
        # Update pipeline configuration with sampled hyperparameters.
        for section, params in hyperparams.items():
            for param, value in params.items():
                # pipeline.config.<section>.<param> = value
                setattr(getattr(pipeline.config, section), param, value)
        
        wandb.config.update(hyperparams, allow_val_change=True)        
        
        # run on the changed parameters that affect the dataset and model. 
        pipeline.train() # automatically sets the seed
        
        # Evaluate on the validation set (epoch=-1 means evaluation outside the training loop).
        epoch_num = pipeline.config.training.epochs - 1
        val_metrics = pipeline.trainer.evaluate(epoch_num) # last epoch 
        
        if pipeline.config.optuna.wandb_log:
            wandb.log(val_metrics)
        
        # Return the metrics based on the provided list. If more than one metric is specified, return a tuple of values.
        if len(metrics) > 1:
            return tuple(val_metrics[m] for m in metrics)
        else:
            return val_metrics[metrics[0]] # single objective
    
    return objective


def run_optuna_study(
    objective_func,
    study_name,
    directions,
    n_trials=15,
    n_jobs=4,
    storage='sqlite:///optuna_results.sqlite3',
    wandbc=None,
):
    """
    Creates and runs an Optuna study.

    Args:
        objective_func (callable): The objective function for the study.
        study_name (str): The study name.
        directions (list or str): Optimization directions (e.g., 'maximize' or a list for multi-objective).
        n_trials (int): Number of trials to run.
        n_jobs (int): Number of parallel jobs.
        storage (str): Storage URL for persisting study results.

    Returns:
        optuna.study.Study: The completed study.
    """
    if len(directions) > 1: # multi objective
        study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            directions=directions,
            load_if_exists=True
        )
    else: 
         study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            direction=directions[0],
            load_if_exists=True
        )
        
    study.optimize(objective_func, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True, callbacks=[wandbc])
    return study



if __name__ == "__main__":
    ##################################  PARAMS TO CHANGE ########################################################
    config_file_path = "/groups/tensorlab/sratala/fno-disruption-pred/config/config_optuna_undersample.yaml"
    study_name = 'FNO_AUPRC_Maximization'
    metrics = ['auprc']
    directions = ['maximize']     # For multi-objective, provide a list of directions (one for each returned metric).
    
    def sample_hyperparameters(trial):
        #                         cmod d3d east
        # interval 50: seql len = [10, 5, 2]
        # interval 100: seq len = [20, 10, 4]
        # interval 150: seq len = [30, 15, 6]
        # interval 200: seq len = [40, 20, 8]
        # interval 250: seq len = [50, 25, 10]
        # ....
        # interval 1000: seq len = [200, 100, 40]
        # 21: 210, 105, 42
        delta_t = trial.suggest_int('delta_t_factor', 5, 32) * 50 #trial.suggest_int('delta_t', 10, 70)
        # calculates based on sample rates for each machine (hard coaded values)
        machine_to_seq_lengths = utils.calculate_seq_length(delta_t)
        max_n_modes = min(machine_to_seq_lengths) // 2  # Dependent range for n_modes, TODO: undersand what n_modes really us
        return {
            'training': {
                'batch_size': trial.suggest_int('batch_size', 8, 64),
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True),
                # This is the interval (number of epochs or iterations) after 
                # which the learning rate will be updated (decayed). 
                # For example, if your step size is 25, the learning rate will 
                # change every 25 epochs.
                'step_size': trial.suggest_int('step_size', 10, 35),
                # multiplicative factor by which the learning rate is decayed at 
                # each update. For instance, if gamma is set to 0.5, the 
                # learning rate will be halved every time the step size interval 
                # is reached.
                'gamma': trial.suggest_float('gamma', 0.01, 1.0),
                'warmup_steps': trial.suggest_int('warmup_steps', 0, 50),
                # Adam
                'adam_weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
                # 'adam_beta1': trial.suggest_float('adam_beta1', 0.0, 1.0),
                # 'adam_beta2': trial.suggest_float('adam_beta2', 0.0, 1.0),
                # 'adam_eps': trial.suggest_float('adam_eps', 1e-8, 1e-1, log=True),
                'n_modes': trial.suggest_int('n_modes', 2, max_n_modes),
                'hidden_channels': trial.suggest_int('hidden_channels', 16, 80),
                'mlp_dropout': trial.suggest_float('mlp_dropout', 0.0, 0.5),
            },
            'data': {
                # cutoff of dataset, before truncation or anything
                'end_cutoff_timesteps': trial.suggest_int('end_cutoff_timesteps', 0, 8), 
                # timesteps from the end
                'delta_t': delta_t,
            },
            'balance_classes': {
                # number of times there are MORE non-disruptions than disruptions
                'undersample_ratio': trial.suggest_float('undersample_ratio', 0.5, 2.0), # lower the number, higher % of disruptions in dataset
                # in addition to ratio d:nd weight in loss function, add'l emphasis multiplied to pos class
                'pos_weight_emphasis_factor': trial.suggest_float('pos_weight_emphasis_factor', 1.0, 3.0),
                # ratio of augmented data to original data
                'data_augmentation_ratio': trial.suggest_float('data_augmentation_ratio', 0.0, 2.0), 
            }
        }
    
    ############################################################################################################
    
    assert len(metrics) == len(directions)
    
    wandb_kwargs = {
    'project': 'FNO_Disruption_Tuning',
    'group': 'fno-tuning',
    'name': study_name
    }
    
    if len(metrics) > 1:
        wandbc = WeightsAndBiasesCallback(metric_name=metrics, wandb_kwargs=wandb_kwargs, as_multirun=True)
    else:
        wandbc = WeightsAndBiasesCallback(metric_name=metrics[0], wandb_kwargs=wandb_kwargs, as_multirun=True)
        
    # Create the objective function using the sampling function.
    # Set multi_objective=False if you want to perform single-objective optimization.
    objective_func = create_objective(config_file_path, sample_hyperparameters, metrics=metrics, study_name=study_name, wandbc=wandbc)
    
    # Load additional configuration (e.g., number of trials and parallel jobs) from your config.
    config = load_config(config_folder="config", config_file=config_file_path, config_name="default", verbose=True)
    
    # Run the study. internally handles multi vs single objective study
    study = run_optuna_study(
        objective_func=objective_func,
        study_name=study_name,
        directions=directions,
        n_trials=config.optuna.num_trials,
        n_jobs=config.optuna.num_jobs,
        wandbc=wandbc
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics:")
    print("  Number of finished trials:", len(study.trials))
    print("  Number of pruned trials:", len(pruned_trials))
    print("  Number of complete trials:", len(complete_trials))
    
    if len(metrics) > 1: # multi objective study
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
