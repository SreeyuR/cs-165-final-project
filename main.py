import logging
import warnings
from model_params import load_config
import utils
warnings.filterwarnings('error', message="RuntimeWarning")
import torch
import torch.nn as nn
from functools import partial
import wandb
import evaluation
from data import collate_fn_seq_to_label, save_load_datasets
from model import get_model, train_model, SimpleTrainer, run_permutation_importance #, CustomImbalancedLoss
import printing
import callbacks

class TrainingPipeline:

    def __init__(self, config_file_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare Parameters
        self.config_file_path = config_file_path
        self.config = load_config(config_folder="config", config_file=config_file_path, config_name="default", verbose=True)
        self.wandb_log = self.config.wandb.log
        self.threshold = self.config.training.reg_to_class.threshold
                
        print("-----------------------------------------------")
        print("Run started!")
        print(self.config.wandb.name)
        print("-----------------------------------------------")
        
        if self.wandb_log and not self.config.optuna.use_optuna:
            utils.wandb_setup(self.config, self.config_file_path)
        self.trainer = None # not defined yet
        
    def _setup_data(self):
        # Prepare Data
        # data is a dict with keys 0,1,2,...22555. Each data[i] is a dict with 4 keys:
        #   label (int; 0 or 1), machine (str; 'd3d'), shot (int; 156336), data (df; hundreds of rows resampled & 13 cols)
        # TODO Check that this doesn't hard code/save the delta_t, undersampling, augmentation, etc (hyperparams that SHOULD change) restricting them from changing
        self.train_dataset, self.test_dataset, self.val_dataset = save_load_datasets(self, force_reload=False) # self.eval_dataset
        
        printing.print_dataset_info(train_dataset=self.train_dataset, test_dataset=self.val_dataset, val_dataset=self.test_dataset)
    
    def _setup_model(self):
        # Prepare Model & Optimizer
        self.model, self.optimizer, self.scheduler = get_model(self.config)
        balanced_params = self.config.balance_classes
        pos_weight = utils.get_pos_weight(self.train_dataset, self.config.balance_classes.pos_weight_emphasis_factor)
        # combines a Sigmoid layer and the BCELoss in one single class; this version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, 
        # by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device)) if balanced_params.balanced_loss else nn.BCEWithLogitsLoss()
        self.best_model_callback = callbacks.BestModelCallback()
        self.data_collator = collate_fn_seq_to_label
        

    def train(self):
        self._setup_data()
        self._setup_model()
        utils.set_seed_across_frameworks(self.config.seed)
        
        if self.config.training.curriculum_steps == 0:
            self.trainer = train_model(self)
            print("Done with the training and evaluation!!")
        else:
            # Define the curriculum: list of end cutoff timesteps (e.g., higher cutoffs initially, then lower)
            ecs = [8, 6, 4, 2, 0]
            # Optionally, select a subset of steps (here we use all, then reverse order if needed)
            curriculum_steps = self.config.training.curriculum_steps #len(ecs)
            selected_ecs = ecs[:curriculum_steps][::-1]  # e.g., [0, 2, 4, 6, 8]
            
            # Initialize a global training step counter
            global_step = 0
            
            # Define the number of epochs per curriculum step (this could be a new hyperparameter)
            epochs_per_step = self.config.training.epochs
            
            for ec in selected_ecs:
                print("---------------------")
                print(f"Training with end cutoff {ec}")
                print("---------------------")
                
                # Update the configuration with the new end cutoff timesteps
                pipeline.config.data.end_cutoff_timesteps = ec
                
                # Re-load (or reprocess) the datasets with the new cutoff applied
                pipeline.train_dataset, pipeline.test_dataset, pipeline.val_dataset = save_load_datasets(pipeline, force_reload=False)
                
                printing.print_dataset_info(train_dataset=self.train_dataset, test_dataset=self.val_dataset, val_dataset=self.test_dataset)
                
                trainer = train_model(self)
                _, global_step = trainer.train(trainer, global_step, num_epochs=epochs_per_step)
              
            
            print("Curriculum learning complete. Final global step:", global_step)
    

    def evaluate_loaded_model(self, run_name=None, path=None):
        self._setup_data()
        self._setup_model()

        if self.config.training.addl_metric:
            temp = self.val_dataset
            self.val_dataset = self.test_dataset
            self.test_dataset = temp 
    
        print("~~~~~~~~~~~~~~~~~")
        print("Evaluating...")
        print("~~~~~~~~~~~~~~~~~")
        run_name = run_name if run_name else self.config.wandb.name
        run_name = run_name.replace(" ", "_").replace("/", "_")
        if path:
            save_path = path
        else:
            save_path = f'/groups/tensorlab/sratala/fno-disruption-pred/.checkpoints/checkpoints_{run_name}/checkpoint_epoch_500.pt'
        
        if not self.trainer:
            checkpoint = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Model loaded from {save_path}.")

        self.model.to(self.device)
        
        # Count total and trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("-----------------------------------------------")
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print("-----------------------------------------------")
                
        print("-----------------------------------------------")
        print("Evaluation Metrics:")
        print("-----------------------------------------------")
        # Print saved metrics from the checkpoint (computed during training evaluation in trainer.evaluate())
        print(f"Validation Metrics (epoch {checkpoint['epoch']}):")
        print(f"  AUC: {checkpoint['auc_score']:.4f}")
        print(f"  AUPRC: {checkpoint['auprc']:.4f}")
        print(f"  F1 Score: {checkpoint['f1']:.4f}")
        print(f"  Accuracy: {checkpoint['accuracy']:.4f}")
        print(f"  Precision: {checkpoint['precision']:.4f}")
        print(f"  Recall: {checkpoint['recall']:.4f}")
        print()
        

        # Initialize trainer with validation dataset only
        
        trainer = SimpleTrainer(
            model=self.model,
            run_name=self.config.wandb.name,
            wandb_log=self.wandb_log,
            config=self.config,
            optimizer=None,  # No need for optimizer during evaluation
            loss_fn=self.loss_fn,   # same loss function as during training
            train_dataset=None,  # No training dataset needed
            val_dataset=self.val_dataset,  # Use validation dataset
            compute_metrics=evaluation.compute_metrics,
            data_collator=self.data_collator,
            device=self.device,
            softmax_before_mean=self.config.training.reg_to_class.softmax_before_mean,
            threshold=self.threshold
        )
    
        eval_metrics = evaluation.compute_metrics_testing(self, trainer, threshold=self.threshold)
        
        # Evaluate model using evaluation dataset
        print('---------------------------')
        print("Test Dataset Metrics:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value}")
        print('---------------------------')
      
        return eval_metrics


if __name__ == "__main__":
    pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/config_best_opt_params.yaml")
    #pipeline.train()
    pipeline.evaluate_loaded_model(run_name="Easy DP - Optuna Best Params v2")

