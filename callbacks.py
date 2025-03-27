import wandb
import torch
import os

class WandbCallback:
    """Logs all metrics to Weights & Biases at every step and evaluation."""
    def __init__(self, training_pipeline, global_step=0, prefix="", log_all_metrics=True):
        self.trainer = training_pipeline.trainer
        self.global_step = global_step
        self.prefix = prefix
        self.log_all_metrics = log_all_metrics
    
    def on_step_end(self):
        """Logs training metrics every 100 steps."""
        self.global_step += 1
        if self.global_step % 100 == 0:
            logs = {"global_step": self.global_step, "train_loss": self.trainer.train_losses[-1]}
            wandb.log(logs, step=self.global_step)
    
    def on_evaluation(self, epoch):
        """Logs evaluation metrics at each validation step."""
        eval_results = self.trainer.evaluate(epoch)
        logs = {f"{self.prefix}{k}": v for k, v in eval_results.items()}
        wandb.log(logs, step=self.global_step)
        print(f"[INFO] Evaluation logged at step {self.global_step}: {logs}")


class BestModelCallback:
    """Stores the best model (by auc) in memory and restores it at the end."""
    
    def __init__(self):
        self.best_metric = float('-inf') 
        self.best_model_state = None
        self.evaluating = False
        self.best_model_epoch = 0
    
    def on_evaluate(self, metrics, epoch, trainer):
        """Checks if model is the best one so far and stores its state."""
        current_metric = metrics.get("auc_score")

        if current_metric is not None and current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_model_epoch = epoch
            self.best_model_state = {k: v.clone().detach() for k, v in trainer.model.state_dict().items()}
            print(f"[INFO] New best model found at epoch {epoch} with f1: {self.best_metric:.4f}")
        
    def on_train_end(self, trainer):
        """Loads the best model's state back into the trainer's model."""
        checkpoint_dir = trainer.checkpoint_dir
        if self.best_model_state:
            try: 
                trainer.model.load_state_dict(self.best_model_state)
                print(f"[INFO] Restored best model state at the end of training, which occurred at epoch {self.best_model_epoch}.")

                model_name = trainer.config.model_type
                total_epochs = trainer.config.training.epochs
                save_path = os.path.join(checkpoint_dir, f"best_{model_name}_epoch{total_epochs}.pt")
                torch.save({'model_state_dict': self.best_model_state}, save_path)
                print(f"[INFO] Best model saved to {save_path}")
            except Exception as e:
                print(f"[WARNING] Failed to save best model: {e}")
