from main import TrainingPipeline

pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/fno/testing/config_test_3train_var.yaml")
model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_7_A-NEW_3Train-FNO-AUC-Maximization/checkpoint_epoch_300_trial_7_A-NEW_3Train-FNO-AUC-Maximization.pt"
pipeline.evaluate_loaded_model(path=model_path)
