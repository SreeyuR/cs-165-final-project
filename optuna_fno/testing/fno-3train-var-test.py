from main import TrainingPipeline

pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/fno/testing/config_test_3train_var.yaml")
model_path = "/groups/tensorlab/sratala/fno-disruption-pred/.checkpoints/checkpoints_trial_7_A-NEW_3Train-FNO-AUC-Maximization/checkpoint_epoch_300_trial_7_A-NEW_3Train-FNO-AUC-Maximization.pt"
pipeline.evaluate_loaded_model(path=model_path)
