from main import TrainingPipeline

model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_0_A-NEW_VAR-SAMPLING_CCNN_3_REACTORS_AUC_MAX/checkpoint_epoch_150_trial_0_A-NEW_VAR-SAMPLING_CCNN_3_REACTORS_AUC_MAX.pt"
pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/ccnn/testing/config_test_3train_var.yaml")
pipeline.evaluate_loaded_model(path=model_path)
