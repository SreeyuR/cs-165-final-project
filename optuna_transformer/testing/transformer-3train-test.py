from main import TrainingPipeline

model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_12_A-NEW_TRANSFORMER_3_REACTORS_AUC_MAX/checkpoint_epoch_300_trial_12_A-NEW_TRANSFORMER_3_REACTORS_AUC_MAX.pt"
pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/transformer/testing/config_3train_test.yaml")
pipeline.evaluate_loaded_model(path=model_path)
