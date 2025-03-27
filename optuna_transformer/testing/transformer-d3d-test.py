from main import TrainingPipeline

model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_10_A-TRANSFORMER_d3d_AUC_MAX/checkpoint_epoch_300_trial_10_A-TRANSFORMER_d3d_AUC_MAX.pt"
pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/transformer/testing/config_d3d_test.yaml")
pipeline.evaluate_loaded_model(path=model_path)
