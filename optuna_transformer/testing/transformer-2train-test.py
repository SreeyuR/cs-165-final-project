from main import TrainingPipeline

model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_15_A-2TRAIN_TRANSFORMER_AUC_MAX/checkpoint_epoch_300_trial_15_A-2TRAIN_TRANSFORMER_AUC_MAX.pt"
pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/transformer/testing/config_2train_test.yaml")
pipeline.evaluate_loaded_model(path=model_path)
