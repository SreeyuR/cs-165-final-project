from main import TrainingPipeline

model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_5_A-CCNN_east_AUC/checkpoint_epoch_300_trial_5_A-CCNN_east_AUC.pt"
pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/ccnn/testing/config_test_east_ccnn.yaml")
pipeline.evaluate_loaded_model(path=model_path)
