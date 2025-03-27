from main import TrainingPipeline

model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_2_A-CCNN_d3d_AUC/checkpoint_epoch_300_trial_2_A-CCNN_d3d_AUC.pt"
pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/ccnn/testing/config_test_d3d_ccnn.yaml")
pipeline.evaluate_loaded_model(path=model_path)
