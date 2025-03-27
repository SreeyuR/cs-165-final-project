from main import TrainingPipeline

model_path = "/groups/tensorlab/sratala/fno-disruption-pred/.checkpoints/checkpoints_trial_5_A-CCNN_east_AUC/checkpoint_epoch_300_trial_5_A-CCNN_east_AUC.pt"
pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/ccnn/testing/config_test_east_ccnn.yaml")
pipeline.evaluate_loaded_model(path=model_path)
