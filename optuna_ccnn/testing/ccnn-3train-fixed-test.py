from main import TrainingPipeline

model_path = ".checkpoints/checkpoints_trial_7_A-NEW_CCNN_3_REACTORS_AUC_MAX/checkpoint_epoch_300_trial_7_A-NEW_CCNN_3_REACTORS_AUC_MAX.pt"
pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/ccnn/testing/config_test_3train_fixed.yaml")
pipeline.evaluate_loaded_model(path=model_path)
