from main import TrainingPipeline

model_path = "/groups/tensorlab/sratala/fno-disruption-pred/.checkpoints/checkpoints_trial_16_A-2_TRAIN_CCNN_AUC_FIXED/checkpoint_epoch_300_trial_16_A-2_TRAIN_CCNN_AUC_FIXED.pt"
pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/ccnn/testing/config_test_2train_fixed.yaml")
pipeline.evaluate_loaded_model(path=model_path)
