from main import TrainingPipeline

model_path = "/groups/tensorlab/sratala/fno-disruption-pred/.checkpoints/checkpoints_trial_3_A-2_TRAIN_CCNN_AUC_VAR'_#'TEST-CCNN_3_REACTORS_AUC_MAX/checkpoint_epoch_300_trial_3_A-2_TRAIN_CCNN_AUC_VAR'_#'TEST-CCNN_3_REACTORS_AUC_MAX.pt"
pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/ccnn/testing/config_test_2train_var.yaml")
pipeline.evaluate_loaded_model(path=model_path)
