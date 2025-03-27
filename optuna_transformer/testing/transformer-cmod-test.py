from main import TrainingPipeline

model_path = "/groups/tensorlab/sratala/fno-disruption-pred/.checkpoints/checkpoints_trial_3_A-TRANSFORMER_CMOD_AUC_MAX/checkpoint_epoch_300_trial_3_A-TRANSFORMER_CMOD_AUC_MAX.pt" #"/groups/tensorlab/sratala/fno-disruption-pred/.checkpoints/checkpoints_trial_13_A-TRANSFORMER_CMOD_AUC_MAX/checkpoint_epoch_300_trial_13_A-TRANSFORMER_CMOD_AUC_MAX.pt" #"/groups/tensorlab/sratala/fno-disruption-pred/.checkpoints/checkpoints_trial_9_A-TRANSFORMER_CMOD_AUC_MAX/checkpoint_epoch_300_trial_9_A-TRANSFORMER_CMOD_AUC_MAX.pt"
pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/transformer/testing/config_cmod_test.yaml")
pipeline.evaluate_loaded_model(path=model_path)
