from main import TrainingPipeline

model_path = "/groups/tensorlab/sratala/fno-disruption-pred/.checkpoints/checkpoints_trial_10_A-TRANSFORMER_d3d_AUC_MAX/checkpoint_epoch_300_trial_10_A-TRANSFORMER_d3d_AUC_MAX.pt"
pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/transformer/testing/config_d3d_test.yaml")
pipeline.evaluate_loaded_model(path=model_path)
