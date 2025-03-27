from main import TrainingPipeline

pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/fno/testing/config_fno_cmod_test.yaml")
model_path = "/groups/tensorlab/sratala/fno-disruption-pred/.checkpoints/checkpoints_trial_19_A-cmod-fno-auc/checkpoint_epoch_300_trial_19_A-cmod-fno-auc.pt"
pipeline.evaluate_loaded_model(path=model_path) 
