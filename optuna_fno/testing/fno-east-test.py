from main import TrainingPipeline

pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/fno/testing/config_fno_east_test.yaml")
model_path = "/groups/tensorlab/sratala/fno-disruption-pred/.checkpoints/checkpoints_trial_1_A-east-fno-auc/checkpoint_epoch_300_trial_1_A-east-fno-auc.pt"
pipeline.evaluate_loaded_model(path=model_path) 
