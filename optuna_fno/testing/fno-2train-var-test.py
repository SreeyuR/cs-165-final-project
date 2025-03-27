from main import TrainingPipeline

pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/fno/testing/config_test_2train_var.yaml")
model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_14_A-2TRAIN_VAR_FNO_AUC/checkpoint_epoch_300_trial_14_A-2TRAIN_VAR_FNO_AUC.pt"
pipeline.evaluate_loaded_model(path=model_path) 
