from main import TrainingPipeline

pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/fno/testing/config_test_2train_fixed.yaml")
model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_2_A-2TRAIN_FIXED_FNO_AUC/checkpoint_epoch_300_trial_2_A-2TRAIN_FIXED_FNO_AUC.pt" #"/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_7_A-2TRAIN_FIXED_FNO_AUC/checkpoint_epoch_300_trial_7_A-2TRAIN_FIXED_FNO_AUC.pt"
pipeline.evaluate_loaded_model(path=model_path) 