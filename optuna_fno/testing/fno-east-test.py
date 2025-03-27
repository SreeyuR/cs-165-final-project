from main import TrainingPipeline

pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/fno/testing/config_fno_east_test.yaml")
model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_1_A-east-fno-auc/checkpoint_epoch_300_trial_1_A-east-fno-auc.pt"
pipeline.evaluate_loaded_model(path=model_path) 
