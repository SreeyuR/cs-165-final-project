from main import TrainingPipeline

pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/fno/testing/config_fno_d3d_test.yaml")
model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_trial_5_A-d3d-fno-auc"
pipeline.evaluate_loaded_model(path=model_path) 
