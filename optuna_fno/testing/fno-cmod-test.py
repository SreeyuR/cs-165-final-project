from main import TrainingPipeline

pipeline = TrainingPipeline("/Users/u235567/Desktop/cs-165-final-project/config/fno/testing/config_fno_cmod_test.yaml")
model_path = "/Users/u235567/Desktop/cs-165-final-project/..checkpoints/checkpoints_cmon_fno/checkpoint_cmod-fno.pt"
pipeline.evaluate_loaded_model(path=model_path) 
