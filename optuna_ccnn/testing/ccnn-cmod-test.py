from main import TrainingPipeline

model_path = TODO
pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/ccnn/testing/config_test_cmod_ccnn.yaml")
pipeline.evaluate_loaded_model(path=model_path)
