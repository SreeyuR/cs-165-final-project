from main import TrainingPipeline

pipeline = TrainingPipeline("/groups/tensorlab/sratala/fno-disruption-pred/config/config_best_opt_params.yaml")
pipeline.train()
pipeline.evaluate_loaded_model(run_name="PLS_WORK_trial_46_3_REACTORS_FNO_AUC_Maximization")
