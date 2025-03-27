import optuna
import pandas as pd

def load_and_get_best_trials(study_name):
    # Load the existing study
    study = optuna.load_study(
        storage='sqlite:///optuna_results.sqlite3',  # Make sure this matches your storage
        study_name=study_name
    )

    # Print the best trials (Pareto front: optimize one objective without worsening a different objective)
    print("Best trials (Pareto front):")
    for trial in study.best_trials:
        print(f"  Trial {trial.number}:")
        print(f"    Values: {trial.values}")  # Objective values
        print(f"    Best Hyperparameters:")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")

    return study.best_trials


def best_trials_to_dataframe(best_trials):
    data = []
    for trial in best_trials:
        row = {"trial_number": trial.number, "values": trial.values}
        row.update(trial.params)  # Add hyperparameters
        data.append(row)

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    best_trials = load_and_get_best_trials('FNO_Valentin_Tuning')
    df_best_trials = best_trials_to_dataframe(best_trials)
    print(df_best_trials)

