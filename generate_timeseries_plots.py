import sys
sys.path.append("/groups/tensorlab/sratala/neuraloperator")
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import warnings
import printing
from model_params import load_config
warnings.filterwarnings('error', message="RuntimeWarning")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import data
import json
from main import TrainingPipeline
import utils

def get_seq_to_label_dataset():
    from seq_to_label import SeqToLabelDataset  
    return SeqToLabelDataset
 

# Function to get 10 indices for disruptive and non-disruptive shots

def get_disruption_indices(dataset, dataset_name, num_samples=10, save_dir="/groups/tensorlab/sratala/fno-disruption-pred/datasets/indices", force_new_indices=False):
    """
    Ensures consistent selection of disruptive and non-disruptive indices across datasets.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    index_file = f"{save_dir}/{dataset_name}_indices.json"

    # Try loading saved indices if they exist
    if os.path.exists(index_file) and not force_new_indices:
        with open(index_file, "r") as f:
            indices = json.load(f)
        return indices["disruptive"], indices["non_disruptive"]

    # Generate new indices if not found
    disruptive_indices = [i for i in range(len(dataset)) if dataset[i]['label'] == 1][:num_samples]
    non_disruptive_indices = [i for i in range(len(dataset)) if dataset[i]['label'] == 0][:num_samples]
    
    with open(index_file, "w") as f:
        json.dump({"disruptive": disruptive_indices, "non_disruptive": non_disruptive_indices}, f)

    return disruptive_indices, non_disruptive_indices

SAMPLING_RATES = {"cmod": 5, "d3d": 10, "east": 25}  # in microseconds (multiply abobe by 10^3)

def plot_shot_time_series(dataset, idx, dataset_name, save_dir="/groups/tensorlab/sratala/fno-disruption-pred/figs/time_series/FIXED_INTERVAL"):
    """
    Plots the time series of all features for a given shot index using machine-specific time spacing.

    Args:
        dataset (list of dict): The processed dataset where each entry corresponds to a shot.
        idx (int): The index of the shot to visualize.
        dataset_name (str): The name of the dataset (e.g., "Train", "Test", "Val").

    Returns:
        None (saves the plot)
    """
    
    plot_dir = f"{save_dir}/{dataset_name.lower()}"
    os.makedirs(plot_dir, exist_ok=True)  # Ensure directory exists

    feature_names = ['lower_gap', 'ip_error_normalized', 'd3d', 'beta_p', 
                     'v_loop', 'kappa', 'east', 'n_equal_1_normalized', 'li', 
                     'cmod', 'Greenwald_fraction', 'q95']

    if idx > len(dataset) - 1:
        idx = len(dataset) - 1

    label = "Disruptive (1)" if dataset[idx]['label'] == 1 else "Non-disruptive (0)"
    input_embedding = dataset[idx]["inputs_embeds"].cpu().numpy()  # Shape: (seq_len, num_features)
    
    seq_len, num_features = input_embedding.shape  # seq_len time steps, num_features features

    # Get the sampling rate for the given shot's machine
    machine = dataset[idx]["machine"]
    
    interval_length = SAMPLING_RATES[machine] * seq_len # in microseconds

    # Generate the time axis based on the machine's sampling rate
    time_steps = np.arange(0, interval_length, SAMPLING_RATES[machine])  # (start, stop)

    plt.figure(figsize=(12, 6))

    # Plot each feature over time
    for feature_idx in range(num_features):
        plt.plot(time_steps, input_embedding[:, feature_idx], label=f"{feature_names[feature_idx]}", alpha=0.8)

    # Formatting the plot
    plt.xlabel("Time (µs)")
    plt.ylabel("Feature Value")
    plt.title(f"{dataset_name} Dataset Shot {idx}: {label} ({machine} machine)")
    #plt.legend(loc="best", bbox_to_anchor=(1.05, 1))
    plt.grid(True)

    # Save and show plot
    plt.savefig(f"{plot_dir}/{dataset_name}_shot_{idx}_timeseries_{dataset[idx]['label']}.png", bbox_inches="tight")
    plt.show()

def plot_side_by_side_scaled_vs_non_scaled(scaled_dataset, non_scaled_dataset, idx, dataset_name, save_dir="/groups/tensorlab/sratala/fno-disruption-pred/figs/side_by_side_scaled"):
    """
    Plots the time series of a given shot from both scaled and non-scaled datasets side by side.
    """
    plot_dir = f"{save_dir}/{dataset_name.lower()}"
    os.makedirs(plot_dir, exist_ok=True)

    feature_names = ['lower_gap', 'ip_error_normalized', 'd3d', 'beta_p', 
                     'v_loop', 'kappa', 'east', 'n_equal_1_normalized', 'li', 
                     'cmod', 'Greenwald_fraction', 'q95']

    label = "Disruptive (1)" if non_scaled_dataset[idx]['label'] == 1 else "Non-disruptive (0)"
    machine = non_scaled_dataset[idx]["machine"]

    # Get input embeddings
    scaled_embedding = scaled_dataset[idx]["inputs_embeds"].cpu().numpy()
    non_scaled_embedding = non_scaled_dataset[idx]["inputs_embeds"].cpu().numpy()

    seq_len, num_features = scaled_embedding.shape
    time_steps = np.arange(0, seq_len)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for feature_idx in range(num_features):
        axes[0].plot(time_steps, non_scaled_embedding[:, feature_idx], label=feature_names[feature_idx], alpha=0.8)
        axes[1].plot(time_steps, scaled_embedding[:, feature_idx], label=feature_names[feature_idx], alpha=0.8)

    # Set labels and titles
    axes[0].set_title(f"{dataset_name} Shot {idx}: {label} (Non-Scaled)")
    axes[1].set_title(f"{dataset_name} Shot {idx}: {label} (Scaled)")
    axes[0].set_ylabel("Feature Value")
    axes[0].set_xlabel("Time (µs)")
    axes[1].set_xlabel("Time (µs)")
    
    # Add a single legend below the entire plot
    fig.legend(feature_names, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=6)
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to fit legend

    # plt.tight_layout()
    plt.savefig(f"{plot_dir}/{dataset_name}_shot_{idx}_side_by_side.png", bbox_inches="tight")
    plt.close()

def plot_side_by_side_disruptions(dataset, dataset_name, save_dir="/groups/tensorlab/sratala/fno-disruption-pred/figs/time_series/comparison", extra_title="", force_new_indices=False, num_samples=5):
    """
    Plots time series for 5 disruptive and 5 non-disruptive shots in a single large figure.

    Args:
        dataset (list of dict): The processed dataset where each entry corresponds to a shot.
        dataset_name (str): The name of the dataset (e.g., "Train", "Test", "Val").

    Returns:
        None (saves the plot)
    """
    plot_dir = f"{save_dir}/{dataset_name.lower()}"
    os.makedirs(plot_dir, exist_ok=True)  # Ensure directory exists

    disruptive_indices, non_disruptive_indices = get_disruption_indices(dataset, dataset_name, force_new_indices=force_new_indices, num_samples=num_samples)

    feature_names = ['lower_gap', 'ip_error_normalized', 'd3d', 'beta_p', 
                     'v_loop', 'kappa', 'east', 'n_equal_1_normalized', 'li', 
                     'cmod', 'Greenwald_fraction', 'q95']

    fig, axes = plt.subplots(10, 1, figsize=(12, 20), sharex=True)

    for i, idx in enumerate(disruptive_indices + non_disruptive_indices):
        shot_data = dataset[idx]["inputs_embeds"].cpu().numpy()
        seq_len, num_features = shot_data.shape
        time_steps = np.arange(0, seq_len)

        label = "Disruptive" if dataset[idx]['label'] == 1 else "Non-Disruptive"
        machine = dataset[idx]["machine"]

        for feature_idx in range(num_features):
            axes[i].plot(time_steps, shot_data[:, feature_idx], label=feature_names[feature_idx], alpha=0.8)

        axes[i].set_ylabel("Feature Value")
        axes[i].set_title(f"Shot {idx} ({label}, {machine})")

    axes[-1].set_xlabel("Time (µs)")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=6)
    plt.suptitle(f"{dataset_name} Dataset - Disruptions vs. Non-Disruptions"+extra_title)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{dataset_name}_side_by_side_comparison.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    config_file_path = "/groups/tensorlab/sratala/fno-disruption-pred/config/config_plot.yaml"
    tp = TrainingPipeline(config_file_path)
    train_dataset, test_dataset, val_dataset = tp.train_dataset, tp.test_dataset, tp.val_dataset #, tp.eval_dataset , eval_dataset

    print('AFTER FULLY PROCESSING DATASETS...')
    print('Undersampled train dataset')
    print('Kept test and val datasets exactly the same.')
    printing.print_dataset_info(tp.train_dataset, tp.test_dataset, tp.val_dataset) # , tp.eval_dataset
    datasets = [train_dataset, test_dataset, val_dataset]
    
    for dataset, name in zip(datasets, ["Train_NEW", "Test", "Val"]):
        printing.plot_sequence_histogram_by_interval(dataset, name, case_num=8, min_interval=50, max_interval=1000, step=50)
    
