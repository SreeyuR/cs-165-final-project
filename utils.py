import numpy as np
import pandas as pd
import pickle
import os
import random
import torch
import wandb
import torch.nn.functional as F


def wandb_setup(config, config_file_path):
    # trial is for optuna
    os.environ['WANDB_DISABLE_CACHE'] = 'true'
    wandb.login()
    wandb.require("service")
    wandb.init(
        reinit=True,
        project=config.wandb.project, #"disruption-prediction",
        group=config.wandb.group,
        name=f"{config.wandb.name}", #_trial_{trial.number}" if trial else config.wandb.name,
        config=dict(config)
    )
    # Save files listed in config
    files_to_save = getattr(config.wandb, "files_to_save", [])
    if config.wandb.files_to_save:
        artifact = wandb.Artifact('run_files', type='files')
        for file_path in files_to_save:
            artifact.add_file(file_path)
        # also save the config file
        artifact.add_file(config_file_path)
        wandb.log_artifact(artifact)


def equals(lst1, lst2):
    assert(len(lst1) == len(lst2))
    res = True
    for i in range(len(lst1)):
        if lst1[i] != lst2[i]:
            res = False
    return res


def save_data(data, data_filename='data/processed_data/processed_data.pkl'):
    """
    Save the data dictionary to a file.

    Args:
        data (dict): The data dictionary to save.
        data_filename (str): The filename where the data should be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(data_filename), exist_ok=True)

    # Iterate over each key in the dictionary and save the DataFrame to a file
    for key, value in data.items():
        df_filename = f"{data_filename}_{key}.pkl"  # Save each DataFrame with a unique key
        value['data'].to_pickle(df_filename)  # Save DataFrame to a pickle file
        value['data'] = df_filename  # Replace the DataFrame with its filename

    # Save the modified data dictionary (with DataFrame filenames) to a pickle file
    with open(data_filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {data_filename}")


def load_data(data_filename='data/processed_data/processed_data.pkl'):
    """
    Load the data dictionary from a file.

    Args:
        data_filename (str): The filename where the data is saved.

    Returns:
        dict: The loaded data dictionary.
    """
    if not os.path.exists(data_filename):
        print(f"Data pickle file {data_filename} not found... creating new raw data pickle.")
        return None

    # Load the data dictionary
    with open(data_filename, 'rb') as f:
        data = pickle.load(f)

    # For each key, load the DataFrame from its respective pickle file
    for key, value in data.items():
        df_filename = value['data']
        value['data'] = pd.read_pickle(df_filename)  # Load the DataFrame from the file

    print(f"Data loaded from {data_filename}")
    return data


def set_seed_across_frameworks(seed=10):
    np.random.seed(seed)    
    random.seed(seed)
    pd.options.mode.chained_assignment = None  # default='warn'
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Set seed: {seed}")

def check_no_row_repeats(df):
    data_vals = df.values
    unique_rows = df.drop_duplicates()  # Drops duplicate rows
    ROWS = len(data_vals)
    return True if len(unique_rows) == ROWS else False

def string_to_boolean(value: str) -> bool:
    # Convert string to lowercase and check for 'true'
    return value.strip().lower() == 'true'

def set_nans_to_mean(shot_df, col_name, sliding_window_len):
    shot_df[col_name] = shot_df[col_name].rolling(sliding_window_len).mean()
    first_idx = shot_df[col_name].first_valid_index()  # first non-null entry
    first_valid_mean = shot_df[col_name].loc[first_idx]
    shot_df[col_name] = shot_df[col_name].fillna(first_valid_mean)

def test_fix_resample_data_issue_v2(data_obj):
    df1 = pd.DataFrame({
        'A': [1, 1, 3],
        'B': [4, 4, 6],
        'C': [7, 7, 9]
    })
    df2 = pd.DataFrame({
        'A': [10, 11, 11],
        'B': [13, 14, 14],
        'C': [16, 17, 17]
    })
    df3 = pd.DataFrame({
        'A': [19, 20, 24],
        'B': [22, 23, 23],
        'C': [25, 26, 26]
    })

    data = [{'data': df1}, {'data': df2}, {'data': df3}]
    for i in range(len(data)):
        data_vals = data[i]["data"].values
        unique_row_idxs = [0]
        # remove rows that consecutively repeat
        for j in range(1, len(data_vals)):
            prev_unique_row = data_vals[unique_row_idxs[-1]]
            curr_row = data_vals[j]
            if not equals(prev_unique_row, curr_row):
                unique_row_idxs.append(j)
        data[i]["data"] = data[i]["data"].iloc[unique_row_idxs]
        assert (check_no_row_repeats(data[i]["data"]))

    df1_exp = pd.DataFrame({
        'A': [1, 3],
        'B': [4, 6],
        'C': [7, 9]
    })
    df2_exp = pd.DataFrame({
        'A': [10, 11],
        'B': [13, 14],
        'C': [16, 17]
    })
    df3_exp = pd.DataFrame({
        'A': [19, 20, 24],
        'B': [22, 23, 23],
        'C': [25, 26, 26]
    })
    # Expected DataFrames
    expected_dfs = [df1_exp, df2_exp, df3_exp]

    def equals_local(arr1, arr2):
        assert (len(arr1) == len(arr2))
        for i in range(len(arr1)):
            for j in range(len(arr1[0])):
                if arr1[i][j] != arr2[i][j]:
                    return False
        return True

    # Assert that the data in the list of DataFrames is equal to the expected DataFrames
    for i, df_dict in enumerate(data):
        if not equals_local(df_dict['data'].values, expected_dfs[i].values):
            print(i)
            print(df_dict['data'], expected_dfs[i])
            assert False, f"Fix data resampling may not be correct."

    print("Fix resampling working as expected.")


def check_row_repeats(df):
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
    data_vals = df.values
    print('LENGTH')
    print(len(data_vals))
    unrepeated = True
    for i in range(0, 5, len(data_vals)):
        for j in range(i+1, i+5):
            if not equals(data_vals[i], data_vals[j]):
                print(f'row {i} and row {j} not repeated!')
                print(f'\t{data_vals[i]}\n\t{data_vals[j]}')
                unrepeated = False
    if unrepeated:
        print('All rows are repeated 5 times!!')
        return True
    else:
        print('Some of the rows are NOT repeated 5 times...')
        return False

def get_pos_weight(train_dataset, emphasis_factor=1.5):
    """Compute pos_weight for BCEWithLogitsLoss.
    
    Args:
        train_dataset (Dataset): The training dataset.
        
    Returns:
        torch.Tensor: A tensor containing the pos_weight, which is the ratio of negative to positive samples.
    """
    n = len(train_dataset)  # Total samples
    d = train_dataset.num_disruptions  # Positive class samples (disruptions)

    if d == 0 or d == n:  
        raise ValueError("Dataset must contain both positive and negative samples.")

    # majority class / minority class
    pos_weight = torch.tensor([(n - d) / d * emphasis_factor], dtype=torch.float32)
    #pos_weight = torch.tensor([(n - d) / d], dtype=torch.float32)  # Compute pos_weight
    return pos_weight


# Evaluation helper functions
def separate_d_and_nd_shots(dataset):
    """Separate disruptive and non-disruptive shots from the dataset provided.

    Args:
        dataset (Dataset): dataset.

    Returns:
        d_test_shots (Dataset): Disruptive shots.
        nd_test_shots (Dataset): Non-disruptive shots.
        nd_shot_inds (list): Indices of non-disruptive shots.
        d_shot_inds (list): Indices of disruptive shots.
        max_len (int): Maximum sequence length in the dataset.
    """
    d_shot_inds = []
    nd_shot_inds = []
    max_len = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample["labels"].to("cpu").numpy()
        disruption_label = labels[1] # 0: nd, 1: d

        if disruption_label > 0.5:
            d_shot_inds.append(i)
        else:
            nd_shot_inds.append(i)

        seq_len = sample["inputs_embeds"].shape[1]
        max_len = max(max_len, seq_len)

    return d_shot_inds, nd_shot_inds, max_len


def get_probs_from_seq_to_lab_model(shot, eval_model, softmax_before_mean=False):
    """Get predicted probabilities from a seq_to_lab model.
    
    Args:
        shot (ModelReadyDataset): Shot object.
        eval_model (object): Model to evaluate.
        
    Returns:
        probs (np.array): Predicted probabilities.
    """        
    import model 
    probs = []
    
    # shot["inputs_embeds"]: (seq_len, 12) -> (70, 12)
    device = next(eval_model.parameters()).device
    inputs_embeds = shot["inputs_embeds"].unsqueeze(0) # add batch dim 1: (batch_size, seq_len, num_features)
    inputs_embeds = inputs_embeds[:, :, :].to(torch.float16) 
    inputs_embeds = model.process_model_inputs(inputs_embeds)
    inputs_embeds = inputs_embeds.to(device)
    logits = model.process_model_outputs(eval_model(inputs_embeds.float())) # (batch_size, 2)
    if not softmax_before_mean:
        disruption_prob = model.get_d_pred_probabilities(logits, softmax_before_mean)
    else:
        prob = logits.cpu().detach().numpy()
    probs.append(prob[0])

    return np.array(probs)


def moving_average_with_buffer(probs, buffer_value=0.1):
    """Compute a moving average with a starting buffer."""
    buffer_size = 5  # Adjust based on config
    buffer = [buffer_value] * buffer_size
    extended_probs = buffer + list(probs)

    averaged_probs = []
    for i in range(buffer_size, len(extended_probs)):
        avg_value = sum(extended_probs[i - buffer_size:i + 1]) / buffer_size
        averaged_probs.append(avg_value)

    return np.array(averaged_probs)

def prediction_time_from_end_positive(probs, threshold):
    """Compute the time from the end of the sequence to the first positive prediction.

    Args:
        probs (np.array): Predicted probabilities.
        threshold (float): Threshold for drawing labels.

    Returns:
        index (int): Index of the first positive prediction.
    """

    exceeds_threshold = (probs > threshold)

    # Find the first index where the condition is True
    index = np.argmax(exceeds_threshold).item()

    # If no probability is above the threshold, return the length of the sequence
    if index == 0 and not np.sum(exceeds_threshold[0]):
        index = probs.shape[0]

    return probs.shape[0] - index


def check_column_order(dataframes):
    """Check if all dataframes have the same column order.
    
    Args:
        dataframes (list): List of pandas DataFrames to check.
        
    Returns:
        bool: True if all dataframes have the same column order, False otherwise.
    """

    # Get the columns of the first dataframe in the list
    first_df_columns = dataframes[0].columns

    # Compare the columns of the first dataframe with the columns of each other dataframe
    for df in dataframes[1:]:
        if not df.columns.equals(first_df_columns):
            return False

    # If we've made it here, all dataframes have the same column order
    return True


def extract_final_sequence_chunks(inputs_embeds, labels, labels_probs, machines, shot_inds, delta_t):
    """
    Get the last chunk of each sequence, delta_t from the end.
    
    Args:
    - inputs_embeds (list of tensors): Each tensor has shape (seq_len, 12), chunk them all to be of shape (delta_t, 12)
    - labels (list): Each index corresponds to the label of the original sequence.
    - delta_t (int): Target sequence length for mini-tensors.

    Returns:
    - last_chunks (list of tensors): Chunked sequences of shape (delta_t, 12).
    - new_labels (list): Labels for chunked sequences, preserving the original label only for the last chunk.
    """
    last_chunks = []
    new_labels = []
    new_labels_probs = []
    new_machines = []
    new_shot_inds = []
    
    print(f"Chunking sequences to ensure uniform length of {delta_t}...")
    print(f"Skipping sequences of length < {delta_t}.")

    for i, (seq, label) in enumerate(zip(inputs_embeds, labels)):
        
        seq_len, _ = seq.shape # (seq_len, num_features=12)

        if seq_len < delta_t:
            continue  # Skip sequences that are too short

        segment = seq[-delta_t:, :]
        assert(segment.shape[0] == delta_t)
    
        last_chunks.append(segment)
        # machine, label, & shot idx stays the same for each chunk in the sequence as 
        # the original sequence
        new_labels.append(label)
        new_labels_probs.append(labels_probs[i])
        new_machines.append(machines[i])
        new_shot_inds.append(shot_inds[i])

    for seq in last_chunks:
        assert(isinstance(seq, torch.Tensor))
        if seq.shape[0] != delta_t:
            raise ValueError(f"❌ Sequence has incorrect shape: {seq.shape}")
    
    assert(len(new_labels_probs) == len(new_labels) == len(new_machines) == len(new_shot_inds) == len(last_chunks))

    return last_chunks, new_labels, new_labels_probs, new_machines, new_shot_inds


def is_valid_delta_t(delta_t_ms, sampling_rates):
    for rate in sampling_rates.values():
        if (delta_t_ms / rate) % 1 != 0:
            return False
    return True

def calculate_seq_length(interval):
    SAMPLING_RATES = {"cmod": 5, "d3d": 10, "east": 25}  # in microseconds (multiply abobe by 10^3)
    machine_to_seq_lengths = {}
    for machine, rate in SAMPLING_RATES.items():
        seq_len = int(interval / rate)
        machine_to_seq_lengths[machine] = seq_len
    return machine_to_seq_lengths
