import os
import warnings
import numpy as np
import pandas as pd
import pickle
from model_params import load_config
from seq_to_seq import SeqToSeqDataset
from seq_to_label import SeqToLabelDataset
import utils
import plotting
from torch.nn.utils.rnn import pad_sequence
import torch
import printing

warnings.filterwarnings('error', message="RuntimeWarning")

os.environ['PLOTTING'] = 'False'


class Data:
    def __init__(self, config_file_name):
        self.config = load_config(config_folder="config", config_file=config_file_name, config_name="default")


    def load_raw_data(self):
        """
        Loads and processes dataset from `data: filename` in configuration settings with sampling rate corrections.
        Returns:
            data (dict): keys 0,1,2,...22555. Each data[i] is a dict with 4 keys:
                         label (int; 0 or 1), machine (str; 'd3d'), shot , data (df; hundreds of rows & 13 cols)
                         - 'label' (int): for shot, 0: non-disruption, 1: disruption.
                         - 'machine' (str): the machine or experimental setup (e.g., 'd3d', 'east', 'cmod').
                         - 'shot' (int): Shot number (like 156336)
                         - 'data' (pandas DataFrame): Has columns
                                                       - 'lower_gap'
                                                       - 'ip_error_normalized'
                                                       - 'd3d'
                                                       - 'beta_p'
                                                       - 'v_loop'
                                                       - 'kappa'
                                                       - 'east'
                                                       - 'n_equal_1_normalized'
                                                       - 'time'
                                                       - 'li'
                                                       - 'cmod'
                                                       - 'Greenwald_fraction'
                                                       - 'q95'
        """
        print("Loading data...")
        data_params = self.config.data
        data_filename = self._get_data_filename(data_params.folder, data_params.filename)
        print(f'- Data Filename: {data_filename}')
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)  # dict with keys 0-22555, each a dict 4 keys: label, machine, shot, data
        plotting.make_all_plots(data[0]['data'], interactive=False, addl_title=' data point 0: before fixed sampling',
                                directory='figs/fixed_resampling')
        if data_params.fix_sampling:
            # standardize sampling rate across different datasets using forward filling; missing vals filled with latest known value
            # pickle file has been forward filled.
            data = self._fix_resample_data_issue(data, data_params.standardize_sampling_rate)

        plotting.make_all_plots(data[0]['data'], interactive=False, addl_title='data point 0: after fixed sampling',
                                directory='figs/fixed_resampling')

        print(f'- Fix sampling to standardize sampling rate across datasets using forward fill): '
              f'{data_params.fix_sampling}')
        print(f'- Number of data entries: {len(data.keys())}')
        print(f'- Keys in each individual data entry: {data[0].keys()}')
        return data


    def _get_data_filename(self, folder, filename):
        """
        Retrieves the full path for the specified data file.
        """
        cwd = os.path.join(os.getcwd(), folder)
        if os.path.exists(os.path.join(cwd, filename)):
            f = os.path.join(cwd, filename)
        else:
            raise FileNotFoundError(f"Could not find the data file in: {os.path.join(cwd, filename)}")
        return f


    def _save_data(self, df, file_name):
        if not os.path.exists("data"):
            os.makedirs("data")
        file_path = os.path.join("data", file_name+".csv")
        print(f"Dataframe saved to {file_path}.")
        df.to_csv(file_path, index=False)


    def _fix_resample_data_issue(self, data, standardize_sampling_rate=False, resample_rate=.005):
        """
         Resample time series data to ensure consistent sampling rates across different machines.
         Standardizes the sampling rate based on the machine type and the provided `resample_rate`.
         Uses forward filling to handle missing values during resampling.
         
         cmod is by default 0.005 sampling rate, that's why default resample rate is 0.005 milli seconds
         or 5000 microseconds since it's the sampling rate of the machine with the highest sampling rate.

         Args:
             data (list of dict)
             resample_rate (float): target sampling rate in secs. Default is 0.005 seconds (200 Hz).
             standardize_sampling_rate: 
                True: all machines are forced to have the same sampling rate (0.005 milliseconds)
                False: each machine has different sampling rates based on what it natively was

         Returns:
             list of dict: The input data with each DataFrame resampled to the specified rate, if necessary.

         Notes:
             - For 'd3d' machines, initial resampling rate is set to 0.025 ms (40 Hz).
             - For 'east' machines, initial resampling rate is set to 0.1 ms (10 Hz).
             - For 'cmod' machines, initial resampling rate will stay to 0.005 ms (200 Hz).
             - data will be resampled to the `resample_rate` specified if standardize_samplingrate is True.
        """
        #self._save_data(data[0]["data"], "before_resampling_data_0")
        # print(f'Data frame (data point 0) before fixed resampling:')
        # print(f'    - Rows: {data[0]["data"].shape[0]}')
        # print(f'    - Columns: {data[0]["data"].shape[1]}')
        # print(f'    - Description:\n{data[0]["data"].describe()}')
        # print()
        print(f"\nEach data key is resampled using resample rate based on machine.\n\n")

        for i in range(len(data)):
            if any(missing > 0 for _, missing in data[i]["data"].isnull().sum().items()):
                raise ValueError(f"Data at index {i} contains missing values.")
            data[i]["data"].set_index("time", inplace=True)
            if data[i]["machine"] == "d3d":
                data[i]["data"] = data[i]["data"].resample("0.025s").ffill() # fill missing entries with last known values
                if standardize_sampling_rate: # False by default
                    data[i]["data"] = data[i]["data"].resample(f"{resample_rate}s").ffill()
            elif data[i]["machine"] == "east":
                data[i]["data"] = data[i]["data"].resample("0.1s").ffill()
                if standardize_sampling_rate:
                    data[i]["data"] = data[i]["data"].resample(f"{resample_rate}s").ffill()
            data[i]["data"].reset_index(inplace=True)

        # print(f'Data frame (data point 0) after fixed resampling:')
        # print(f'    - Rows: {data[0]["data"].shape[0]}')
        # print(f'    - Columns: {data[0]["data"].shape[1]}')
        # print(f'    - Description:\n{data[0]["data"].describe()}')
        # print()
        #self._save_data(data[0]["data"], "after_resampling_data_0")
        return data


    def _fix_resample_data_issue_v2(self, data):
        """
        Resample time series data to remove consecutive duplicates in rows.
        """
        self._save_data(data[0]["data"], "before_resampling_data_0")
        print(f'Data frame (data point 0) before fixed resampling:')
        print(f'    - Rows: {data[0]["data"].shape[0]}')
        print(f'    - Columns: {data[0]["data"].shape[1]}')
        print(f'    - Description:\n{data[0]["data"].describe()}')
        print()
        print(f"\nEach data key is resampled using resample rate based on machine.\n\n")

        for i in range(len(data)):
            if any(missing > 0 for _, missing in data[i]["data"].isnull().sum().items()):
                raise ValueError(f"Data at index {i} contains missing values.")
            data_vals = data[i]["data"].values
            unique_row_idxs = [0]
            # remove rows that consecutively repeat
            for j in range(1, len(data_vals)):
                prev_unique_row = data_vals[unique_row_idxs[-1]]
                curr_row = data_vals[j]
                if not utils.equals(prev_unique_row, curr_row):
                    unique_row_idxs.append(j)
            data[i]["data"] = data[i]["data"].iloc[unique_row_idxs]
            assert(utils.check_no_row_repeats(data[i]["data"]))

        print(f'Data frame (data point 0) after fixed resampling:')
        print(f'    - Rows: {data[0]["data"].shape[0]}')
        print(f'    - Columns: {data[0]["data"].shape[1]}')
        print(f'    - Description:\n{data[0]["data"].describe()}')
        print()

        self._save_data(data[0]["data"], "after_resampling_data_0")
        return data


    def prepare_datasets(self, raw_data, pretraining=False):
        """
        Prepares and returns train, test, and validation datasets.
        Also prepares evaluation dataset if it's not for pretraining.

        Args:
            raw_data (dict): A dictionary where each key (integer) corresponds to a unique shot.
                         Each shot is a dictionary containing:
                         - 'label': An integer (0 or 1) indicating the class or category of the record.
                         - 'machine': A string representing the machine or experimental setup (e.g., 'd3d', 'east', 'cmod').
                         - 'shot': Unique index of shot
                         - 'data': A pandas DataFrame with the following columns:
                           - 'lower_gap'
                           - 'ip_error_normalized'
                           - 'd3d'
                           - 'beta_p'
                           - 'v_loop'
                           - 'kappa'
                           - 'east'
                           - 'n_equal_1_normalized'
                           - 'time'
                           - 'li'
                           - 'cmod'
                           - 'Greenwald_fraction'
                           - 'q95'

        Returns:
            tuple: Train, test, validation, and evaluation datasets.
        """
        data_params = self.config.data
        
        if not utils.check_column_order([df["data"] for _, df in raw_data.items()]):
            raise ValueError("Dataframe columns are out of order.")

        # Chose indices based on Chang's paper
        # Use dataset indices columns
        train_inds = pd.read_csv(
            f"{os.getcwd()}/data/indices/train_inds_case{data_params.case_number}.csv")["dataset_index"].tolist()
        test_inds = pd.read_csv(
            f"{os.getcwd()}/data/indices/holdout_inds_case{data_params.case_number}.csv")["dataset_index"].tolist()
        val_inds = pd.read_csv(
            f"{os.getcwd()}/data/indices/val_inds_case{data_params.case_number}.csv")["dataset_index"].tolist()

        # truncate data for testing
        if self.config.training.testing_for_debugging:
            train_inds = train_inds[:50]
            test_inds = test_inds[:50]
            val_inds = val_inds[:50]

        data_processing_args = self._get_data_parameters_dict(train_inds, test_inds, val_inds)

        # Verify if all DataFrames in the list have the same column order.
        dataframes = [data_dict["data"] for _, data_dict in raw_data.items()]

        first_df_columns = dataframes[0].columns

        for df in dataframes[1:]:
            if not df.columns.equals(first_df_columns):
                raise ValueError("Dataframe columns of data are out of order, this is not allowed!")

        # if data_params.dataset_type == "state":
        #     Dataset = SeqToSeqDataset
        if data_params.dataset_type == "seq_to_label":  # predict the disruption label (0 or 1)
            Dataset = SeqToLabelDataset
        else:
            raise ValueError("Invalid dataset type.")

        train_dataset = Dataset(shots=[raw_data[i] for i in train_inds], **data_processing_args)
        if data_processing_args["scaling_type"] != "none":
            train_scaler = train_dataset.scale_data(scaling_type=data_processing_args["scaling_type"])

        # scale according to train dataset
        # all_dataset = Dataset(shots=[raw_data[i] for i in raw_data.keys()], **data_processing_args)
        # if data_processing_args["scaling_type"] != "none":
        #     all_dataset.scale_data(scaler=train_scaler)

        test_dataset = Dataset(shots=[raw_data[i] for i in test_inds], **data_processing_args)
        if data_processing_args["scaling_type"] != "none":
            test_dataset.scale_data(scaler=train_scaler)
    
        val_dataset = Dataset(shots=[raw_data[i] for i in val_inds], **data_processing_args)
        if data_processing_args["scaling_type"] != "none":
            val_dataset.scale_data(scaler=train_scaler)

        return train_dataset, test_dataset, val_dataset 
    

    def _get_data_parameters_dict(self, train_inds=None, test_inds=None, val_inds=None):
        data_params = self.config.data
        machine_hyperparameters = {
            "cmod": [data_params.cmod_hyperparameter_non_disruptions, data_params.cmod_hyperparameter_disruptions],
            "d3d": [data_params.d3d_hyperparameter_non_disruptions, data_params.d3d_hyperparameter_disruptions],
            "east": [data_params.east_hyperparameter_non_disruptions, data_params.east_hyperparameter_disruptions],
        }
        taus = {
            "cmod": data_params.tau_cmod,
            "d3d": data_params.tau_d3d,
            "east": data_params.tau_east,
        }
        data_processing_args = {
            "train_inds": train_inds if train_inds is not None else [],
            "test_inds": test_inds if test_inds is not None else [],
            "val_inds": val_inds if val_inds is not None else [],
            "end_cutoff_timesteps": data_params.end_cutoff_timesteps,
            "machine_hyperparameters": machine_hyperparameters,
            "dataset_type": data_params.dataset_type,
            "taus": taus,
            "data_augmentation_windowing": self.config.balance_classes.data_augmentation_windowing,
            "data_augmentation_ratio": self.config.balance_classes.data_augmentation_ratio,
            "scaling_type": data_params.scaling_type,
            "max_length": data_params.max_length,
            "use_smoothed_tau": data_params.use_smoothed_tau,
            "window_length": self.config.balance_classes.disruptivity_distance_window_length,
            "context_length": data_params.data_context_length,
            "smooth_v_loop": data_params.smooth_v_loop,
            "v_loop_smoother": data_params.v_loop_smoother,
            "delta_t": data_params.delta_t, # length of input sequence
            "uniform_seq_length": data_params.uniform_seq_length,
            "undersample": self.config.balance_classes.undersample,
            "undersample_ratio": self.config.balance_classes.undersample_ratio
        }
        return data_processing_args

def collate_fn_seq_to_label(dataset):
    """
    Takes in an instance of Torch Dataset, corresponding to the current match.
    dataset: (64, ...)
    Returns:
     * input_embeds: tensor, size: Batch x (padded) seq_length x embedding_dim
     * label_ids: tensor, size: Batch x 2
    """
    # dataset = list of 64 tensors/sequences (as batch size is 64)
    # for each df in dataset:
    # - df["inputs_embeds"]: [177, 12], [55, 12], [56, 12], etc...
    # - df["labels"]: [2] (non_disruption_prob, disruption_prob)    
    output = {}
    output['inputs_embeds'] = torch.stack([df["inputs_embeds"].to(dtype=torch.float16) for df in dataset], dim=0) 
    output['labels'] = torch.vstack([df["labels"].to(torch.float32) for df in dataset]) 
    return output 

def produce_smoothed_curve(shot_len, shot_tau, window_length):
    """Produce a smoothed curve to match the disruptivity curve.

    Args:
        shot_len (int): Length of the shot.
        shot_tau (int): Tau of the shot.
        window_length (int): Window length for smoothing.

    Returns:
        smoothed_curve (np.array): Smoothed curve to match the
            disruptivity curve.
    """

    curve_to_match = np.zeros(shot_len)
    curve_to_match[-shot_tau:] = 1

    curve_series = pd.Series(curve_to_match)

    # Calculate the moving average with a window size of 10
    smoothed_curve = curve_series.rolling(
        window=window_length,
        min_periods=1,
        center=True
    ).mean()

    return smoothed_curve


def augment_seq_to_label_data_windowing(
        dataset, data_augmentation_ratio,
        use_smoothed_tau,
        window_length, **kwargs):
    """
    Generate additional pairs (X_bar, y_bar) from each (X, y) by randomly sampled starting and ending
    windows of the shot.
    
    Next, in order to enforce greater reasoning about the start of instabilities, we set tau_r_i variable
    (avg length of time before a disruption occurs which is tokamak-dependent) per tokamak:
        if (X_bar, y_bar) of length n ends before n - tau_r_i, we set y_bar = 0 (No disruption)
    
    -------
    Augment a sequence to label dataset by taking smaller 
    windows of the data and labelling them according to tau.

    Args:
        dataset (SeqToLabelDataset): dataset.
        data_augmentation_ratio (int): Ratio of augmented data to original data.
        use_smoothed_tau (bool): Whether to use smoothed tau.
        window_length (int): Length of window to take in smoothing.

    Returns:
        aug_train_dataset (object): Augmented training data.
    """

    augmented_shots = {}
    augmented_shot_ind = 0

    taus = dataset.taus

    for j in range(len(dataset)):
        train_label = dataset[j]["labels"][1]

        augmented_shots[augmented_shot_ind] = {
            "label": train_label,
            "data": dataset[j]["inputs_embeds"],
            "machine": dataset[j]["machine"],
            "shot": dataset[j]["shot"],
        }
        augmented_shot_ind += 1

        for i in range(data_augmentation_ratio):
            shot_length = len(dataset[j]["inputs_embeds"])
            
            
            if shot_length <= dataset.uniform_seq_length:
                continue  # Skip if the sequence is too short to create a fixed-length window

            # Ensure the selected window has the desired fixed length
            window_start = np.random.randint(0, shot_length - dataset.uniform_seq_length)
            window_end = window_start + dataset.uniform_seq_length
            
            windowed_inputs_embeds = dataset[j]["inputs_embeds"][window_start:window_end, :]

            tau_value = taus[dataset[j]["machine"]]

            # if len(train_dataset[j]) - tau_value is within the window, label as 1, else 0
            if train_label < .5:
                label = 0

            elif use_smoothed_tau: # currently True
                smoothed_curve = produce_smoothed_curve(
                    shot_len=shot_length,
                    shot_tau=tau_value,
                    window_length=window_length
                )
                ind = min(window_end, shot_length - 1)
                label = smoothed_curve[ind]

            elif (shot_length - tau_value) < window_end:
                label = 1
            else:
                label = 0

            augmented_shots[augmented_shot_ind] = {
                "label": label,
                "data": windowed_inputs_embeds,
                "machine": dataset[j]["machine"],
                "shot": dataset[j]["shot"] + "_" + str(window_start) + "_to_" + str(window_end),
            }
            augmented_shot_ind += 1

    augmented_data = SeqToLabelDataset(
        shots=[augmented_shots[i] for i in range(augmented_shot_ind)],
        machine_hyperparameters=dataset.machine_hyperparameters,
        taus=dataset.taus,
        max_length=dataset.max_length,
        smooth_v_loop=dataset.smooth_v_loop,
        v_loop_smoother=dataset.v_loop_smoother,
    )

    return augmented_data

# SAVE DATASET AND LOAD

def save_load_datasets(tp, force_reload=False, save=True):
    # if not save:
    #     print("------------------------------------------------------------")
    #     print("Processing datasets...")
    #     print("------------------------------------------------------------")
    #     data_obj = Data(tp.config_file_path)
    #     raw_data = data_obj.load_raw_data()
    #     train_dataset, test_dataset, val_dataset, eval_dataset = data_obj.prepare_datasets(raw_data)
    
    # if save:
    
    config = tp.config #load_config(config_folder="config", config_file=tp.config_file_path, config_name="default", verbose=True)
    
    # 0. GET DATASET INDICES
    if tp.config.data.case_number == 6:
        dataset_dir = "/groups/tensorlab/sratala/fno-disruption-pred/datasets/case_6"
    elif tp.config.data.case_number == 8:
        dataset_dir = "/groups/tensorlab/sratala/fno-disruption-pred/datasets/case_8"
    elif tp.config.data.case_number == 12:
        dataset_dir = "/groups/tensorlab/sratala/fno-disruption-pred/datasets/case_12"
    elif tp.config.data.case_number == 1:
        dataset_dir = "/groups/tensorlab/sratala/fno-disruption-pred/datasets/case_1"
    elif tp.config.data.case_number == 2:
        dataset_dir = "/groups/tensorlab/sratala/fno-disruption-pred/datasets/case_2"
    elif tp.config.data.case_number == 3:
        dataset_dir = "/groups/tensorlab/sratala/fno-disruption-pred/datasets/case_3"
    else:
        raise ValueError(f"Unknown case number: {tp.config.data.case_number}")
        
    os.makedirs(dataset_dir, exist_ok=True)  # Ensure directory exists

    # 1. SCALE DATASETS
    if tp.config.data.scaling_type == "none":
        dataset_dir = f"/groups/tensorlab/sratala/fno-disruption-pred/datasets/case_{tp.config.data.case_number}/no_scale"
    elif tp.config.data.scaling_type == "standard":
        dataset_dir = f"/groups/tensorlab/sratala/fno-disruption-pred/datasets/case_{tp.config.data.case_number}/standard_scale"
    elif tp.config.data.scaling_type == "robust":
        dataset_dir = f"/groups/tensorlab/sratala/fno-disruption-pred/datasets/case_{tp.config.data.case_number}/robust_scale"
    else:
        raise ValueError(f"Unknown scaling type: {tp.config.data.scaling_type}")
    
    os.makedirs(dataset_dir, exist_ok=True)  # Ensure directory exists

    dataset_paths = {
        "train": os.path.join(dataset_dir, "train_dataset.pt"),
        "test": os.path.join(dataset_dir, "test_dataset.pt"),
        "val": os.path.join(dataset_dir, "val_dataset.pt"),
    }

    # Save datasets if they don't exist
    if force_reload or (not all(os.path.exists(path) for path in dataset_paths.values())):
        print("------------------------------------------------------------")
        print("Processing and saving datasets...")
        print("------------------------------------------------------------")
        data_obj = Data(tp.config_file_path)
        raw_data = data_obj.load_raw_data()

        # eval_dataset
        train_dataset, test_dataset, val_dataset  = data_obj.prepare_datasets(raw_data)

        # Save datasets
        torch.save(train_dataset, dataset_paths["train"])
        torch.save(test_dataset, dataset_paths["test"])
        torch.save(val_dataset, dataset_paths["val"])
        print("Datasets saved successfully!")
    else:
        print("------------------------------------------------------------")
        print("Datasets already exist, loading them...")
        print("------------------------------------------------------------")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.serialization.add_safe_globals({"SeqToLabelDataset": SeqToLabelDataset})
        # Load datasets
        train_dataset = torch.load(dataset_paths["train"], weights_only=False, map_location=device)
        test_dataset = torch.load(dataset_paths["test"], weights_only=False, map_location=device)
        val_dataset = torch.load(dataset_paths["val"], weights_only=False, map_location=device)
        print("Datasets loaded successfully!")
    
    # SPECIFIC DATA PROCESSING LOGIC
    ###################################
    # do the truncation/undersampling/oversampling here to save time. 
    # above datasets are "generic" and can be used across all tasks.
    # the only difference is scale/no scale for which we have different directories.
    # below we truncate for uniform length, and augment.
    
    train_dataset.move_data_to_device()
    test_dataset.move_data_to_device()
    val_dataset.move_data_to_device()
    
    # get the arguments    
    data_params = tp.config.data
    machine_hyperparameters = {
        "cmod": [data_params.cmod_hyperparameter_non_disruptions, data_params.cmod_hyperparameter_disruptions],
        "d3d": [data_params.d3d_hyperparameter_non_disruptions, data_params.d3d_hyperparameter_disruptions],
        "east": [data_params.east_hyperparameter_non_disruptions, data_params.east_hyperparameter_disruptions],
    }
    taus = {
        "cmod": data_params.tau_cmod,
        "d3d": data_params.tau_d3d,
        "east": data_params.tau_east,
    }
    data_processing_args = {
        "end_cutoff_timesteps": data_params.end_cutoff_timesteps,
        "machine_hyperparameters": machine_hyperparameters,
        "dataset_type": data_params.dataset_type,
        "taus": taus,
        "data_augmentation_windowing": config.balance_classes.data_augmentation_windowing,
        "data_augmentation_ratio": config.balance_classes.data_augmentation_ratio,
        "scaling_type": data_params.scaling_type,
        "max_length": data_params.max_length,
        "use_smoothed_tau": data_params.use_smoothed_tau,
        "window_length": config.balance_classes.disruptivity_distance_window_length,
        "context_length": data_params.data_context_length,
        "smooth_v_loop": data_params.smooth_v_loop,
        "v_loop_smoother": data_params.v_loop_smoother,
        "delta_t": data_params.delta_t, # length of input sequence
        "uniform_seq_length": data_params.uniform_seq_length,
        "probabilistic_labels": data_params.probabilistic_labels,
        "undersample": config.balance_classes.undersample,
        "undersample_ratio": config.balance_classes.undersample_ratio
    }
    
    # 2. AUGMENT DATASETS
    if data_processing_args["data_augmentation_windowing"] and config.data.dataset_type == "seq_to_label":
        raise ValueError("Augmentation should not be happening right now.")
        # if data_params.uniform_seq_length:
        #     raise ValueError("Data augmentation windowing is not supported for uniform length dataset.")
        train_dataset = augment_seq_to_label_data_windowing(train_dataset, **data_processing_args)
        val_dataset = augment_seq_to_label_data_windowing(val_dataset, **data_processing_args)
        # all_dataset = Data.augment_seq_to_label_data_windowing(all_dataset, **data_processing_args)
    
    # 3. APPLY END CUTOFF TIMESTEPS
    for dataset in [train_dataset, test_dataset, val_dataset]: # eval_dataset
        dataset.apply_end_cutoff_timesteps(data_processing_args["end_cutoff_timesteps"])
    
    # 4. TRUNCATE DATASET
    datasets = [train_dataset, test_dataset, val_dataset] # eval_dataset
    if tp.config.data.uniform_seq_length:
        for dataset in datasets:
            # make everything uniform
            dataset.make_uniform_seq(data_processing_args["delta_t"], tp.config.data.standardize_sampling_rate)
     
    # 5. UNDERSAMPLE DATASETS
    if data_processing_args["undersample"]:
        print('undersampling datasets...')
        train_dataset.undersample_majority(data_processing_args["undersample_ratio"])
    
    return train_dataset, test_dataset, val_dataset #, eval_dataset
