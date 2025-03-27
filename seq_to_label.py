from torch.utils.data import Dataset
import copy
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch.nn.functional as F
import plotting
import utils
import numpy as np
import random

# SAMPLING_RATES = {"cmod": 0.005, "d3d": 0.01, "east": 0.025}  # in milliseconds
SAMPLING_RATES = {"cmod": 5, "d3d": 10, "east": 25}  # in microseconds (multiply abobe by 10^3)

class SeqToLabelDataset(Dataset):
    """
    Torch Dataset for model-ready data.

    Args:
        shots (list): List of shots.
        max_length (int): Maximum length of the input sequence.

    Attributes:
        input_embeds (list): List of input embeddings.
        labels (list): List of labels.
        shot_inds (list): List of shot inds
        shot_lens (list): List of shot lengths.
    """
    def __init__(
        self,
        shots,
        machine_hyperparameters,
        taus,
        max_length,
        smooth_v_loop,
        v_loop_smoother,
        **kwargs,
    ):
        self.inputs_embeds = []
        self.labels_probs = []
        self.num_disruptions = 0
        self.machines = []
        self.machine_hyperparameters = machine_hyperparameters
        self.max_length = max_length
        self.smooth_v_loop = smooth_v_loop
        self.v_loop_smoother = v_loop_smoother
        self.taus = taus
        self.shot_inds = []
        
        self.labels = []

        # Make a plot of the labels/see what values the labels range from.
        plotting.plot_labels(shots)

        for i, shot in enumerate(shots):
            # we discard shots that are shorter than 125ms
            if not (22 < len(shot["data"]) < max_length): # 22 < seq_len < 2048
                #print(f"Removing shot {i} because it has too few rows in dataframe.")
                continue

            shot_df = copy.deepcopy(shot['data']) # (seq_len, 12)
            
            # shot_df is sometimes a tensor of shape (seq_len, 12) so time col has already been removed. 
            if isinstance(shot_df, pd.DataFrame): # remove time column if not already a tensor with time column removed
                shot_df.drop(columns=["time"], inplace=True)
                # set nan values in v_loop column to be the first valid mean in v_loop column
                if smooth_v_loop:
                    utils.set_nans_to_mean(shot_df, "v_loop", v_loop_smoother)
    
            # GET LABELS
            # set machine hyperparameter scaling of 0,1 labels.   
            [non_disruptions_prob, disruptions_prob] = machine_hyperparameters[shot['machine']]
            is_disruptive = int(shot["label"])
            
            label_onehot = torch.nn.functional.one_hot(torch.tensor(is_disruptive, dtype=torch.long), num_classes=2).float()

            if 0 < shot["label"] < 1:
                # shot["label"]: (0.05 0.92); this is true sometimes
                l = max(min(shot["label"], disruptions_prob), non_disruptions_prob)
                probabilities = torch.tensor([1 - l, l], dtype=torch.float32)   # (non_disruptions_prob, disruptions_prob)
            # Put non_disruptions_prob first and disruptions_prob second when the label is 1 (disruption).
            elif is_disruptive:  # shot["label"] is 1.
                # machine scaling of probabilities, usually 0.9 for disruption, 0.1 for not.
                # shot["label"]: 0 for non-disruption, 1 for disruption
                # probabilities: [non_disruptions_prob, disruptions_prob] = [0.1, 0.9], (0.0500, 0.9200)
                probabilities = torch.tensor([non_disruptions_prob, disruptions_prob], dtype=torch.float32)
            # Flip them when the label is 0 (non-disruption).
            else:
                probabilities = torch.tensor([disruptions_prob, non_disruptions_prob], dtype=torch.float32)

            # GET INPUT EMBEDDINGS
            # manually set this value and later can change end cutoff timesteps via the apply function
            # Don't apply end cutoff timesteps just yet!!
            end_cutoff_timesteps = 0
            shot_end = int(len(shot_df) - end_cutoff_timesteps) # gets too messy right before disruption, hard to train on
            input_embedding = torch.clone(torch.tensor(shot_df[:shot_end].values if isinstance(shot_df, pd.DataFrame) else shot_df[:shot_end], dtype=torch.float32)).detach()

            self.inputs_embeds.append(input_embedding)  # input_embedding = (len(shot_df), 12) = (num_rows, 12)
            #self.labels_probs.append(probabilities)  # (2,) [non_disruptions_prob, disruptions_prob] if disruption, else [disruption, non disruption]
            self.labels_probs.append(probabilities)
            self.labels.append(is_disruptive)
            self.num_disruptions += shot["label"] # 1 = disruption, 0 = non-disruption
            self.machines.append(shot["machine"])
            self.shot_inds.append(str(shot["shot"]))  # int representing identifier for the shot
    
        self.move_data_to_device()
        
    # overall: inputs_embeds is (num_shots_in_range, len(shot_df), 12)
    #          labels is (num_shots_in_range, 2,)

    def apply_end_cutoff_timesteps(self, end_cutoff_timesteps):
        """Cuts off a certain interval length from the end of the sequence
        based on the sequence's sampling rate which is based on the machine
        type.

        Args:
            end_cutoff_timesteps (int): number of timesteps to slice off from the end     
        """
        if end_cutoff_timesteps == 0:
           return 
        
        # get the seq length per machine
        for machine in SAMPLING_RATES:
            sampling_rate_micro_seconds = SAMPLING_RATES[machine]
            print(f'End cutoff timesteps (µs) used for {machine}: ', end_cutoff_timesteps*sampling_rate_micro_seconds)
        
        self.inputs_embeds = [inputs[:-end_cutoff_timesteps, :] for inputs in self.inputs_embeds]
        
    def make_uniform_seq(self, time_interval_micro_seconds, standardize_sampling_rate=False):
        # delta_t can change so don't make it class variable 
        # Starting from the end of inputs_embeds, get delta_t sequence lengths 
        # and add that as a new input embedding with the correct label, etc. 
        # ended in a disruption, else non disruption.
        # total number of disruptions is the same
        
        # compute closest sequence length to time_interval_micro_seconds that are divisible by all sampling rates
        lcm_sampling_rate = 50 #math.lcm(*SAMPLING_RATES.values())
        time_interval_micro_seconds = round(time_interval_micro_seconds / lcm_sampling_rate) * lcm_sampling_rate
        
        print("Time interval (µs) used across all reactor data: ", time_interval_micro_seconds)
        
        # get the seq length per machine
        seq_len_per_machine = {}
        for machine in SAMPLING_RATES:
            sampling_rate_micro_seconds = SAMPLING_RATES[machine] if not standardize_sampling_rate else 5 # already uniform
            seq_len_per_machine[machine] = int(time_interval_micro_seconds / sampling_rate_micro_seconds)
        
        new_inputs_embeds = []
        new_labels = []
        new_labels_probs = []
        new_machines = []
        new_shot_inds = []
        
        for idx, inputs in enumerate(self.inputs_embeds):
            # inputs is (seq_len, 12)
            seq_len = seq_len_per_machine[self.machines[idx]]
            if len(inputs) < seq_len:
                continue
            inputs = inputs[-seq_len:, :]  # keep last `seq_len` timesteps
            new_inputs_embeds.append(inputs)
            new_labels.append(self.labels[idx])
            new_labels_probs.append(self.labels_probs[idx])
            new_machines.append(self.machines[idx])
            new_shot_inds.append(self.shot_inds[idx])
        
        self.inputs_embeds = new_inputs_embeds
        self.labels = new_labels
        self.labels_probs = new_labels_probs
        self.machines = new_machines
        self.shot_inds = new_shot_inds
        self.num_disruptions = sum(self.labels)
    
    def __len__(self):
        return len(self.inputs_embeds)


    def __getitem__(self, idx):
        assert(isinstance(self.inputs_embeds[idx], torch.Tensor))
        assert(isinstance(self.labels_probs[idx], torch.Tensor))
        return {
            'inputs_embeds': self.inputs_embeds[idx],
            'labels': self.labels_probs[idx], # probabilities (non_dis_prob, dis_prob)
            'machine': self.machines[idx],
            'shot': self.shot_inds[idx],
            'label': self.labels[idx], # 0 for non-disruption, 1 for disruption
        }
    
    def undersample_majority(self, undersample_ratio=2.0):
        # undersample_ratio can change so don't make it a class variable 
        """
        Undersamples the majority class while keeping the minority class intact.
        Keeps a target ratio of majority to minority class.

        Args:
            target_ratio (float): Desired ratio of non-disruptions to disruptions.
        """
        
        print(f'Undersampling majority class with ratio {undersample_ratio}...')

        majority_indices = [i for i, label in enumerate(self.labels) if label == 0]
        minority_indices = [i for i, label in enumerate(self.labels) if label == 1]

        num_minority = len(minority_indices)
        num_majority_to_keep = int(num_minority * undersample_ratio)

        if len(majority_indices) > num_majority_to_keep:
            majority_indices = random.sample(majority_indices, num_majority_to_keep)

        selected_indices = majority_indices + minority_indices
        random.shuffle(selected_indices)

        # Subset data
        self.inputs_embeds = [self.inputs_embeds[i] for i in selected_indices]
        self.labels_probs = [self.labels_probs[i] for i in selected_indices]
        self.labels = [self.labels[i] for i in selected_indices]
        self.num_disruptions = sum(self.labels)
        self.machines = [self.machines[i] for i in selected_indices]
        self.shot_inds = [self.shot_inds[i] for i in selected_indices]


    def scale_data(self, scaler=None, scaling_type="standard"):
        """
        Robustly scale the data.
        shot_dfs: list of dataframs of processed shots

        Returns:
            scaler (object): Scaler used to scale the data.
        """
        # (num_shots, rows_per_shot, cols) -> list of tensors of the df values
        # (num_shots *rows_per_shot, cols)
        
        # inputs_embeds_bef_scaling = [seq.clone() for seq in self.inputs_embeds]

        # CREATE NEW SCALAR
        new_scaler = False
        if not scaler:
            new_scaler = True
            if scaling_type == 'robust':
                scaler = RobustScaler()
            elif scaling_type == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError(f'Invalid scaling type: {scaling_type}')
            # remove the outer dimension seperating the shots so rows of all the shot go together to get 2D tensor
            # this is because the number of rows in each shot varies.
            inputs_embeds_cat = torch.cat(self.inputs_embeds) # inputs_embeds is (num_shots_in_range*shot_df_rows, 12) = (46190, 12)
            scaler.fit(inputs_embeds_cat.cpu().numpy())

        num_shots = len(self.inputs_embeds)
        for i in range(num_shots):
            # Center and scale the data
            self.inputs_embeds[i] = torch.tensor(scaler.transform(self.inputs_embeds[i].cpu().numpy()), dtype=torch.float32).to(self.inputs_embeds[i].device) # (shot_df_rows, 12) shape doesn't change
    
        if new_scaler:
            threshold_std = 1.5
        else:
            threshold_std = 1
    
        self.check_scaled_data(torch.cat(self.inputs_embeds).cpu().numpy(), threshold_std=threshold_std)


    def check_scaled_data(self, df_scaled, threshold_std=3.0):
        df_scaled = pd.DataFrame(df_scaled)
        stds = df_scaled.std(axis=0)
        for column, std in zip(df_scaled.columns, stds):
            if column == 0:
                print("------------------------------------------------------------")
                print("Checking if data is scaled properly...")
            if std > threshold_std:
                print(f'Column {column} has high standard deviation after scaling: {std}')
        print("------------------------------------------------------------")


    def move_data_to_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(len(self.inputs_embeds)):
            self.inputs_embeds[i] = self.inputs_embeds[i].to(device)
            self.labels_probs[i] = self.labels_probs[i].to(device)
        return


    def subset(self, indices):
        """
        Args:
            indices (list): List of indices to subset the dataset with.
        Returns:
            subset (ModelReadyDataset): Subset of the dataset.
        """
        # removed the end cutoff timesteps here, must manually apply them
        if isinstance(indices, int):
            indices = [indices]
        subset_inputs_embeds = [self.inputs_embeds[i] for i in indices]
        subset_labels = [self.labels_probs[i] for i in indices]
        subset_machines = [self.machines[i] for i in indices]
        subset_shots = [self.shot_inds[i] for i in indices]
        subset = SeqToLabelDataset(
            [],
            machine_hyperparameters=self.machine_hyperparameters,
            taus=self.taus,
            max_length=self.max_length,
            smooth_v_loop=self.smooth_v_loop,
            v_loop_smoother=self.v_loop_smoother,
        )
        subset.inputs_embeds = subset_inputs_embeds
        subset.labels_probs = subset_labels
        subset.machines = subset_machines
        subset.shot_inds = subset_shots
        return subset

    def concat(self, new_dataset):
        """Concatenate this dataset with another dataset.
        Args:
            new_dataset (ModelReadyDataset): Dataset to concatenate with.
        Returns:
            concat_dataset (ModelReadyDataset): Concatenated dataset."""
        # removed the end cutoff timesteps here, must manually apply them
        concat_dataset = SeqToLabelDataset(
            [],
            machine_hyperparameters=self.machine_hyperparameters,
            taus=self.taus,
            max_length=self.max_length,
            smooth_v_loop=self.smooth_v_loop,
            v_loop_smoother=self.v_loop_smoother,
        )
        concat_dataset.inputs_embeds = self.inputs_embeds + new_dataset.inputs_embeds
        concat_dataset.labels_probs = self.labels_probs + new_dataset.labels
        concat_dataset.machines = self.machines + new_dataset.machines
        concat_dataset.shot_inds = self.shot_inds + new_dataset.shots
        return concat_dataset
    
