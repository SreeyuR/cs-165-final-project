from data import Data
import os
import csv
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

def stratified_split(indices, labels, train_size=0.7, val_size=0.15, test_size=0.15, seed=50):
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices, labels, train_size=train_size, random_state=seed, stratify=labels
    )
    val_ratio = val_size / (val_size + test_size)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, train_size=val_ratio, random_state=seed, stratify=temp_labels
    )
    return train_idx, val_idx, test_idx

def save_indices_csv(path, indices, raw_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'dataset_index', 'machine', 'shot'])
        for idx, i in enumerate(indices):
            writer.writerow([idx, i, raw_data[i]['machine'], raw_data[i]['shot']])

def main(raw_data):
    machine_to_indices = defaultdict(list)
    for i, data in raw_data.items():
        machine_to_indices[data['machine']].append(i)

    for machine, indices in machine_to_indices.items():
        labels = [raw_data[i]['label'] for i in indices]
        train_idx, val_idx, test_idx = stratified_split(indices, labels)

        base_dir = os.path.join('data', 'indices', machine)
        save_indices_csv(os.path.join(base_dir, 'train.csv'), train_idx, raw_data)
        save_indices_csv(os.path.join(base_dir, 'val.csv'), val_idx, raw_data)
        save_indices_csv(os.path.join(base_dir, 'test.csv'), test_idx, raw_data)

if __name__ == "__main__":
    data_obj = Data("/Users/u235567/Desktop/cs-165-final-project/config/config_data_generation.yaml")
    raw_data = data_obj.load_raw_data()
    main(raw_data)
