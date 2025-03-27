import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
import os 


def string_to_boolean(value: str) -> bool:
    # Convert string to lowercase and check for 'true'
    return value.strip().lower() == 'true'

def get_sequence_length_stats(dataset, name="Dataset"):
    """
    Computes and prints sequence length statistics for input embedding sequences,
    separately for disruptive (label=1) and non-disruptive (label=0) sequences.

    Args:
        dataset: The processed dataset object (e.g., train_dataset, test_dataset).
        name (str): Name of the dataset for printing.

    Returns:
        dict: A flat list of rows suitable for tabulate, containing overall stats, stats for disruptions (label=1),
              and stats for non_disruptions (label=0), plus machine-specific breakdowns.
    """
    sequence_lengths = [entry["inputs_embeds"].shape[0] for entry in dataset]
    disruptive_lengths = [entry["inputs_embeds"].shape[0] for entry in dataset if entry["label"] == 1]
    non_disruptive_lengths = [entry["inputs_embeds"].shape[0] for entry in dataset if entry["label"] == 0]
    cmod_disruptive_lengths = [entry["inputs_embeds"].shape[0] for entry in dataset if entry["label"] == 1 and entry["machine"] == "cmod"]
    d3d_disruptive_lengths = [entry["inputs_embeds"].shape[0] for entry in dataset if entry["label"] == 1 and entry["machine"] == "d3d"]
    cmod_non_disruptive_lengths = [entry["inputs_embeds"].shape[0] for entry in dataset if entry["label"] == 0 and entry["machine"] == "cmod"]
    d3d_non_disruptive_lengths = [entry["inputs_embeds"].shape[0] for entry in dataset if entry["label"] == 0 and entry["machine"] == "d3d"]
    east_disruptive_lengths = [entry["inputs_embeds"].shape[0] for entry in dataset if entry["label"] == 1 and entry["machine"] == "east"]
    east_non_disruptive_lengths = [entry["inputs_embeds"].shape[0] for entry in dataset if entry["label"] == 0 and entry["machine"] == "east"]

    def compute_stats(lengths):
        return {
            "Min": np.min(lengths) if lengths else None,
            "Max": np.max(lengths) if lengths else None,
            "Mean": round(np.mean(lengths), 2) if lengths else None,
            "Median": float(np.median(lengths)) if lengths else None,
            "Range": np.max(lengths) - np.min(lengths) if lengths else None,
        }

    stats = {
        "Overall": compute_stats(sequence_lengths),
        "Disruptions": compute_stats(disruptive_lengths),
        "Non-Disruptions": compute_stats(non_disruptive_lengths),
        "CMOD Disruptions": compute_stats(cmod_disruptive_lengths),
        "D3D Disruptions": compute_stats(d3d_disruptive_lengths),
        "CMOD Non-Disruptions": compute_stats(cmod_non_disruptive_lengths),
        "D3D Non-Disruptions": compute_stats(d3d_non_disruptive_lengths),
        "EAST Disruptions": compute_stats(east_disruptive_lengths),
        "EAST Non-Disruptions": compute_stats(east_non_disruptive_lengths),
    }

    def format_sequence_length_stats(name, stat_dict):
        rows = []
        for category, values in stat_dict.items():
            rows.append([
                name,
                category,
                values["Min"],
                values["Max"],
                values["Mean"],
                values["Median"],
                values["Range"]
            ])
        return rows

    return format_sequence_length_stats(name, stats)


def print_dataset_info(train_dataset, test_dataset, val_dataset): # eval_dataset
    """Print information about the datasets.
    Args:
        train_dataset (Dataset): The training dataset.
        test_dataset (Dataset): The testing dataset.
        val_dataset (Dataset): The validation dataset.
        
    Returns:
        None
    """
    print("----------------------------------------------------------")
    print("OVERALL DATASET STATS")
    print("----------------------------------------------------------")
    train_disruptions = sum(train_dataset.labels) # np.sum([(train_dataset[i]["labels"][-1].detach().clone().cpu() > .5).tolist() for i in range(len(train_dataset))])
    test_disruptions =  sum(test_dataset.labels) # np.sum([(test_dataset[i]["labels"][-1].detach().clone().cpu() > .5).tolist() for i in range(len(test_dataset))])
    val_disruptions =  sum(val_dataset.labels) # np.sum([(val_dataset[i]["labels"][-1].detach().clone().cpu() > .5).tolist() for i in range(len(val_dataset))])
    # eval_disruptions =  sum(eval_dataset.labels)
    train_pct = (train_disruptions / len(train_dataset)) * 100 if len(train_dataset) > 0 else 0
    test_pct = (test_disruptions / len(test_dataset)) * 100 if len(test_dataset) > 0 else 0
    val_pct = (val_disruptions / len(val_dataset)) * 100 if len(val_dataset) > 0 else 0
    #eval_pct = (eval_disruptions / len(eval_dataset)) * 100 if len(eval_dataset) > 0 else 0
    
    overall_stats = [
        ["Train", len(train_dataset), f"{train_pct:.2f}%"],
        ["Test", len(test_dataset), f"{test_pct:.2f}%"],
        ["Validation", len(val_dataset), f"{val_pct:.2f}%"],
    ]
    print(tabulate(overall_stats, headers=["Dataset", "Total Samples", "Disruption %"], tablefmt="grid"))
    
    print()

    def get_machine_stats(dataset, name):
        machine_counts = {}
        machine_disruptions = {}
        for entry in dataset:
            machine = entry["machine"]
            label = entry["label"]
            machine_counts[machine] = machine_counts.get(machine, 0) + 1
            if label == 1:
                machine_disruptions[machine] = machine_disruptions.get(machine, 0) + 1

        table_data = []
        for machine in sorted(machine_counts):
            count = machine_counts[machine]
            disruptions = machine_disruptions.get(machine, 0)
            pct = (disruptions / count) * 100 if count > 0 else 0
            table_data.append([name, machine.upper(), count, f"{pct:.2f}%"])

        return table_data

    print("----------------------------------------------------------")
    print("MACHINE SPECIFIC STATS")
    print("----------------------------------------------------------")
    all_stats = (
        get_machine_stats(train_dataset, "Train") +
        get_machine_stats(test_dataset, "Test") +
        get_machine_stats(val_dataset, "Validation")
    )
    print(tabulate(all_stats, headers=["Dataset", "Machine", "Total Samples", "Disruption %"], tablefmt="grid"))

    print("----------------------------------------------------------")
    print("SEQ LENGTH STATS")
    print("----------------------------------------------------------")
    train_stats = get_sequence_length_stats(train_dataset, name="Train")
    test_stats = get_sequence_length_stats(test_dataset, name="Test")
    val_stats = get_sequence_length_stats(val_dataset, name="Validation")
    all_seq_stats = train_stats + test_stats + val_stats
    print(tabulate(all_seq_stats, headers=["Dataset", "Category", "Min", "Max", "Mean", "Median", "Range"], tablefmt="grid"))
    print("--------------------")
    # datasets = [train_dataset, test_dataset, val_dataset]
    # for dataset, name in zip(datasets, ['TRAIN', 'TEST', 'VAL']):    
    #     plot_sequence_histogram_by_interval(dataset, name)


def plot_sequence_histogram_by_interval(dataset, dataset_type, case_num, min_interval=50, max_interval=1000, step=50):
    """
    Plots a histogram of sequence counts that fit into each interval length bucket,
    separated by machine type and disruption status.

    Args:
        dataset: The dataset object with make_uniform_seq() method and required fields.
        dataset_type: str - 'TRAIN', 'TEST', or 'VAL'
        min_interval (int): Minimum time interval in microseconds (must be multiple of 50).
        max_interval (int): Maximum time interval in microseconds (must be multiple of 50).
        step (int): Step size for interval length (must be multiple of 50).
    """
    assert min_interval % 50 == 0 and max_interval % 50 == 0 and step % 50 == 0, "All intervals must be multiples of 50."

    SAMPLING_RATES = {"cmod": 5, "d3d": 10, "east": 25}
    machine_colors = {"cmod": "lightskyblue", "d3d": "green", "east": "orange"}

    # Store original dataset state to restore after each interval
    original_dataset = copy.deepcopy(dataset)

    interval_buckets = list(range(min_interval, max_interval + 1, step))
    bar_data = defaultdict(lambda: defaultdict(lambda: {"disruptive": 0, "non_disruptive": 0}))
    old_len = len(dataset)
    for interval in interval_buckets:
        dataset.make_uniform_seq(interval)
        for inputs, label, machine in zip(dataset.inputs_embeds, dataset.labels, dataset.machines):
            label_type = "disruptive" if label == 1 else "non_disruptive"
            bar_data[interval][machine][label_type] += 1

        dataset = copy.deepcopy(original_dataset)
        assert len(dataset) == old_len

    fig, ax = plt.subplots(figsize=(15, 7))
    bar_width = step / 4  # narrower bars for better spacing
    x_ticks = interval_buckets
    bar_offset = {"cmod": -bar_width, "d3d": 0, "east": bar_width}

    for machine in machine_colors:
        for label_type in ["non_disruptive", "disruptive"]:
            heights = [bar_data[interval][machine][label_type] for interval in interval_buckets]
            positions = [interval + bar_offset[machine] for interval in interval_buckets]
            hatch = "//" if label_type == "disruptive" else None
            edgecolor = 'black' if label_type == "disruptive" else None
            linestyle = "dashed" if label_type == "disruptive" else "solid"

            ax.bar(
                positions,
                heights,
                width=bar_width,
                label=f"{machine.upper()} - {'Disruptive' if label_type == 'disruptive' else 'Non-Disruptive'}",
                color=machine_colors[machine],
                edgecolor=edgecolor,
                hatch=hatch,
                linestyle=linestyle,
                linewidth=1
            )

    ax.set_xlabel("Interval Length (μs)")
    ax.set_ylabel("Number of Sequences")
    ax.set_title("Number of Sequences by Interval Length and Machine")
    ax.set_xticks(interval_buckets)
    ax.set_xticklabels(interval_buckets, rotation=45)
    ax.legend(loc='upper left', fontsize='medium')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs('figs/sequence_length_stats', exist_ok=True)
    plt.savefig(f'figs/sequence_length_stats/CASE_{case_num}_{dataset_type}_histogram_by_interval_times.png')
    
    dataset = copy.deepcopy(original_dataset)
    assert len(dataset) == old_len
