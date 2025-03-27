import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from IPython.display import display, HTML
import numpy as np
from dotenv import load_dotenv

plotting = False # Set this to true if you want lots of plots generated throughout the data processing

def set_up_disruptivity_prediction_plot(
        probs_len,
        threshold,
        lims = [0, 1]):
    
    """Setup the plot that will have a lot of disruptivity predictions on it.
    
    Args:
        probs_len (int): length of max sequence
        threshold (float): Threshold for predicting disruption.
    """
    fig, ax = plt.subplots()
    # plot threshold line as a dashed cyan horizontal line across the entire 
    # sequence length
    ax.plot([threshold]*probs_len, "c--")
    ax.set_xlabel('Time (or other common index)')
    ax.set_ylabel('Disruptivity')
    # set ax limits between 0 and 1
    ax.set_ylim(lims)
    plt.title("Predictions of Disruptivity across Holdout Set")
    return fig, ax

def save_figure(fig, plot_title, directory='figs'):
    """
    Saves the given figure to the specified directory with a title-based filename.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        plot_title (str): Title of the plot to use as part of the filename.
        directory (str): Directory where the figure should be saved.
    """
    if not plotting:
        return
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    # Create a valid filename from the plot title
    filename = os.path.join(directory, f"{plot_title}.png")
    fig.savefig(filename, bbox_inches='tight')
    print(f"Figure saved to: {filename}")



def visualize_scaled_data(inputs_embeds_before_scaling, inputs_embeds_after_scaling, before_scaling_dir, after_scaling_dir):
    """
    Visualize data before and after scaling.

    Parameters:
        shot_dfs (list): List of DataFrames before scaling.
        inputs_embeds (list): List of tensors after scaling.
        scaler (object): Scaler used for scaling the data.
        before_scaling_dir (str): Directory to save plots before scaling.
        after_scaling_dir (str): Directory to save plots after scaling.
    """
    if not plotting:
        return

    for i, shot_df in enumerate(shot_dfs[:3]):
        scaled_data = pd.DataFrame(inputs_embeds[i].numpy(), columns=shot_df.columns)

        # histograms before and after scaling
        fig, axes = plt.subplots(2, shot_df.shape[1], figsize=(15, 8))
        for j, column in enumerate(shot_df.columns):
            sns.histplot(shot_df[column], bins=20, ax=axes[0, j], color='blue', kde=True)
            axes[0, j].set_title(f'Before Scaling: {column}')
            sns.histplot(scaled_data[column], bins=20, ax=axes[1, j], color='green', kde=True)
            axes[1, j].set_title(f'After Scaling: {column}')

        plt.tight_layout()
        save_figure(fig, f'shot_{i}_histograms.png',f"{before_scaling_dir}")
        plt.show()

        # Boxplots for each feature
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(data=shot_df, ax=axes[0])
        axes[0].set_title(f'Shot {i}: Before Scaling')
        sns.boxplot(data=scaled_data, ax=axes[1])
        axes[1].set_title(f'Shot {i}: After Scaling')

        plt.tight_layout()
        save_figure(fig, f'shot_{i}_boxplots.png',f"{after_scaling_dir}")
        plt.show()


def line_plot(df, addl_title='', columns_to_plot=None, directory='figs'):
    if not plotting:
        return
    if not columns_to_plot:
        columns_to_plot = [
            "lower_gap", "ip_error_normalized", "beta_p", "v_loop",
            "kappa", "n_equal_1_normalized", "li", "Greenwald_fraction", "q95"
        ]
    for col in columns_to_plot:
        if col not in df.columns:
            print(f"Column '{col}' does not exist in the DataFrame.")
            continue
        fig = plt.figure(figsize=(10, 5))
        plt.plot(df["time"], df[col], label=col)
        plt.xlabel("Time")
        plt.ylabel(col)
        plot_title = f"{col} vs Time {addl_title}"
        plt.title(plot_title)
        plt.legend()
        plt.grid(True)
        plt.show()
        save_figure(fig, plot_title, directory)

def scatter_plot(df, addl_title='', directory='figs'):
    if not plotting:
        return
    discrete_columns = ["d3d", "east", "cmod"]
    for col in discrete_columns:
        fig = plt.figure(figsize=(10, 5))
        plt.scatter(df["time"], df[col], alpha=0.6, label=col)
        plt.xlabel("Time")
        plt.ylabel(col)
        plot_title = f"{col} vs Time"+addl_title
        plt.title(plot_title)
        plt.legend()
        plt.grid(True)
        plt.show()
        save_figure(fig, plot_title, directory)


def plot_timeseries_vs_features(df, addl_title='', directory='figs'):
    if not plotting:
        return
    continuous_columns = ['lower_gap', 'ip_error_normalized', 'beta_p', 'v_loop', 'kappa', 'n_equal_1_normalized',
                          'li', 'Greenwald_fraction', 'q95']
    fig = plt.figure(figsize=(50, 25))
    # Plot each continuous variable over time
    for col in continuous_columns:
        plt.plot(df['time'], df[col], label=col)
    plot_title = f'Continuous Variables vs. Time at {addl_title}'
    plt.title(plot_title, fontsize=36)  # Larger title
    plt.xlabel('Time', fontsize=25)  # Larger x-axis label
    plt.ylabel('Value', fontsize=25)  # Larger y-axis label
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.grid(True)
    plt.show()
    save_figure(fig, plot_title, directory)


def heatmap(df, addl_title='', directory='figs'):
    if not plotting:
        return
    data_for_heatmap = df.drop("time", axis=1).set_index(df["time"])
    fig = plt.figure(figsize=(12, 6))
    sns.heatmap(data_for_heatmap.T, cmap="coolwarm", cbar=True)
    plot_title = "Heatmap of Variables Over Time"+addl_title
    plt.title(plot_title)
    plt.xlabel("Time")
    plt.ylabel("Variables")
    save_figure(fig, plot_title, directory)


def interactive_plot(df, addl_title=''):
    if not plotting:
        return
    print('I SHOULDNT HAPPEN')
    # Display an overall title for the entire visualization set
    overall_title = f"Interactive Plots: {addl_title}"
    display(HTML(f"<h2 style='text-align: center;'>{overall_title}</h2>"))

    # Generate interactive plots for each column
    for col in df.columns:
        if col != "time":
            fig = px.line(df, x="time", y=col, title=f"{col} vs Time")
            fig.show()


def make_all_plots(df, interactive=False, addl_title='', directory='figs'):
    if not plotting:
        return
    line_plot(df=df, addl_title=addl_title, directory=directory)
    scatter_plot(df, addl_title, directory)
    plot_timeseries_vs_features(df, addl_title, directory)
    heatmap(df, addl_title, directory)
    if interactive:
        interactive_plot(df, addl_title)


def plot_labels(shots, addl_title='', directory='figs/labels'):
    if not plotting:
        return
    # Assuming 'shots' is a list of dictionaries containing the 'label' information.
    labels = []
    for shot in shots:
        label = shot['label']
        if 0 < label < 1:  # continuous label
            labels.append(label)
        else:  # discrete label
            labels.append(int(label))

    # Convert to a NumPy array for easier plotting
    labels = np.array(labels)

    # Find unique label values and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Print the exact values of the labels and their occurrences
    print("Unique label values and their counts:")
    for label, count in zip(unique_labels, counts):
        print(f"\tLabel: {label}, Count: {count}")

    # Plot the distribution of exact label values
    fig = plt.figure(figsize=(8, 6))
    plt.bar(unique_labels, counts, edgecolor='black', alpha=0.7)
    plot_title = "Exact Values of Labels"+addl_title
    plt.title(plot_title)
    plt.xlabel("Label Value")
    plt.ylabel("Frequency")
    plt.xticks(unique_labels)  # Display exact label values on x-axis
    save_figure(fig, plot_title, directory)


    # Create a histogram for the label distribution
    fig = plt.figure(figsize=(8, 6))

    # Plot histogram for continuous labels
    plt.hist(labels, bins=20, edgecolor='black', alpha=0.7)
    plot_title = "Distribution of Labels"+addl_title
    plt.title(plot_title)
    plt.xlabel("Label Value")
    plt.ylabel("Frequency")

    # Show the plot
    plt.show()
    save_figure(fig, plot_title, directory)

def plot_probabilities(labels, addl_title='', directory='figs/seq_to_label/probs'):
    if not plotting:
        return
    prob_pairs = []

    # Loop over the labels to ensure the correct order of probabilities
    for probabilities in labels:
        # Assuming probabilities is a torch tensor of shape (2,)
        if probabilities[0] > probabilities[1]:  # Non-disruption comes first
            prob_pairs.append([probabilities[0].item(), probabilities[1].item()])
        else:  # Disruption comes first
            prob_pairs.append([probabilities[1].item(), probabilities[0].item()])

    # Create DataFrame
    prob_df = pd.DataFrame(prob_pairs, columns=['Non-Disruption Probability', 'Disruption Probability'])

    # Plot heatmap
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(prob_df.T, cmap="coolwarm", cbar=True)
    plot_title = f"Heatmap of Disruption and Non-Disruption Probabilities {addl_title}"
    plt.title(plot_title)
    plt.xlabel("Shots")
    plt.ylabel("Probabilities")
    plt.show()

    # Save the figure (assuming save_figure is defined elsewhere)
    save_figure(fig, plot_title, directory)
