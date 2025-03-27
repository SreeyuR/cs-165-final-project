import json
import pickle
import numpy as np
import torch.nn.functional as F
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn import metrics
import evaluate as hf_evaluate # from hugging face
from sklearn.metrics import auc, roc_auc_score, classification_report, roc_auc_score, confusion_matrix, average_precision_score, precision_recall_fscore_support, precision_recall_curve
import random
import seaborn as sns
import utils
import os
import plotting
import model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def compute_metrics_testing(training_pipeline, trainer=None, threshold=0.6):
    """
    Evaluate the model using various metrics and save the results.

    Args:
        training_pipeline (TrainingPipeline): The pipeline containing the model and datasets.
        trainer: must be defined in training_pipeline if it's not defined into this function

    Returns:
        None
    """
    print("Evaluating...")
    
    #outputs = trainer.predict(training_pipeline.test_dataset)
    if training_pipeline.trainer:
        trainer = training_pipeline.trainer
    # Evaluate final trained model on test dataset
    
    print(len(training_pipeline.test_dataset))
    
    outputs = trainer.predict(training_pipeline.test_dataset)
    # -> for each shot, binary labels: [P(non-disrupt), P(disrupt)])
    true_labels_probs, logits = outputs.true_labels_probs, outputs.prediction_probs # (2805, 2), # (2805, 2) -> raw logits from model (not probabilities yet between 0 and 1)

    eval_metrics = compute_metrics(logits, true_labels_probs, threshold=threshold)
    last_epoch = training_pipeline.config.training.epochs
    
    if training_pipeline.config.wandb.log or (training_pipeline.config.optuna.use_optuna and training_pipeline.config.optuna.wandb_log):
        wandb.log({ "eval_f1": eval_metrics['f1'],
                    "eval_accuracy": eval_metrics['accuracy'],
                    "eval_precision": eval_metrics['precision'],
                    "eval_recall": eval_metrics['recall'],
                    "eval_auc": eval_metrics['auc_score'],
                    "eval_auprc": eval_metrics['auprc'],
                    "eval_disruptions_predicted_pctg": eval_metrics['disruptions_predicted_pctg'],
                    "eval_cm": wandb.Image(eval_metrics['cm_fig']),
                    }) 
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    # Vanilla labelling refers to simple labelling technique to go from regression -> classification problem (using all possible thresholds)
    fpr = eval_metrics["fpr"]
    tpr = eval_metrics["tpr"]
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    best_idx = np.argmin(distances)
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]
    plot = plt.plot(fpr, tpr, label="Test ROC curve (area = %0.3f)" % eval_metrics["auc_score"])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.plot(best_fpr, best_tpr, "r*", markersize=15)
    # Annotate with text
    plt.annotate(f"FP={best_fpr:.3f}\nTP={best_tpr:.3f}",
                xy=(best_fpr, best_tpr),
                xytext=(best_fpr + 0.1, best_tpr - 0.1),
                fontsize=12,
                color='red',
                arrowprops=dict(facecolor='red', shrink=0.05))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC")
    plt.legend(loc="lower right")
   
    # Save or log plot
    if training_pipeline.config.wandb.log or (training_pipeline.config.optuna.use_optuna and training_pipeline.config.optuna.wandb_log):
        run_name = wandb.run.name
    else:
        run_name = training_pipeline.config.wandb.name
    
    run_name = run_name.replace(" ", "_").replace("/", "_")
    # save_path = f"{os.getcwd()}/figs/{run_name}"
    # os.makedirs(save_path, exist_ok=True)
    # img_path = f"{save_path}/roc_curve_epoch_{epoch + 1}.png"
    # plt.savefig(img_path)
    save_path = f"{os.getcwd()}/figs/{run_name}/EVAL_ROC_PLOT"
    os.makedirs(save_path, exist_ok=True)
    img_path = f"{save_path}/roc_curve_epoch_{last_epoch}.png"
    plt.savefig(img_path)

    if training_pipeline.config.wandb.log or (training_pipeline.config.optuna.use_optuna and training_pipeline.config.optuna.wandb_log):
        wandb.log({f"eval_roc_curve_epoch_{last_epoch}": wandb.Image(img_path)})

    plt.close()
        
    
    # # plot fpr and tpr as a line plot
    # if training_pipeline.wandb_log:
    #     wandb.log({"roc_test": wandb.Image(plot[0]), "auc_test": eval_metrics["auc_score"]})
    #     # log eval metrics
    #     for metric_name, metric_value in eval_metrics.items(): # eval_metrics
    #         wandb.log({metric_name: metric_value})
        
    return eval_metrics


def compute_auc(y_true, y_pred_probs):
    """Compute the area under the curve (AUC) for a given set of predictions.
    
    Args:
        y_true (np.array): True labels.
        y_pred_probs (np.array): probability estimates of the positive class, not thresholded!

    Returns:
        auc (float): Area under the curve (AUC).
        fpr (np.array): False positive rate array based on roc_curve computed thresholds.
        tpr (np.array): True positive rate based on roc_curve computed thresholds.
    """

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_probs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc #, fpr, tpr


def compute_metrics(logits, true_labels_probs, threshold=0.6):
    """
    
    All methods for fpr, accuracy, tpr, auc_score, auprc match!
    
    Mismatch in precision:
       My Way: [0.8588, 1.0]
       Eval Metrics Way One: 0.0
       Eval Metrics Way Two: 0.4294
    
    Mismatch in f1:
        My Way: [0.924, 0.0]
        Eval Metrics Way One: 0.0
        Eval Metrics Way Two: 0.462
        
    Mismatch in recall:
        My Way: [1.0, 0.0]
        Eval Metrics Way One: 0.0
        Eval Metrics Way Two: 0.5

    Args:
        logits (_type_): _description_
        true_labels_probs (_type_): _description_
        threshold (float, optional): _description_. Defaults to 0.6.

    Returns:
        _type_: _description_
    """
    
    # disruption probabilities
    pred_probs = model.get_d_pred_probabilities(logits)  # (num_shots,) 
    # 1 if disruption probability >= thresh=0.6, 0 otherwise
    pred_labels = (pred_probs >= threshold).astype(int) #.to(torch.int) #.astype(int)
   
    # labels are probabilities before: (nd_prob, d_prob)    
    # SHAPE after coverting to true labels: (num_shots,): [1 0 0 ... 1 0 0]
    true_labels = np.argmax(true_labels_probs, axis=-1) #torch.argmax(true_labels_probs, dim=-1) # converts one-hot encoding/probs into class label representation.
    
    
    ##########################################################################
    my_accuracy = np.sum(true_labels == pred_labels) / len(true_labels)
    my_prec, my_rec, my_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, zero_division=1)
    my_auroc = roc_auc_score(true_labels, pred_probs)
    my_precision_curve, my_recall_curve, _ = precision_recall_curve(true_labels, pred_probs)
    my_auprc = auc(my_recall_curve, my_precision_curve)
        
    output = {
        'accuracy': my_accuracy, # class indepedent
        f'auc_score': my_auroc,
        f'auprc': my_auprc,
        f'precision': my_prec,
        f'recall': my_rec,
        f'f1': my_f1,
    }
    
    output = {k: np.round(v, 4) for k, v in output.items()}
    for k in output:
        if isinstance(output[k], np.ndarray):
            output[k] = list([float(x) for x in output[k]])
        else:
            output[k] = float(output[k])
    
    print('my eval metrics')
    for key, value in output.items():
        print(f"  {key}: {value}")
          

    # auc isn't thresholded, so it needs prediction probabilities
    auc_score = compute_auc(true_labels, pred_probs) #roc_auc_score(true_labels, pred_labels) 
    
    # roc_curve is for auc and expects probabilities for preds and True binary labels for labels
    fpr, tpr, _ = metrics.roc_curve(true_labels, pred_probs) 
    # precision_recall_curve is for auc and expects probabilities for preds and True binary labels for labels
    precision_curve, recall_curve, thresholds = precision_recall_curve(true_labels, pred_probs, pos_label=1)
    auprc = auc(recall_curve, precision_curve) # (x, y)
    
    # classification_report() expects labels
    report = metrics.classification_report( # expects 1d arays for y_true and y_pred
        y_true=true_labels,
        y_pred=pred_labels, # not supposed to be probs
        digits=4,
        output_dict=True,
        zero_division=0 # replace undefined metrics (precision, recall, F1) with 0.0 instead of raising warnings
    )

    cm = confusion_matrix(true_labels, pred_labels)
    # true positive rate
    disruptions_predicted_pctg = (cm[1][1] / (cm[1][1] + cm[1][0])) * 100 # TP/TP+FN
    
    # Compute and flip the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm = cm[::-1, ::-1]  # flip vertically and horizontally

    # Plot
    cm_fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False, ax=ax)

    # Set custom axis labels
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    # Set custom tick labels
    ax.set_xticklabels(['1', '0'])
    ax.set_yticklabels(['1', '0'])
    
    
    eval_metrics_way_two = {
        "f1": report["weighted avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "auc_score": auc_score,
        "auprc": auprc,
        "fpr": fpr, # a list
        "tpr": tpr,  # a list
        "confusion_matrix": cm,
        "disruptions_predicted_pctg": disruptions_predicted_pctg,
        "cm_fig": cm_fig,
    }
    
    return eval_metrics_way_two
            
    
