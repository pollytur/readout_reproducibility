import ast
import numpy as np
import pandas as pd
from cascade.utility.metrics import Metrics


def load_submission_data(submission_path):
    """
    Extracts necessary data for model evaluation from the submitted csv file.

    Args:
        submission_path (str): Complete path to the submission file.

    Returns:
        tuple: Contains:
               - trial indices (1D array)
               - image IDs (1D array)
               - neuron IDs (1D array)
               - predictions (2d array: trials x neurons)
    """
    submission_df = pd.read_csv(submission_path)
    trial_idx = submission_df["trial_indices"].values
    image_ids = submission_df["image_ids"].values
    neuron_ids = np.array(ast.literal_eval(submission_df["neuron_ids"].values[0]))
    predictions = np.array(
        [ast.literal_eval(v) for v in submission_df["prediction"].values]
    )

    return trial_idx, image_ids, neuron_ids, predictions


def load_groundtruth_data(groundtruth_path):
    """
    Extracts necessary data for model evaluation from the ground truth data file.

    Args:
        groundtruth_path (str): Complete path to the ground truth data file.

    Returns:
        tuple: Contains:
               - trial indices (1D array)
               - image IDs (1D array)
               - neuron IDs (1D array)
               - predictions (2d array: trials x neurons)
    """
    raise NotImplementedError()


def evaluate(submission_path, ground_truth_path):
    """
    Compute evaluation metrics for a specific submission given the ground truth data.

    Args:
        submission_path (str): Complete path to the submission csv file.
        ground_truth_path (str): Complete path to the ground truth data file.

    Returns:
        dict: Containing all the evaluation results for all the evaluation metrics.
    """
    trial_idx_gt, image_ids_gt, neuron_ids_gt, responses = load_groundtruth_data(
        ground_truth_path
    )
    (
        trial_idx_submitted,
        image_ids_submitted,
        neuron_ids_submitted,
        predictions,
    ) = load_submission_data(submission_path)

    metric = Metrics(responses, trial_idx_gt, image_ids_gt, neuron_ids_gt)

    output = {}
    output["correlation_to_single_trials"] = metric.correlation_to_single_trials(
        predictions, trial_idx_submitted, neuron_ids_submitted, per_neuron=False
    )
    output[
        "correlation_to_mean_across_repeats"
    ] = metric.correlation_to_mean_across_repeats(
        predictions, trial_idx_submitted, neuron_ids_submitted, per_neuron=False
    )
    output["feve"] = metric.feve(
        predictions, trial_idx_submitted, neuron_ids_submitted, per_neuron=False
    )
    return output
