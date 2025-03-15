from evaluate.module import EvaluationModule
from typing import Tuple
import random
import evaluate
import numpy as np
from pprint import pprint


def computing_metric(prediction_and_actual: Tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Computes the accuracy, precision, F1 score, and recall for given predictions and actual labels.

    Args:
        prediction: The predicted labels.
        actual: The actual ground truth labels.

    Returns:
        A dictionary containing the computed metrics: accuracy, precision, recall, and F1 score.
    """
    prediction, actual = prediction_and_actual
    
    # Load the evaluation modules for each metric
    acc: EvaluationModule = evaluate.load("accuracy")
    precision: EvaluationModule = evaluate.load("precision")
    f1_score: EvaluationModule = evaluate.load("f1")
    recall: EvaluationModule = evaluate.load("recall")
    
    # Compute the metrics
    accuracy = acc.compute(predictions=prediction, references=actual)
    precision_score = precision.compute(predictions=prediction, references=actual)
    f1 = f1_score.compute(predictions=prediction, references=actual)
    recall_score = recall.compute(predictions=prediction, references=actual)
    
    # Returning all metrics as a dictionary with the values
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision_score["precision"],
        "f1_score": f1["f1"],
        "recall": recall_score["recall"]
    }


random.seed(7)
# Creating fake prediction and actual lists for testing
predictions = [random.choice([0, 1]) for _ in range(100)]
actual_labels = [random.choice([0, 1]) for _ in range(100)]
predictions_actual = (predictions, actual_labels)

print(predictions_actual)
metric_dict = computing_metric(prediction_and_actual=predictions_actual)
pprint(metric_dict)