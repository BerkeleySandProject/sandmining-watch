def compute_metrics(ground_truth, predicted):
    TP = ((predicted == 1) & (ground_truth == 1)).sum()  # True Positives
    FP = ((predicted == 1) & (ground_truth == 0)).sum()  # False Positives
    FN = ((predicted == 0) & (ground_truth == 1)).sum()  # False Negatives
    TN = ((predicted == 0) & (ground_truth == 0)).sum()  # True Negatives

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score
