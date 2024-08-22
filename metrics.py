from sklearn import metrics
from skimage import measure

import cv2
import numpy as np
import pandas as pd


def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    """
    Computes the best precision, recall and threshold for a given set of
    anomaly ground truth labels and anomaly prediction weights.
    """
    precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_precision = precision[np.argmax(f1_scores)]
    best_recall = recall[np.argmax(f1_scores)]
    print(best_threshold, best_precision, best_recall)

    return best_threshold, best_precision, best_recall


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).
    """
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0. if path == 'training' else metrics.average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    # 計算 Precision 和 Recall
    precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    
    # 選擇一個合適的閾值來計算F1
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    max_f1_index = np.argmax(f1_scores)
    best_threshold = thresholds[max_f1_index] if len(thresholds) > 0 else 0.5  # 確保有閾值存在
    
    f1 = f1_scores[max_f1_index]
    f1 = 0. if path == 'training' else f1
    precision = 0. if path == 'training' else precision
    recall = 0. if path == 'training' else recall
    # 返回對應的 Precision 和 Recall
    best_precision = 0. if path == 'training' else precision[max_f1_index]
    best_recall = 0. if path == 'training' else recall[max_f1_index]
    
    return {"auroc": auroc, "ap": ap,"f1": f1,"precision":best_precision,"recall":best_recall,"best_threshold": best_threshold}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='train'):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    auroc = metrics.roc_auc_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)
    ap = 0. if path == 'training' else metrics.average_precision_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)

    return {"auroc": auroc, "ap": ap}


def compute_pro(masks, amaps, num_th=200):
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    df = df[df["fpr"] < 0.3]
    df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc
