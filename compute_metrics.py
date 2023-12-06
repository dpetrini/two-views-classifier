# from two_views_clf_test import compute_metrics
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np


def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0
    

def compute_metrics(df_res: pd.DataFrame, thresh: float = 0.5):
    logger.info("-" * 100)
    df_res_ = df_res.dropna(subset = ["Predict_prob"]).drop_duplicates(subset = ["Patient_id", "Laterality"])
    try:
        auc = roc_auc_score(df_res_["Cancer"].values, df_res_["Predict_prob"].values)
        logger.info(f"Overall AUC: {round(auc, 3)}")
    except ValueError:
        logger.info("Only one class present in y_true. ROC AUC score is not defined in that case.")
    try:
        pf1 = pfbeta(df_res_["Cancer"].values, df_res_["Predict_prob"].values, beta = 1)
        logger.info(f"Overall pF1-score: {round(pf1, 3)}")
    except ZeroDivisionError:
        logger.info("Division by zero")
    acc = accuracy_score(df_res_["Cancer"].values,
                          (df_res_["Predict_prob"] > thresh).values.astype(np.int32))
    logger.info(f"Overall accuracy: {round(acc, 3)}")
    f1 = f1_score(df_res_["Cancer"].values, (df_res_["Predict_prob"] > thresh).values.astype(np.int32))
    logger.info(f"Overall F1-score: {round(f1, 3)}")
    logger.info("-" * 100)
    for dataset in df_res_["Dataset"].unique():
        df_dataset = df_res_[df_res_["Dataset"] == dataset]
        try:
            auc = roc_auc_score(df_dataset["Cancer"].values, df_dataset["Predict_prob"].values)
            logger.info(f"{dataset} AUC: {round(auc, 3)}")
        except ValueError:
            logger.info("Only one class present in y_true. ROC AUC score is not defined in that case.")
        try:
            pf1 = pfbeta(df_dataset["Cancer"].values, df_dataset["Predict_prob"].values, beta = 1)
            logger.info(f"{dataset} pF1-score: {round(pf1, 3)}")
        except ZeroDivisionError:
            logger.info("Division by zero")
        acc = accuracy_score(df_dataset["Cancer"].values,
                              (df_dataset["Predict_prob"] > thresh).values.astype(np.int32))
        logger.info(f"{dataset} accuracy: {round(acc, 3)}")
        f1 = f1_score(df_dataset["Cancer"].values, (df_dataset["Predict_prob"] > thresh).values.astype(np.int32))
        logger.info(f"{dataset} F1-score: {round(f1, 3)}")
        logger.info("-" * 100)


if __name__ == '__main__':
    data_path = "/home/zakharov_test/data/classification_cancer/all_data_preds.csv"
    df = pd.read_csv(data_path)
    patients = df["Patient_id"].unique()
    df_test = df[df["Patient_id"].isin(patients[:3480])]
    print(df_test["Cancer"].value_counts())
    compute_metrics(df_test)