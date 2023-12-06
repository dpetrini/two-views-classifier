# 2 views CLASSIFIER - test script
#
# Test inference for 2 views mammograms
#
# run: python3 2views_clf_test.py -c [cc image file] -m [mlo image file]
#
# DGPP 06/Sep/2021

import argparse
import sys
from pathlib import Path
from warnings import filterwarnings

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.autograd import Variable
from tqdm import tqdm

from compute_metrics import pfbeta
from two_views_net import SideMIDBreastModel

filterwarnings("ignore")

log_level = "DEBUG"
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
logger.add(
    "logs.log",
    level=log_level,
    format=log_format,
    colorize=False,
    backtrace=True,
    diagnose=True,
)

TRAIN_DS_MEAN = 13369
NETWORK = "EfficientNet-b0"
TOPOLOGY = "side_mid_clf"
DEVICE = "gpu"
gpu_number = 1

TOP_LAYER_N_BLOCKS = 2
TOP_LAYER_BLOCK_TYPE = "mbconv"
USE_AVG_POOL = True
STRIDES = 2


def get_2views_model(model, model_file, device):
    """Load model weights from file"""
    print("Model 2views: ", model_file)
    model.load_state_dict(torch.load(model_file, map_location=device))

    return model


def load_model(network, topology):
    """load model structure and device"""
    if (DEVICE == "gpu") and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_number))
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    if topology == "side_mid_clf":
        model = SideMIDBreastModel(
            device,
            network,
            TOP_LAYER_N_BLOCKS,
            b_type=TOP_LAYER_BLOCK_TYPE,
            avg_pool=USE_AVG_POOL,
            strides=STRIDES,
        )
    else:
        raise NotImplementedError(f"Net type error: {topology}")

    model = model.to(device)

    return model, device


def standard_normalize(image):
    """Normalize accordingly for model"""
    image = np.float32(image)
    image -= TRAIN_DS_MEAN
    image /= 65535  # float [-1,1]

    return image


def make_prediction(image_cc, image_mlo, model, device):
    """
    Execute deep learning inference
    inputs: [vector of] image
    output: full image mask
    """
    img_cc = standard_normalize(image_cc)
    img_mlo = standard_normalize(image_mlo)
    img_cc = cv2.resize(img_cc, (896, 1152))
    img_mlo = cv2.resize(img_mlo, (896, 1152))

    img_cc_t = torch.from_numpy(img_cc.transpose(2, 0, 1))
    img_mlo_t = torch.from_numpy(img_mlo.transpose(2, 0, 1))
    batch_t = torch.cat([img_cc_t, img_mlo_t], dim=0)
    batch_t = batch_t.unsqueeze(0)

    # prediction
    with torch.no_grad():
        model.eval()  # if not here, BN is enabled and mess everything
        input = Variable(batch_t.to(device))
        output_t = model(input)

    pred = output_t.squeeze()
    pred = torch.softmax(pred, dim=0)

    return pred, batch_t


def simple_prediction(image_cc, image_mlo, model, device):
    """Execute simple inference"""
    tta_predictions = np.array([])
    for i in range(1, 2):
        aug_image_cc = image_cc
        aug_image_mlo = image_mlo
        prediction, _ = make_prediction(aug_image_cc, aug_image_mlo, model, device)
        tta_predictions = np.append(
            tta_predictions, prediction[1].cpu().detach().numpy()
        )

    return tta_predictions


def translation_aug(image_cc, image_mlo, model, device, type=None):
    """Execute inference with translation augmentation"""
    tta_predictions = np.array([])
    rows, cols, _ = image_cc.shape
    # Translation
    for i in range(-1, +2):
        for j in range(-1, +2):
            M = np.float32(
                [[1, 0, i * cols // 40], [0, 1, j * rows // 40]]
            )  # de 0.8414=>0.8476
            aug_image_cc = cv2.warpAffine(
                image_cc, M, (cols, rows), borderMode=cv2.BORDER_REFLECT
            )
            aug_image_mlo = cv2.warpAffine(
                image_mlo, M, (cols, rows), borderMode=cv2.BORDER_REFLECT
            )
            prediction, _ = make_prediction(aug_image_cc, aug_image_mlo, model, device)
            tta_predictions = np.append(
                tta_predictions, prediction[1].cpu().detach().numpy()
            )

    return tta_predictions


def read_dicom(path):
    img = pydicom.dcmread(path)
    img = np.array(img.pixel_array, dtype=np.float32)
    return img


def load_image(img_path: str):
    img_ext = Path(img_path).suffix
    if img_ext in [".dcm", ".dicom"]:
        image = read_dicom(img_path)
    elif img_ext in [".png", ".pgm"]:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError(f"Can't load extesion {img_ext}")
    image_ = np.zeros((*image.shape[0:2], 3), dtype=np.uint16)
    image_[:, :, 0] = image
    image_[:, :, 1] = image
    image_[:, :, 2] = image
    return image_


def predict(cc_path, mlo_path, device, model, use_aug=True):
    image_cc = load_image(cc_path)
    image_mlo = load_image(mlo_path)
    if not use_aug:
        tta_predictions = simple_prediction(image_cc, image_mlo, model, device)
    else:
        tta_predictions = translation_aug(image_cc, image_mlo, model, device)
    return np.mean(tta_predictions)


def get_patient_predict(df_patient, device, model, use_aug):
    if df_patient.shape[0] < 2:
        return float("nan")
    else:
        try:
            path_mlo = df_patient[df_patient["View"] == "MLO"].iloc[0]["Path_abs"]
            path_cc = df_patient[df_patient["View"] == "CC"].iloc[0]["Path_abs"]
            return predict(path_cc, path_mlo, device, model, use_aug=use_aug)
        except IndexError:
            return float("nan")


def compute_metrics(df_res: pd.DataFrame, thresh: float = 0.5):
    logger.info("-" * 100)
    df_res_ = df_res.dropna(subset=["Predict_prob"]).drop_duplicates(
        subset=["Patient_id", "Laterality"]
    )
    try:
        auc = roc_auc_score(df_res_["Cancer"].values, df_res_["Predict_prob"].values)
        logger.info(f"Overall AUC: {round(auc, 3)}")
    except ValueError:
        logger.info(
            "Only one class present in y_true. ROC AUC score is not defined in that case."
        )
    try:
        pf1 = pfbeta(df_res_["Cancer"].values, df_res_["Predict_prob"].values, beta=1)
        logger.info(f"Overall pF1-score: {round(pf1, 3)}")
    except ZeroDivisionError:
        logger.info("Division by zero")
    acc = accuracy_score(
        df_res_["Cancer"].values,
        (df_res_["Predict_prob"] > thresh).values.astype(np.int32),
    )
    logger.info(f"Overall accuracy: {round(acc, 3)}")
    f1 = f1_score(
        df_res_["Cancer"].values,
        (df_res_["Predict_prob"] > thresh).values.astype(np.int32),
    )
    logger.info(f"Overall F1-score: {round(f1, 3)}")
    logger.info("-" * 100)
    for dataset in df_res_["Dataset"].unique():
        df_dataset = df_res_[df_res_["Dataset"] == dataset]
        try:
            auc = roc_auc_score(
                df_dataset["Cancer"].values, df_dataset["Predict_prob"].values
            )
            logger.info(f"{dataset} AUC: {round(auc, 3)}")
        except ValueError:
            logger.info(
                "Only one class present in y_true. ROC AUC score is not defined in that case."
            )
        try:
            pf1 = pfbeta(
                df_dataset["Cancer"].values, df_dataset["Predict_prob"].values, beta=1
            )
            logger.info(f"{dataset} pF1-score: {round(pf1, 3)}")
        except ZeroDivisionError:
            logger.info("Division by zero")
        acc = accuracy_score(
            df_dataset["Cancer"].values,
            (df_dataset["Predict_prob"] > thresh).values.astype(np.int32),
        )
        logger.info(f"{dataset} accuracy: {round(acc, 3)}")
        f1 = f1_score(
            df_dataset["Cancer"].values,
            (df_dataset["Predict_prob"] > thresh).values.astype(np.int32),
        )
        logger.info(f"{dataset} F1-score: {round(f1, 3)}")
        logger.info("-" * 100)


# <<<<<<<<<<<<<<<<<< main <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def main():
    ap = argparse.ArgumentParser(
        description="[Poli-USP] Two Views Breast Cancer inference"
    )
    ap.add_argument(
        "--mode", required=True, help="mode of testing: test_two or test_many"
    )
    ap.add_argument("-c", "--cc", help="CC image file.")
    ap.add_argument("-m", "--mlo", help="MLO image file.")
    ap.add_argument("-d", "--model", help="two-views detector model")
    ap.add_argument(
        "-a", "--aug", help="select to use translation augmentation: -a true"
    )
    ap.add_argument("--data", help="path to dataset csv file")

    args = vars(ap.parse_args())

    if args["model"]:
        model_file = args["model"]
    else:
        model_file = "models_side_mid_clf_efficientnet-b0/2021-08-03-03h54m_100ep_1074n_last_model_BEST.pt"

    use_aug = False
    if args["aug"]:
        if args["aug"].lower() == "true":
            use_aug = True

    print(f"\n--> {NETWORK} {TOPOLOGY} \n")

    model, device = load_model(NETWORK, TOPOLOGY)
    # now overwirte the original model with 2-views-pre-trained for test
    model = get_2views_model(model, model_file, device)
    logger.info(f"Mode: {args['mode']}")
    logger.info(f"Use inference augmentation: {use_aug}")
    logger.info("Model loaded")
    logger.info("Start inference")
    if args["mode"] == "test_two":
        file_cc = args["cc"]
        file_mlo = args["mlo"]
        pred = predict(file_cc, file_mlo, device, model, use_aug=use_aug)
        logger.info(f"\nPrediction: {pred:.4f}")
    elif args["mode"] == "test_many":
        assert (
            args["data"] is not None
        ), "You should specify test data csv for mode `test_many`"
        path_annotation = args["data"]
        df_ann = (
            pd.read_csv(path_annotation)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        df_ann["Predict_prob"] = [float("nan")] * df_ann.shape[0]
        path_save = path_annotation.replace(".csv", "_preds.csv")
        for i, patient in enumerate(
            tqdm(df_ann["Patient_id"].unique(), desc="Patient")
        ):
            df_patient_L = df_ann[
                (df_ann["Patient_id"] == patient) & (df_ann["Laterality"] == "L")
            ]
            df_patient_R = df_ann[
                (df_ann["Patient_id"] == patient) & (df_ann["Laterality"] == "R")
            ]
            pred_L = get_patient_predict(df_patient_L, device, model, use_aug)
            pred_R = get_patient_predict(df_patient_R, device, model, use_aug)
            df_ann.loc[
                (df_ann["Patient_id"] == patient) & (df_ann["Laterality"] == "L"),
                "Predict_prob",
            ] = pred_L
            df_ann.loc[
                (df_ann["Patient_id"] == patient) & (df_ann["Laterality"] == "R"),
                "Predict_prob",
            ] = pred_R
            if (i + 1) % 20 == 0:
                df_ann.to_csv(path_save, index=False)
                logger.info(
                    f"Dataframe with predictions saved to {path_save} on patient {i}"
                )
                try:
                    patients = df_ann["Patient_id"].unique()
                    df_test = df_ann[df_ann["Patient_id"].isin(patients[: i + 1])]
                    compute_metrics(df_test)
                except Exception as e:
                    print(str(e))

        df_ann.to_csv(path_save, index=False)
        logger.info(f"Dataframe with predictions saved to {path_save}")
        logger.info("Computing final metrics")
        compute_metrics(df_ann)
    else:
        print(f"No mode `{args['mode']}`. Available modes: `test_two`, `test_many`")


if __name__ == "__main__":
    main()
