# 2 views CLASSIFIER - test script
#
# Test inference for 2 views mammograms
#
# run: python3 2views_clf_test.py -c [cc image file] -m [mlo image file]
#
# DGPP 06/Sep/2021

import argparse
import numpy as np
import torch
from torch.autograd import Variable
import cv2

from two_views_net import SideMIDBreastModel

TRAIN_DS_MEAN = 13369
NETWORK = 'EfficientNet-b0'
TOPOLOGY = 'side_mid_clf'
DEVICE = 'gpu'
gpu_number = 0

TOP_LAYER_N_BLOCKS = 2
TOP_LAYER_BLOCK_TYPE = 'mbconv'
USE_AVG_POOL = True
STRIDES=2


def get_2views_model(model, model_file, device):
    """ Load model weights from file  """
    print('Model 2views: ', model_file)
    model.load_state_dict(torch.load(model_file, map_location=device))

    return model


def load_model(network, topology):
    """ load model structure and device """
    if (DEVICE == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(gpu_number))
    else:
        device = torch.device("cpu")
    if topology == 'side_mid_clf':
        model = SideMIDBreastModel(device, network, TOP_LAYER_N_BLOCKS,
                                   b_type=TOP_LAYER_BLOCK_TYPE, avg_pool=USE_AVG_POOL,
                                   strides=STRIDES)
    else:
        raise NotImplementedError(f"Net type error: {topology}")

    model = model.to(device)

    return model, device

def standard_normalize(image):
    """ Normalize accordingly for model """
    image = np.float32(image)
    image -= TRAIN_DS_MEAN
    image /= 65535    # float [-1,1]

    return image


def make_prediction(image_cc, image_mlo, model, device):
    """ 
    Execute deep learning inference
    inputs: [vector of] image
    output: full image mask
    """
    img_cc = standard_normalize(image_cc)
    img_mlo = standard_normalize(image_mlo)

    img_cc_t = torch.from_numpy(img_cc.transpose(2, 0, 1))
    img_mlo_t = torch.from_numpy(img_mlo.transpose(2, 0, 1))
    batch_t = torch.cat([img_cc_t, img_mlo_t], dim=0)
    batch_t = batch_t.unsqueeze(0)

    # prediction
    with torch.no_grad():
        model.eval()        # if not here, BN is enabled and mess everything
        input = Variable(batch_t.to(device))
        output_t = model(input)

    pred = output_t.squeeze()
    pred = torch.softmax(pred, dim=0)

    return pred, batch_t


def simple_prediction(image_cc, image_mlo, model, device):
    """ Execute simple inference """
    tta_predictions = np.array([])
    for i in range(1,2):
        aug_image_cc = image_cc
        aug_image_mlo = image_mlo
        prediction, _ = make_prediction(aug_image_cc, aug_image_mlo, model, device)
        tta_predictions = np.append(tta_predictions, prediction[1].cpu().detach().numpy())
    
    return tta_predictions


def translation_aug(image_cc, image_mlo, model, device, type=None):
    """ Execute inference with translation augmentation """
    tta_predictions = np.array([])
    rows, cols, _ = image_cc.shape
    # Translation
    for i in range(-1, +2):
        for j in range(-1, +2):
            M = np.float32([[1, 0, i*cols//40], [0, 1, j*rows//40]]) # de 0.8414=>0.8476
            aug_image_cc = cv2.warpAffine(image_cc, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
            aug_image_mlo = cv2.warpAffine(image_mlo, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
            prediction, _ = make_prediction(aug_image_cc, aug_image_mlo, model, device)
            tta_predictions = np.append(tta_predictions, prediction[1].cpu().detach().numpy())
    
    return tta_predictions


# <<<<<<<<<<<<<<<<<< main <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def main():

    ap = argparse.ArgumentParser(description='[Poli-USP] Two Views Breast Cancer inference')
    ap.add_argument("-c", "--cc", required=True, help="CC image file.")
    ap.add_argument("-m", "--mlo", required=True, help="MLO image file.")
    ap.add_argument("-d", "--model", help="two-views detector model")
    ap.add_argument("-a", "--aug", help="select to use translation augmentation: -a true")

    args = vars(ap.parse_args())

    file_cc = args['cc']
    file_mlo = args['mlo']

    if args['model']:
        model_file = args['model']
    else:
        model_file = 'models_side_mid_clf_efficientnet-b0/2021-08-03-03h54m_100ep_1074n_last_model_BEST.pt'

    use_aug = False
    if args['aug']:
        if args['aug'].lower() == 'true':
            use_aug = True

    print(f'\n--> {NETWORK} {TOPOLOGY} \n')

    model, device = load_model(NETWORK, TOPOLOGY)

    # now overwirte the original model with 2-views-pre-trained for test
    model = get_2views_model(model, model_file, device)

    image = cv2.imread(file_cc, cv2.IMREAD_UNCHANGED)

    image_cc = np.zeros((*image.shape[0:2], 3), dtype=np.uint16)
    image_cc[:, :, 0] = image
    image_cc[:, :, 1] = image
    image_cc[:, :, 2] = image

    image = cv2.imread(file_mlo, cv2.IMREAD_UNCHANGED)

    image_mlo = np.zeros((*image.shape[0:2], 3), dtype=np.uint16)
    image_mlo[:, :, 0] = image
    image_mlo[:, :, 1] = image
    image_mlo[:, :, 2] = image

    if not use_aug:
        tta_predictions = simple_prediction(image_cc, image_mlo, model, device)
    else:
        tta_predictions = translation_aug(image_cc, image_mlo, model, device)
    pred = np.mean(tta_predictions)

    print(f'\nPrediction: {pred:.4f}')

if __name__ == '__main__':
    main()
