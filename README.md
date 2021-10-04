# two-views-classifier
Two Views breast cancer classifier. Two view classifier for breast cancer. This is the inference code of the EfficientNet-based classifier to classify the two mammography views at once. It was trained in CBIS-DDSM dataset with original test split. It means that any pair of mammograms in test set can be used in this inference.

## Instructions for inference with two views

python3 2views_clf_test.py -h

```
usage: 2views_clf_test.py [-h] -c CC -m MLO [-d MODEL] [-a AUG]

[Poli-USP] Two Views Breast Cancer inference

optional arguments:
  -h, --help               show this help message and exit
  -c CC, --cc CC           CC image file.
  -m MLO, --mlo MLO        MLO image file.
  -d MODEL, --model MODEL  two-views detector model (default model already included)
  -a AUG, --aug AUG        select to use translation augmentation: -a true
  
```

  Example:
```
  python3 2views_clf_test.py -c samples/Calc-Test_P_00127_RIGHT_CC.png -m samples/Calc-Test_P_00127_RIGHT_MLO.png
```
Obs. Some sample files from CBIS-DDSM test set are included in samples folder for evaluation. Files were resized for network input.
Obs2. In order to perform test inference download our two-views model from [here](https://drive.google.com/file/d/1mOicNn1lCtXxXb2ficPmWFOnR4HM5c5M/view?usp=sharing) and place it in "models_side_mid_clf_efficientnet-b0" folder.

### Acknowlegments
Parts of EfficientNet from https://github.com/lukemelas/EfficientNet-PyTorch/ is included here and slightly modified, based in version 0.7.0.

### Dependencies
argparse
numpy
torch
cv2
