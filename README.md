# two-views-classifier
Two Views breast cancer classifier

# Instructions for inference with two views

python3 2views_clf_test.py -h

usage: 2views_clf_test.py [-h] -c CC -m MLO [-d MODEL] [-a AUG]

[Poli-USP] Two Views Breast Cancer inference

optional arguments:
  -h, --help               show this help message and exit
  
  -c CC, --cc CC           CC image file.
  
  -m MLO, --mlo MLO        MLO image file.
  
  -d MODEL, --model MODEL  two-views detector model (default model already included)
  
  -a AUG, --aug AUG        select to use translation augmentation: -a true
