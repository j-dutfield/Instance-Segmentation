# Instance-Segmentation

Instance segmentation task using the oxford pets dataset: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/oxford_iiit_pet.py

MRCNN used with coco pre-trained weights (ref in dependencies). Methodology is transfer learning from that start-point.


Script run on:

macOS Catalina 10.15.7

Run on Virtual Env with the following spec & dependencies:

Python 3.7

numpy
pandas
tensorflow=1.1.5
keras=2.2
sklearn
matplotlib
PIL
h5py=2.9.0
coco https://github.com/cocodataset/cocoapi
cv2
mcrnn https://github.com/matterport/Mask_RCNN

NOTES:

Only 1 epoch of 2 images currently set on training_script.py. This is becuase macbook could not handle more. Manually increase generator count and number of epochs for full training. 

Augmentation parameter on mrcnn in training mode can be set to augment inputs during full training.

Training weights are automatically saved as callback. The path to these weights needs to be added at the start of detection_script.py before running. 
