# Comprehensive Road Scene Understanding for Autonomous Driving
This repository contains the implementation of a university project aiming to "design, train, and evaluate
an anomaly segmentation model" for road scenes, using the EoMT (Everything on Mask Transformers) architecture.
## Student Produced Files
The following files were specifically developed to follow the project instructions:
### Core Project Steps
* `Segmentation Project-STEP_4.ipynb`: Initial model setup and baseline evaluations.
* `Segmentation Project-STEP_5.ipynb`: Implements the fine-tuning of the COCO-pretrained model on the Cityscapes dataset.
* `Segmentation Project-STEP_7.ipynb`: Evaluates the performance of the pretrained ERFNet model on the anomaly segmentation task.
* `Segmentation Project-STEP_8_COCO+Cityscapes.ipynb`: Evaluates the performance of the COCO-pretrained and Cityscapes-pretrained models on the anomaly segmentation task.
* `Segmentation Project-STEP_8_Finetuned.ipynb`: Evaluates the performance of the final fine-tuned model on the anomaly segmentation task, and performs post-hoc calibration with temperature scaling.
### Utility and Tools
* `COCO_contiguous_labels.txt`: Contiguous labeling of COCO dataset used as a reference to create the mapping to Cityscapes dataset classes. ([Source](https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json))
* `Push_To_GitHub.ipynb`: A utility script used to push updates to GitHub.
* `tool_ckpt_to_bin.ipynb`: A conversion tool to transform PyTorch Lightning checkpoints (.ckpt) into PyTorch weight files (.bin) for easier loading.
