# nnU-Net for Pancreas Segmentation + Classification

This repository is a fork of [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) with a modified architecture to enable both segmentation and classification with a single model. The new classifier decoder head shares the same encoder as the default segmentation decoder head.

### Methods

An additional classification decoder head is added to the original architecture, taking as input the bottleneck features from the shared segmentation encoder (contraction pathway). 

The following classes have been added:
1. `CustomUNet`: Modifies the original generic u-net architecture with classification heads of various configurations. Located in [custom_unet.py](https://github.com/alif-munim/nnUNet/blob/classification/nnunet/network_architecture/custom_unet.py).
2. `nnUNetTrainerV2_Custom`: Combines segmentation and classification losses for training. Located in [custom_trainer.py](https://github.com/alif-munim/nnUNet/blob/classification/nnunet/training/network_training/custom_trainer.py).


### Experiments

Alongside the dice loss in the original setup, cross-entropy loss is used for the classification head. Various architectures are tested to find the ideal configuration. The experiments / configurations can be divided into the following categories:

1. Network depth. Tested networks with 2, 4, 5, 6, and 10 layers.
2. Batch normalization. To increase training stability.
3. Features. Feeding the classifier features from the u-net bottleneck, or later upsampling layers. Also combining both for local and global feature capture.
4. Pooling. Global average and attention.
5. Dropout. Set to 0.5 in most cases, but final experiment is set to 0.3.
6. Loss weighting. Assigning higher weight to classification loss for better performance.


### Training

To train, we use the custom trainer with the following command. The fold must be 0 to preserve the original UHN dataset split for training and validation.

```
nnUNet_train 3d_fullres nnUNetTrainerV2_Custom Task006_PancreasUHN 0
```

![Losses and Evaluation Metrics](https://i.ibb.co/8XfvfCX/final-progress-min.jpg)

Complete training run. Solid lines for segmentation, dash-dotted lines for classification. Red denotes training loss and yellow denotes validation loss. Green for evaluation performance (dice score for segmentation, accuracy for classification).

### Results

The Metrics Reloaded framework was used to determine a more comprehensive suite of evaluations.


### Environments and Requirements

To install the modified nnU-Net, clone the repository, checkout the classification branch, and install in editable mode.

```
git clone https://github.com/alif-munim/nnUNet.git
cd nnUNet
git checkout classification
pip install -e .
```

The environment variables below must be set in order to run the data preprocessing, training, and inference.

```
export nnUNet_raw="nnUNet/nnUNet_raw"
export nnUNet_preprocessed="nnUNet/nnUNet_preprocessed"
export nnUNet_results="nnUNet/nnUNet_results"
export RAW_DATA_PATH="nnUNet/original_data"

export nnUNet_raw_data_base="nnUNet/nnUNet_raw_data_base"
export nnUNet_preprocessed="nnUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="nnUNet/nnUNet_trained_models"
```


### Dataset

The dataset consists of 360 de-identified 3D pancreas CT scans from the University Health Network (UHN).

```
Train/
├── subtype0/
│   ├── quiz_0_041.nii.gz # Mask (0-background; 1-pancreas; 2-lesion)
│   ├── quiz_0_041_0000.nii.gz # Image
│   ├── ...
├── subtype1/
│   ├── ...
├── subtype2/
│   ├── ...
```

```
Validation/
├── subtype0/
│   ├── quiz_0_168.nii.gz # Mask (0-background; 1-pancreas; 2-lesion)
│   ├── quiz_0_168_0000.nii.gz # Image
│   ├── ...
├── subtype1/
│   ├── ...
├── subtype2/
│   ├── ...
```

```
Test/
├── quiz_037_0000.nii.gz # Image
├── quiz_045_0000.nii.gz # Image
├── quiz_047_0000.nii.gz # Image
├── ...
```

To be compatible with nnU-Net, the directory structure is reformatted as follows:

```
Task006_PancreasUHN/
├── dataset.json
├── class_mapping.json
├── imagesTr/
│   ├── case_001_0000.nii.gz
│   ├── case_002_0000.nii.gz
│   ├── ...
├── imagesTs/
│   ├── case_289_0000.nii.gz
│   ├── case_300_0000.nii.gz
│   ├── ...
└── labelsTr/
    ├── case_001.nii.gz
    ├── case_002.nii.gz
    ├── ...
```

Where `dataset.json` contains the train and test cases, and `class_mapping.json` maps each case to a subtype (0, 1, or 2) as a classification label.


### Preprocessing

Most of the preprocessing for the dataset is done in [pancreas.ipynb](https://github.com/alif-munim/nnUNet/blob/classification/pancreas.ipynb). A custom dataset split is created in [splits.ipynb](https://github.com/alif-munim/nnUNet/blob/classification/splits.ipynb). To preserve the original training and validation split, since nnU-Net uses k-fold cross validation by default.

Once processed, the dataset must be moved to `nnUNet/nnUNet_raw_data_base/nnUNet_raw_data` and the folder must be named in the format `TaskXXX_Dataset`.

Then, we must verify the cases and their label alignment with the following command.

```
nnUNet_plan_and_preprocess -t 006 --verify_dataset_integrity
```

After verification, we can run the full planning and preprocessing for image normalization.

```
nnUNet_plan_and_preprocess -t 006
```

The [hippocampus.ipynb](https://github.com/alif-munim/nnUNet/blob/classification/hippocampus.ipynb) notebook is a general reference / practice file for planning and preprocessing.




### Inference

> Note: if training was stopped before completion, you must rename your checkpoints to include model_final_checkpoint.model and model_final_checkpoint.model.pkl. You can either do this with model_latest.model or model_best.model.

**Classification** <br>

> Note: set self.inference_mode = False in custom_trainer.py before running.

For classification inference on the validation set, run the following:
```
nnUNet_predict -i "original_data/pancreas_validation/images" -o "original_data/pancreas_validation_preds" -t 6 -tr nnUNetTrainerV2_Custom -m 3d_fullres --num_threads_preprocessing 28 --classification -f 5 -chk model_latest --overwrite_existing
```

For classification inference on the test set, run the following:
```
nnUNet_predict -i "original_data/pancreas_test" -o "original_data/pancreas_test_labels" -t 6 -tr nnUNetTrainerV2_Custom -m 3d_fullres --num_threads_preprocessing 28 --classification -f 5 -chk model_latest  --overwrite_existing
```

**Segmentation** <br>

> Note: set self.inference_mode = True in custom_trainer.py before running.

For segmentation inference on the validation set, run the following:
```
nnUNet_predict -i "original_data/pancreas_validation/images" -o "original_data/pancreas_validation_preds" -t 6 -tr nnUNetTrainerV2_Custom -m 3d_fullres --num_threads_preprocessing 28 -f 5 -chk model_best --overwrite_existing
```

For segmentation inference on the test set, run the following:
```
nnUNet_predict -i "original_data/pancreas_test" -o "original_data/pancreas_test_preds" -t 6 -tr nnUNetTrainerV2_Custom -m 3d_fullres --num_threads_preprocessing 28 -f 5 -chk model_best --overwrite_existing
```

The data is prepared for predictions and visualized in [prediction.ipynb](https://github.com/alif-munim/nnUNet/blob/classification/prediction.ipynb).

### Evaluation

Evaluation is done using Metrics Reloaded in [evaluation.ipynb](https://github.com/alif-munim/nnUNet/blob/classification/evaluation.ipynb).

```
nnUNet_evaluate_folder -ref "original_data/pancreas_validation/labels"  -pred "original_data/pancreas_validation_preds" -l 0 1 2
```

