# COMP-472-Project
# CIFAR-10 Image Classification Project

## Overview

This project implements and compares four different AI/ML models for image classification on the CIFAR-10 dataset:

1. **Naive Bayes** (Custom + Scikit-learn implementations)
2. **Decision Trees** (Custom + Scikit-learn implementations, various depths)
3. **Multi-Layer Perceptrons (MLP)** (PyTorch, 5 variants)
4. **Convolutional Neural Networks (VGG11)** (PyTorch, 3 variants)

**Total Models Trained:** 16 variants across 4 model types

## Development Environment

### **All code was developed in Google Colab**

The main Jupyter notebook (`Trained Models.ipynb`) contains:
- Complete training code for all 4 model types
- All outputs, results, and visualizations saved in-notebook
- Confusion matrices for each model
- Training logs and accuracy metrics

**You can view all results directly in the notebook without running any code**

## To run code  (Run evaluate_all_models.py)

To independently verify the reported results using the saved models:

#### **1. Install Required Libraries**

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn pillow
```

Or using conda:
```bash
conda install pytorch torchvision numpy scikit-learn matplotlib seaborn pillow -c pytorch
```

#### **2. Run the Evaluation Script**

```bash
python evaluate_all_models.py
```

**What this script does:**
- Loads all pre-trained models from their respective folders
- Loads test data from `cifar10_features.pkl`
- Evaluates all 16 model variants on the test set
- Prints accuracy, precision, recall, and F1-score for each model
- **Runtime: ~10-30 seconds** (no training, just evaluation)

**Important:** The evaluation script uses the saved models that were originally trained in the Colab notebook. It does NOT retrain models - it simply loads them and verifies the reported metrics.

See `Report.pdf` for detailed analysis.

project/
│
├── Trained Models.ipynb # Main Colab notebook (all training code + outputs)
├── evaluate_all_models.py # Quick evaluation script for graders
├── cifar10_features.pkl # Preprocessed data (features + images)
├── cifar10_features_output.txt # expected output of evaluate_all_models.py
│
├── Naive Bayes/
│ ├── naive_bayes_custom.pkl
│ ├── naive_bayes_sklearn.pkl
│ └── confusion_matrix_*.png
│
├── Decision Tree/
│ ├── decision_tree_custom_depth5.pkl
│ ├── decision_tree_custom_depth10.pkl
│ ├── decision_tree_custom_depth20.pkl
│ ├── decision_tree_custom_depth30.pkl
│ ├── decision_tree_custom_depth50.pkl
│ ├── decision_tree_sklearn_depth50.pkl
│ └── confusion_matrix_*.png
│
├── MLP/
│ ├── mlp_base.pth
│ ├── mlp_shallow.pth
│ ├── mlp_deep.pth
│ ├── mlp_small.pth
│ ├── mlp_large.pth
│ └── confusion_matrix_*.png
│
├── VGG11/
│ ├── vgg11_base.pth
│ ├── vgg_shallow.pth
│ ├── vgg_large_kernel.pth
│ └── confusion_matrix_*.png
│
├── README.md # This file
└── Report.pdf # Project report

## File Descriptions

### **Main Files:**
- `cifar10_image_classification.ipynb` - Complete training notebook with all outputs
- `evaluate_all_models.py` - Standalone evaluation script
- `cifar10_features.pkl` - Pre-processed dataset (ResNet-18 features + original images)
- `Report.pdf` - Detailed project report with analysis

### **Model Files:**
- `.pkl` files - Saved scikit-learn models and custom Python models
- `.pth` files - Saved PyTorch model weights (state dictionaries)
- `.png` files - Confusion matrix visualizations

## Resources

- **CIFAR-10 Dataset:** https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html

**Last Updated:** 2025-11-23