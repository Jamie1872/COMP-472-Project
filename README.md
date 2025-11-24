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

## To run code  (Run Evaluation Script)**

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

See `report.pdf` for detailed analysis.

## File Descriptions

### **Main Files:**
- `cifar10_image_classification.ipynb` - Complete training notebook with all outputs
- `evaluate_all_models.py` - Standalone evaluation script
- `cifar10_features.pkl` - Pre-processed dataset (ResNet-18 features + original images)
- `report.pdf` - Detailed project report with analysis

### **Model Files:**
- `.pkl` files - Saved scikit-learn models and custom Python models
- `.pth` files - Saved PyTorch model weights (state dictionaries)
- `.png` files - Confusion matrix visualizations

## Resources

- **CIFAR-10 Dataset:** https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html

**Last Updated:** 2025-11-23