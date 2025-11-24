"""
CIFAR-10 Model Evaluation Script
=================================
This script loads all pre-trained models and evaluates them on the test set.
Doesnt do the training - just loads saved models(that were already trained) and computes metrics.

Run this to verify all reported results

"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("CIFAR-10 Model evalutaion - of all models")
print("="*70)

# ==================== Load Test Data ====================
print("\n2.Loading test data")
with open('cifar10_features.pkl', 'rb') as f:
    data = pickle.load(f)

test_features_50 = data['test_features_50']
test_labels = data['test_labels']
test_images = data['test_images']
classes = data['classes']

print(f"Loaded {len(test_labels)} test samples")

# ==================== Helper Functions ====================

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and return metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'Model': model_name,
        'Accuracy': f"{accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1-Score': f"{f1:.4f}"
    }

# ==================== Naive Bayes ====================
print("\n3.Evaluating Naive Bayes models:")

results = []

# Custom Naive Bayes - reconstructing the class
class GaussianNaiveBayes:
    """Custom Gaussian Naive Bayes for loading saved models."""
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}
    
    def _calculate_likelihood(self, x, mean, var):
        eps = 1e-9
        coefficient = 1.0 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        return coefficient * exponent
    
    def _calculate_posterior(self, x):
        posteriors = {}
        for c in self.classes:
            posterior = np.log(self.priors[c])
            likelihood = self._calculate_likelihood(x, self.mean[c], self.var[c])
            posterior += np.sum(np.log(likelihood + 1e-9))
            posteriors[c] = posterior
        return posteriors
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = self._calculate_posterior(x)
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        return np.array(predictions)

try:
    with open('Naive Bayes/naive_bayes_custom.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Reconstruct the model
    custom_nb = GaussianNaiveBayes()
    custom_nb.classes = model_data['classes']
    custom_nb.mean = model_data['mean']
    custom_nb.var = model_data['var']
    custom_nb.priors = model_data['priors']
    
    predictions = custom_nb.predict(test_features_50)
    results.append(evaluate_model(test_labels, predictions, "Custom Naive Bayes"))
    print("Custom Naive Bayes")
except FileNotFoundError:
    print("Naive Bayes/naive_bayes_custom.pkl not found")
except Exception as e:
    print(f"Error loading Custom Naive Bayes: {e}")

# Sklearn Naive Bayes
try:
    with open('Naive Bayes/naive_bayes_sklearn.pkl', 'rb') as f:
        sklearn_nb = pickle.load(f)
    predictions = sklearn_nb.predict(test_features_50)
    results.append(evaluate_model(test_labels, predictions, "Sklearn Naive Bayes"))
    print("Sklearn Naive Bayes")
except FileNotFoundError:
    print("Naive Bayes/naive_bayes_sklearn.pkl not found")
except Exception as e:
    print(f"Error loading Sklearn Naive Bayes: {e}")

# ==================== Decision Trees ====================
print("\n4.Evaluating Decision Tree models:")

# Define Node and DecisionTree classes for loading custom models
class Node:
    """Node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    """Custom Decision Tree classifier for loading saved models."""
    def __init__(self, max_depth=50, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None
        self.n_features = None
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction for single sample."""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

dt_variants = [
    ('Decision Tree/decision_tree_custom_depth5.pkl', 'Custom DT (depth=5)'),
    ('Decision Tree/decision_tree_custom_depth10.pkl', 'Custom DT (depth=10)'),
    ('Decision Tree/decision_tree_custom_depth20.pkl', 'Custom DT (depth=20)'),
    ('Decision Tree/decision_tree_custom_depth30.pkl', 'Custom DT (depth=30)'),
    ('Decision Tree/decision_tree_custom_depth50.pkl', 'Custom DT (depth=50)'),
    ('Decision Tree/decision_tree_sklearn_depth50.pkl', 'Sklearn DT (depth=50)'),
]

for filename, name in dt_variants:
    try:
        with open(filename, 'rb') as f:
            dt_model = pickle.load(f)
        predictions = dt_model.predict(test_features_50)
        results.append(evaluate_model(test_labels, predictions, name))
        print(f"{name}")
    except FileNotFoundError:
        print(f"{filename} not found")
    except Exception as e:
        print(f"Error loading {name}: {e}")

# ==================== MLPs ====================
print("\n5.Evaluating MLP models:")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define MLP architectures 
class MLP_Base(nn.Module):
    def __init__(self):
        super(MLP_Base, self).__init__()
        self.fc1 = nn.Linear(50, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class MLP_Shallow(nn.Module):
    def __init__(self):
        super(MLP_Shallow, self).__init__()
        self.fc1 = nn.Linear(50, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP_Deep(nn.Module):
    def __init__(self):
        super(MLP_Deep, self).__init__()
        self.fc1 = nn.Linear(50, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

class MLP_Small(nn.Module):
    def __init__(self):
        super(MLP_Small, self).__init__()
        self.fc1 = nn.Linear(50, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class MLP_Large(nn.Module):
    def __init__(self):
        super(MLP_Large, self).__init__()
        self.fc1 = nn.Linear(50, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

def evaluate_mlp(model, model_path, model_name):
    """Load and evaluate an MLP model."""
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        X_test = torch.FloatTensor(test_features_50).to(device)
        with torch.no_grad():
            outputs = model(X_test)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
        
        results.append(evaluate_model(test_labels, predictions, model_name))
        print(f"{model_name}")
    except FileNotFoundError:
        print(f"{model_path} not found")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# Evaluate all MLP variants
evaluate_mlp(MLP_Base(), 'MLP/mlp_base.pth', 'Base MLP (3 layers, h=512)')
evaluate_mlp(MLP_Shallow(), 'MLP/mlp_shallow.pth', 'Shallow MLP (2 layers)')
evaluate_mlp(MLP_Deep(), 'MLP/mlp_deep.pth', 'Deep MLP (4 layers)')
evaluate_mlp(MLP_Small(), 'MLP/mlp_small.pth', 'Small MLP (h=256)')
evaluate_mlp(MLP_Large(), 'MLP/mlp_large.pth', 'Large MLP (h=1024)')


# ==================== CNNs (VGG11) ====================
print("\n6.Evaluating VGG11 CNN models:")

# Prepare test images for CNN
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def images_to_tensors(images, transform):
    tensors = []
    for img in images:
        tensor = transform(img)
        tensors.append(tensor)
    return torch.stack(tensors)

X_test_images = images_to_tensors(test_images, transform)

# Define VGG11 architectures
class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU()
        
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(512, 4096)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(4096, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool3(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool4(x)
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

class VGG_Shallow(nn.Module):
    def __init__(self):
        super(VGG_Shallow, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(512 * 2 * 2, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, 10)
    
    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = nn.ReLU()(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = nn.ReLU()(self.bn5(self.conv5(x)))
        x = nn.ReLU()(self.bn6(self.conv6(x)))
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class VGG_LargeKernel(nn.Module):
    def __init__(self):
        super(VGG_LargeKernel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(512, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, 10)
    
    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = nn.ReLU()(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = nn.ReLU()(self.bn5(self.conv5(x)))
        x = nn.ReLU()(self.bn6(self.conv6(x)))
        x = self.pool4(x)
        x = nn.ReLU()(self.bn7(self.conv7(x)))
        x = nn.ReLU()(self.bn8(self.conv8(x)))
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def evaluate_cnn(model, model_path, model_name):
    """Load and evaluate a CNN model."""
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        batch_size = 64
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_test_images), batch_size):
                batch = X_test_images[i:i+batch_size].to(device)
                outputs = model(batch)
                _, predictions = torch.max(outputs, 1)
                all_predictions.extend(predictions.cpu().numpy())
        
        predictions = np.array(all_predictions)
        results.append(evaluate_model(test_labels, predictions, model_name))
        print(f"{model_name}")
    except FileNotFoundError:
        print(f"{model_path} not found")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# Evaluate VGG11 variants (only the 3 that completed successfully)
evaluate_cnn(VGG11(), 'CNN/vgg11_base.pth', 'Base VGG11 (8 conv, 3x3)')
evaluate_cnn(VGG_Shallow(), 'CNN/vgg_shallow.pth', 'Shallow VGG (6 conv)')
evaluate_cnn(VGG_LargeKernel(), 'CNN/vgg_large_kernel.pth', 'VGG Large Kernel (5x5)')

# ==================== Print Results Table ====================
print("\n" + "="*70)
print("Final results - of all models")
print("="*70)

if results:
    # Print header
    print(f"\n{'Model':<40} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 88)
    
    # Print each result
    for result in results:
        print(f"{result['Model']:<40} {result['Accuracy']:<12} {result['Precision']:<12} "
              f"{result['Recall']:<12} {result['F1-Score']:<12}")
    
    print(f"\nEvaluated {len(results)} models")
else:
    print("\nNo models were evaluated. Check that .pkl and .pth files exist in correct folder structure.")
