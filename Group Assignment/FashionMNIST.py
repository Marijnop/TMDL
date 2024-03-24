import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models  
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC


(train_X , train_y), (test_X, test_y) = datasets.fashion_mnist.load_data()

# data is 28x28 pixels -> 784 pixels total, all with one pixel value (integer value between 0 and 255)
# training and test datasets have 785 columns. The first column is for class labels (0 to 9).

# list with all 10 items
class_labels = {0:"T-shirt/top",
           1:"Trouser",
           2:"Pullover",
           3:"Dress",
           4:"Coat",
           5:"Sandal",
           6:"Shirt",
           7:"Sneaker",
           8:"Bag",
           9:"Ankle boot"}

n_train = 60000
n_test = 10000

def image_show(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(class_labels[y[index]])

# prints each items once (for introduction)
printed_items = set()
for index in range(len(train_y)):
    if train_y[index] not in printed_items:
        image_show(train_X, train_y, index)
        printed_items.add(train_y[index])
    if len(printed_items) == 10:
        break

# validating the data (balanced dataset and show an example of each (for introduction)
# Number of bins should be equal to the number of classes
num_classes = 10
bins = np.arange(num_classes + 1) - 0.5  # to center bins on integers
tick_marks = np.arange(num_classes)  # ticks should be from 0 to 9

# Plotting the histogram
plt.figure(figsize=(10, 5))  # Optional: Adjust the size to fit the class labels
plt.hist(train_y, bins=bins, rwidth=0.8)
plt.xticks(tick_marks, [class_labels[i] for i in tick_marks])
plt.ylabel("Frequency")
plt.xlabel("Class Label")
plt.show()

# preprocessing
# Flatten the images from 28x28 to 784 pixels for train and test sets
train_X_flat = train_X.reshape((train_X.shape[0], -1))
test_X_flat = test_X.reshape((test_X.shape[0], -1))

print("New shape of train_X:", train_X_flat.shape)
print("New shape of test_X:", test_X_flat.shape)

# Initialize the models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTree1': tree.DecisionTreeClassifier(max_depth=15),
    'RandomForest': RFC(n_estimators=100, max_depth=10),
    'SVM': SVC(kernel="rbf", C=1)
}

# Train and evaluate each model
for name, model in models.items():
    # Training
    start_time = time.time()
    model.fit(train_X_flat, train_y)
    training_time = time.time() - start_time
    
    # Predictions
    train_predictions = model.predict(train_X_flat)
    test_predictions = model.predict(test_X_flat)
    
    # Evaluation
    train_accuracy = accuracy_score(train_y, train_predictions)
    test_accuracy = accuracy_score(test_y, test_predictions)
    
    # Output results
    print(f'{name} Model:')
    print(f'Train Accuracy: {train_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Training Time: {training_time} seconds\n')
    print(f'Classification Report for {name}:')
    print(classification_report(test_y, test_predictions))
    print('-' * 40)
