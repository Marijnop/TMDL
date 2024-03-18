import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models  
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
plt.hist(train_y, bins=10, rwidth=0.9)
plt.xticks(0.9*np.arange(10)+0.45, range(10))
plt.ylabel("Frequency")
plt.xlabel("Class Label")
plt.show()

# preprocessing?

# method for accuracy calculation
def get_accuracies(model, train_X, train_y, test_X, test_y):
    print("Training score:", model.score(train_X, train_y))
    print("Testing score: ", model.score(test_X, test_y))

# Logistic Regression
modelLR = LogisticRegression()
modelLR.fit(train_X, train_y)
y_pred = modelLR.predict(test_X)
get_accuracies(modelLR, train_X, train_y, test_X, test_y)
print(classification_report(test_y, y_pred))

# Decision Tree
modelDT = tree.DecisionTreeClassifier(max_depth=15)
modelDT = modelDT.fit(train_X, train_y)
print("Decision Tree")
get_accuracies(modelDT, train_X, train_y, test_X, test_y)

# Random Forest
modelRFC = RFC(100, max_depth=10)
modelRFC = modelRFC.fit(train_X, train_y)
print("Random Forest")
get_accuracies(modelRFC, train_X, train_y, test_X, test_y)

# SVM
modelSVC = SVC(kernel="rbf", C=1) # change these values, idk yet what is right here
modelSVC = modelSVC.fit(train_X, train_y)
print("SVM")
get_accuracies(modelSVC, train_X, train_y, test_X, test_y)