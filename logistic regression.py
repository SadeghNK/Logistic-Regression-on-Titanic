# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 07:00:16 2018

@author: sadegh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
import itertools

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.close('all')

###############################################################################
# A: just plotting SSE and CE loss funtions for comparison in range [0, 1],
# assuming real labels are all 0

def cross_ent(x, y):
    ce = -(y * np.log(x) + (1-y) * np.log(1-x))
    return ce

def sse(x, y):
    sse = np.power(x - y, 2)
    return sse

y_hat = np.linspace(0.001, 0.999, 100)
y = np.zeros_like(y_hat)
plt.figure()
plt.plot(y_hat, sse(y_hat, y))
plt.plot(y_hat, cross_ent(y_hat, y))
plt.grid(True)
plt.legend(labels=["SSE", "Cross Entropy"], loc=2,
           fontsize='large', fancybox=True)
plt.xlabel("probability")
plt.ylabel("Loss")
del y_hat, y

###############################################################################
# Logistic regression on titanic:

def input():
    df = pd.read_csv(filepath_or_buffer='Titanic.csv',
                     usecols=[1, 2, 4, 5])
    df["Age"] = df["Age"].fillna(0)
    
    # Gender to Boolean
    df.Sex = pd.Series(np.where(df.Sex.values == 'male', 1, 0), df.index)
    
    # Pclass to one-hot-encoding
    Pclass = pd.get_dummies(df.Pclass).values
    
    X = np.concatenate((Pclass, df.Sex.values[:, np.newaxis],
                        df.Age.values[:, np.newaxis]), axis=1)
    y = df.iloc[:, 0].values[:, np.newaxis].astype(np.float64)
    
    return X, y


X, y = input()

merged = np.concatenate([X, y], axis=1)
np.random.shuffle(merged)
X, y = np.split(merged, [5], axis=1)
n = len(y)
thresh = int(n * 0.8)
X_train = X[0:thresh, :]
y_train = y[0:thresh, :]
X_test = X[thresh:, :]
y_test = y[thresh:, :]
del X, y, merged


def initializer():
    global W, b
    np.random.seed(8)
    W = np.random.randn(X_train.shape[1], 1)
    b = np.random.randn(1)


def inference(X):
    lin_reg = X @ W + b
    probs = 1 / (1 + np.exp(-lin_reg)) # Sigmoid fucntion
    return probs


def loss(probs, y, epsilon=1e-12):
    probs = np.clip(probs, epsilon, 1.0 - epsilon)
    ce = -np.sum(y * np.log(probs + 1e-9))
    return ce


def gradient(X, y):
    probs = inference(X)
    Dce_Dw = X.T @ (probs - y)
    Dce_Db = np.sum(probs - y)
    
    return Dce_Dw, Dce_Db


def plot_confusion_matrix(cm, classes=["Survived", "Dead"],
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


@np.vectorize
def evaluate(X):
    if X >= 0.5:
       return 1
    else:
        return 0


def train(X, y, batch_size, epoch, lr):
    
    initializer()
    global W, b
    n = len(y)
    accuracy = np.sum(evaluate(inference(X)) == y) / n
    steps = n // batch_size
    if n % batch_size != 0:
        steps += 1
    X_steps = np.array_split(X, steps)
    y_steps = np.array_split(y, steps)
    loss_per_epoch = np.array([])
    
    print("\nBatch size:", batch_size)
    
    for i in range(epoch):
        
        for s in range(steps):
            d_dw, d_db = gradient(np.asarray(X_steps[s]),
                                  np.asarray(y_steps[s]))
            W -= lr * d_dw
            b -= lr * d_db
        
        accuracy = np.sum(evaluate(inference(X)) == y) / n
        loss_per_epoch = np.append(loss_per_epoch, loss(inference(X), y))
        print("Epoch: %d" % (i+1) + "\tAccuracy: %.4f" % accuracy)
    
    return loss_per_epoch, epoch


plt.figure()
loss_per_batch = list()
total_epochs = list()
epoch = 100
lr = 1e-2
train_precisions = np.zeros([4])
train_accuracies = np.zeros([4])
test_precisions = np.zeros([4])
test_accuracies = np.zeros([4])

for i, batch_size in enumerate([1, 5, 10, 100]):
    
    loss_per_epoch, epoch  = train(X_train, y_train, batch_size, epoch, lr)
    loss_per_batch.append(loss_per_epoch)
    total_epochs.append(epoch)
    
    y_hat_train = evaluate(inference(X_train))
    train_precisions[i] = precision_score(y_train, y_hat_train)
    train_accuracies[i] = accuracy_score(y_train, y_hat_train)
    
    y_hat_test = evaluate(inference(X_test))
    test_precisions[i] = precision_score(y_test, y_hat_test)
    test_accuracies[i] = accuracy_score(y_test, y_hat_test)
    conf_mat = confusion_matrix(y_test, y_hat_test)
    plt.subplot(2, 2, i+1)
    title = "Batch size = " + str(batch_size)
    plot_confusion_matrix(conf_mat, title=title)
    plt.pause(0.15)


print("-" * 60)
print("Summary of Training for %d" % epoch + " epochs:")
print("Number of samples: %d" %len(y_train))
for i, batch_size in enumerate([1, 5, 10, 100]):
    print("Batch Size: %d" % batch_size,
          "  \tAccuracy: %0.4f" % train_accuracies[i],
          "  \tPrecision: %0.4f" % train_precisions[i])

print("-" * 60)
print("Summary of Test:")
print("Number of samples: %d" %len(y_test))
for i, batch_size in enumerate([1, 5, 10, 100]):
    print("Batch Size: %d" % batch_size,
          "  \tAccuracy: %0.4f" % test_accuracies[i],
          "  \tPrecision: %0.4f" % test_precisions[i])
