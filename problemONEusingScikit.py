from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#This code is essentially the same as Microchips.py. Just that the knn is not self implemented and used from library instead

data = pd.read_csv("microchips.csv", header=None).values
x = data[:, :-1]
y = data[:, -1]
labels = data[:, 2]
dataWithoutLabels = data[:, :2] 
trainingSet = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])
kval = [1, 3, 5, 7]

#Plotting the original microchip data
def plotFirst():
    plt.figure(1)
    plt.title("Original microship data")
    colors = ['green' if i == 1 else 'red' for i in labels]
    plt.scatter(x[:,0], x[:,1], s=12, c=colors)

#Getting the predicted labels
for k in kval:
  classifier = KNeighborsClassifier(n_neighbors=k)
  classifier.fit(x,y)
  print(f"k = {k}")
  for i in range(len(trainingSet)):
    prediction = classifier.predict([trainingSet[i]])
    if (prediction[0] == 0):
        print(f"chip{i+1}: {trainingSet[i]} ==> FAIL")
    else:
        print(f"chip{i+1}: {trainingSet[i]} ==> OK")

#Calculates training errors
def trainingError(data, dataWithoutLabels, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data, labels)
    pred = knn.predict(data)

    errors = 0
    for i in range(len(labels)):
        if labels[i] != pred[i]:
            errors += 1

    return errors

#Plotting decision boundary
def plotDecisionBoundary():
    plt.figure(2)
    h = 0.1
    xMin, xMax = data[:, 0].min() - h, data[:, 0].max() + h
    yMin, yMax = data[:, 1].min() - h, data[:, 1].max() + h
    xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.04), np.arange(yMin, yMax, 0.04))
    xyMesh = np.c_[xx.ravel(), yy.ravel()]

    for i, k in enumerate(kval):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(dataWithoutLabels, labels)

        plt.subplot(221+i)

        errors = trainingError(dataWithoutLabels, labels, k)
        plt.title(f"K= {k} Training errors= {errors}")
        
        Z = knn.predict(xyMesh)
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap="Spectral")
        plt.scatter(data[:, 0], data[:, 1], c=labels, marker=".", cmap="prism")



plotFirst()
plotDecisionBoundary()
plt.show()