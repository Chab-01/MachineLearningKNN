import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

data = pd.read_csv("microchips.csv", header=None).values
dataWithoutLabels = data[:, :2]  # First two columns, used as training data
labels = data[:, 2]  # Third column with the labels
trainingSet = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])
x = data[:, 0]
y = data[:, 1]
kval = [1, 3, 5, 7]

# 1.PLOTTING OK AND FAIL USING DIFFERENT MARKERS
def plotFirst():
    plt.figure(1)
    plt.title("Original microship data")
    colors = ['green' if i == 1 else 'red' for i in labels]
    plt.scatter(x, y, s=12, c=['green' if i == 1 else 'red' for i in labels])

#Distance formula
def euclideanDistance(x1, y1, x2, y2):
    distance = np.sqrt(((x2-x1)**2) + ((y2-y1)**2))
    return distance

def knn(data, trainingSet, k):
    distances = []

    for i in range(len(data)):
        distances.append(euclideanDistance(trainingSet[0], trainingSet[1], data[i][0], data[i][1])) # Calculate distances between csv file data and the test data given in assignment
    sortedDistances = sorted(distances)[:k] # Gets the values up to the kth number (if k=5 get the 5 nearest neighbors)
    indices = [distances.index(i) for i in sortedDistances] # Creates a list with the indices of the original values
    kNearestLabel = [data[j][2] for j in indices] # Creates a list that fetches 1 or 0 from the third column
    mostCommonLabel = Counter(kNearestLabel).most_common()[0][0] # Stores the most common label in a variable.

    return mostCommonLabel


def trainingError(data, dataWithoutLabels, k):
    errors = 0
    pred = []

    for j in range(len(dataWithoutLabels)):
        pred.append(knn(data, dataWithoutLabels[j], k)) #We perform knn on data and datawithoutlabels for all k
    for i in range(len(labels)):
        if (labels[i] != pred[i]): #We check if the predicted values are the same as the true label values
            errors += 1

    return errors

for k in kval:
    print(f"k = {k}") #Print what k value we are on
    for i in range(len(trainingSet)):
        prediction = knn(data, trainingSet[i], k) #Perform knn on data and the trainingset from assignment for all k values
        if (prediction == 0): #We check if the label is either 1 or 0 thus giving OK or FAIL
            print(f"chip{i+1}: {trainingSet[i]} ==> FAIL")
        else:
            print(f"chip{i+1}: {trainingSet[i]} ==> OK")

def plotDecisionBoundary():
    plt.figure(2)
    h = 0.1
    xMin, xMax = x.min() - h, x.max() + h
    yMin, yMax = y.min() - h, y.max() + h
    xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.04), np.arange(yMin, yMax, 0.04))
    xyMesh = np.c_[xx.ravel(), yy.ravel()] #Grid

    for i,k in enumerate(kval): #We use enumerate so i can get the index values and k the k values. We then use i to increment the subplot and k for operations
        plt.subplot(221+i) 

        errors = trainingError(data, dataWithoutLabels, k) #Getting errors for each k value to add into plots
        plt.title(f"K= {k} Training errors= {errors}") #Setting title of plots
        lst = []
        for s in xyMesh:  #iterates over the grid (and sets color accordingly line 78-79)
            res = knn(data, s, k)
            lst.append([res])
        lstArray = np.array(lst)
        plt.pcolormesh(xx, yy, lstArray.reshape(xx.shape), cmap="Spectral")
        plt.scatter(x, y, c=labels, marker=".", cmap="prism")


plotFirst()
plotDecisionBoundary()
plt.show()


