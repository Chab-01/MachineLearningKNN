import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("polynomial200.csv", header=None).values
trainingSet = data[:100, :] # take the first 100 elements in both columns
testSet = data[100:, :] # take the remaining elements from both columns
kval = [1,3,5,7,9,11]

def plotTrainingAndTestSet():
  plt.figure(1)
  for i in range(2):
    plt.subplot(121+i)
    if (i == 0):
      plt.title("Training set")
      plt.scatter(trainingSet[:,0], trainingSet[:,1], s=5)
    else:
      plt.title("Test set")
      plt.scatter(testSet[:,0], testSet[:,1], s=5)

#Distance formula
def euclideanDistance(x1, x2):
  distance = np.sqrt(np.sum((x1 - x2) ** 2))
  return distance

def knn(data, x, k):
    distances = []

    for i in range(len(data)):
      distances.append(euclideanDistance(x, data[i][0])) #Calculate the distance between x and each point in data 

    sortedDistances = sorted(distances)[:k] #Sort the distances 
    indices = [distances.index(i) for i in sortedDistances] #Get the indices of the k nearest neighbors in data
    prediction = sum(data[i][1] for i in indices) / k #Calculate the mean of the k nearest neighbors

    return prediction

  
def plotRegression():
  plt.figure(2)
  print("MSE TEST ERRORS:")
  for i,k in enumerate(kval): #Enumerate used so i can increment subplot and k can take on k values
    plt.subplot(231+i)

    #Predictions for yTrain
    yPredTrain = []
    for x in trainingSet[:,0]:
      yPredTrain.append(knn(trainingSet, x, k))
    yPredTrain = np.array(yPredTrain)

    #Predictions for yTest
    yPredTest = []
    for x in testSet[:,0]:
      yPredTest.append(knn(testSet, x, k))
    yPredTest = np.array(yPredTest)

    trainingMse = np.mean( (yPredTrain - trainingSet[:,1]) **2 ) #MSE for training set
    testMse = np.mean( (yPredTest - testSet[:,1]) **2 ) #MSE for test set

    plt.title(f"K= {k} TrainingMSE= {trainingMse:.2f}")
    plt.plot(trainingSet[:, 0], trainingSet[:, 1], 'bo', markersize=2)

    # plot regression line
    xvals = np.linspace(min(trainingSet[:,0]), max(trainingSet[:,0]), num=150)
    yPred = [knn(trainingSet, x, k) for x in xvals]
    plt.plot(xvals, yPred, 'c')

    print(f"k = {k} ==> MSE = {testMse:.2f}")

#5. WHICH K GIVES THE BEST REGRESSION?
#ANSWER: I believe k=7 gives the best regression. If you compare and subtract the training MSE and the test MSE you see that
#        the difference between train and test MSE for k=7 has the lowest difference of them all which means that they are most similar. 
#        They dont differ to much which gives me the reason to believe that k=7 is the best value for regression

plotTrainingAndTestSet()
plotRegression()
plt.show()