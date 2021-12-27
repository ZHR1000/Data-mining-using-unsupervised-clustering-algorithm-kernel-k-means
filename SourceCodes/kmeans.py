import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import time

# Below code has comments to add or remove input datasets to form different linearly seperated clusters for each run.
# The code can take only .txt because .csv or .tsv type data are not hashable.
# No.of.Iterations vary until final convergence based on the datasets inputted.

filePath1 = "test1_data.txt"
#filePath1 = "test2_data.txt"
#filePath1 =  "led7.tsv"
#filePath2 = "test2_data.txt"
#filePath2 = "led7.tsv"
#filePath1 = "xclara.txt"
#filePath2 = "xclara.csv"
#filePath1 = "quiz_A.data"
#filePath1 = "self_test.data"
dataTesting1 = np.loadtxt(filePath1, delimiter=" ")
#dataTesting2 = np.loadtxt(filePath2, delimiter=" ")
#dataTesting1 = pd.read_csv('xclara.csv')
#dataTesting2 = pd.read_csv('xclara.csv')
#dataTesting1 = pd.read_table('led7.tsv',sep='\t')
#dataTesting2 = pd.read_table('led7.tsv',sep='\t')

print("data testing: ", dataTesting1.shape)

# define params
k = 2  # numb of clusters
iterationCounter = 0  # clustering iteration counter
input = dataTesting1
Centroid_Initialization_Method = "badInitialization" # options: random, kmeans++, badInitialization, zeroInitialization


def initializeCentroid(dataIn, method, k):
    if (method == "random"):
        result = dataIn[np.random.choice(dataIn.shape[0], k, replace=False)]
    if (method == "kmeans++"):
        euclideanMatrix_ofAllCentroid = np.ndarray(shape=(dataIn.shape[0], 0))
        allCentroid = np.ndarray(shape=(0, dataIn.shape[1]))
        first = dataIn[np.random.choice(dataIn.shape[0], 1, replace=False)]
        allCentroid = np.concatenate((allCentroid, first), axis=0)
        repeatedCentroid = np.repeat(first, dataIn.shape[0], axis=0)
        deltaMatrix = abs(np.subtract(dataIn, repeatedCentroid))
        euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))
        indexof_NextCentroid = (np.argmax(np.matrix(euclideanMatrix)))
        if (k > 1):
            for a in range(1, k):
                nextCentroid = np.matrix(dataIn[np.asscalar(indexof_NextCentroid), :])
                allCentroid = np.concatenate((allCentroid, nextCentroid), axis=0)
                for i in range(0, allCentroid.shape[0]):
                    repeatedCentroid = np.repeat(allCentroid[i, :], dataIn.shape[0], axis=0)
                    deltaMatrix = abs(np.subtract(dataIn, repeatedCentroid))
                    euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))
                    euclideanMatrix_ofAllCentroid = \
                        np.concatenate((euclideanMatrix_ofAllCentroid, euclideanMatrix), axis=1)
                euclideanFinal = np.min(np.matrix(euclideanMatrix_ofAllCentroid), axis=1)
                indexof_NextCentroid = np.argmax(np.matrix(euclideanFinal))
        result = allCentroid
    if (method == "badInitialization"):
        allCentroid = np.ndarray(shape=(0, dataIn.shape[1]))
        firstIndex = np.random.randint(0, dataIn.shape[0])
        first = np.matrix(dataIn[firstIndex, :])
        dataIn = np.delete(dataIn, firstIndex, 0)
        allCentroid = np.concatenate((allCentroid, first), axis=0)
        repeatedCentroid = np.repeat(first, dataIn.shape[0], axis=0)
        deltaMatrix = abs(np.subtract(dataIn, repeatedCentroid))
        euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))
        indexof_NextCentroid = (np.argmin(np.matrix(euclideanMatrix)))
        if (k > 1):
            for a in range(1, k):
                nextCentroid = np.matrix(dataIn[np.asscalar(indexof_NextCentroid), :])
                dataIn = np.delete(dataIn, np.asscalar(indexof_NextCentroid), 0)
                euclideanMatrix_ofAllCentroid = np.ndarray(shape=(dataIn.shape[0], 0))
                allCentroid = np.concatenate((allCentroid, nextCentroid), axis=0)
                for i in range(0, allCentroid.shape[0]):
                    repeatedCentroid = np.repeat(allCentroid[i, :], dataIn.shape[0], axis=0)
                    deltaMatrix = abs(np.subtract(dataIn, repeatedCentroid))
                    euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))
                    euclideanMatrix_ofAllCentroid = \
                        np.concatenate((euclideanMatrix_ofAllCentroid, euclideanMatrix), axis=1)
                euclideanFinal = np.min(np.matrix(euclideanMatrix_ofAllCentroid), axis=1)
                indexof_NextCentroid = np.argmin(np.matrix(euclideanFinal))
        result = allCentroid
    if (method == "zeroInitialization"):
        result = np.matrix(np.full((k, dataIn.shape[1]), 0))

    color = iter(cm.rainbow(np.linspace(0, 1, k)))
    plt.figure("centroid initialization")
    plt.title("centroid initialization")
    plt.scatter(dataIn[:, 0], dataIn[:, 1], marker=".", s=100)
    for i in range(0, k):
        col = next(color)
        plt.scatter((result[i, 0]), (result[i, 1]), marker="*", s=400, c=col)
        plt.text((result[i, 0]), (result[i, 1]), str(i + 1), fontsize=20)
    return result


def plotClusterResult(listof_ClusterMembers, centroid, iteration, converged):
    n = listof_ClusterMembers.__len__()
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    plt.figure("result")
    plt.clf()
    plt.title("iteration-" + iteration)
    for i in range(n):
        col = next(color)
        memberof_Cluster = np.asmatrix(listof_ClusterMembers[i])
        plt.scatter(np.ravel(memberof_Cluster[:, 0]), np.ravel(memberof_Cluster[:, 1]), marker=".", s=100, c=col)
        plt.scatter((centroid[i, 0]), (centroid[i, 1]), marker="*", s=400, c=col, edgecolors="black")
    if (converged == 0):
        plt.ion()
        plt.show()
        plt.pause(0.1)
    if (converged == 1):
        plt.show(block=True)


def kMeans(data, centroidInitialized):
    nCluster = centroidInitialized.shape[0]
    # looping until converged
    global iterationCounter
    centroidInitialized = np.matrix(centroidInitialized)
    while (True):
        iterationCounter += 1
        euclideanMatrixOf_AllCluster = np.ndarray(shape=(data.shape[0], 0))
        # assign data to cluster whose centroid is the closest one
        for i in range(0, nCluster):
            centroidRepeated = np.repeat(centroidInitialized[i, :], data.shape[0], axis=0)
            deltaMatrix = abs(np.subtract(data, centroidRepeated))
            euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))
            euclideanMatrixOf_AllCluster = \
                np.concatenate((euclideanMatrixOf_AllCluster, euclideanMatrix), axis=1)
        clusterMatrix = np.ravel(np.argmin(np.matrix(euclideanMatrixOf_AllCluster), axis=1))
        listof_ClusterMember = [[] for i in range(k)]
        for i in range(0, data.shape[0]):  # assign data to cluster regarding cluster matrix
            listof_ClusterMember[np.asscalar(clusterMatrix[i])].append(data[i, :])
        # calculate new centroid
        newCentroid = np.ndarray(shape=(0, centroidInitialized.shape[1]))
        for i in range(0, nCluster):
            memberof_Cluster = np.asmatrix(listof_ClusterMember[i])
            centroidOf_Cluster = memberof_Cluster.mean(axis=0)
            newCentroid = np.concatenate((newCentroid, centroidOf_Cluster), axis=0)
        # break when converged
        print("iter: ", iterationCounter)
        print("centroid: ", newCentroid)
        if ((centroidInitialized == newCentroid).all()):
            break
        # update new centroid
        centroidInitialized = newCentroid
        plotClusterResult(listof_ClusterMember, centroidInitialized, str(iterationCounter), 0)
        time.sleep(1)
    return listof_ClusterMember, centroidInitialized


centroidInitialized = initializeCentroid(input, Centroid_Initialization_Method, k)
clusterResults, centroid = kMeans(input, centroidInitialized)
plotClusterResult(clusterResults, centroid, str(iterationCounter) + " (converged)", 1)
