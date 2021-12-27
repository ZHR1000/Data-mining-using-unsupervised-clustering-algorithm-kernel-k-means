import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import time

#Below code has comments to add or remove input dataset to form different non-linear clusters for each run.
#The code can take only .txt dataset because .csv or .tsv type data are not hashable.
#No.of.Iterations vary until final convergence based on the datasets inputted.

#filePath1 = "test1_data.txt"
filePath2 = "test1_data.txt"
#filePath2 = "test2_data.txt"
#filePath2 = "quiz_A.data"
#filePath2 = "self_test.data" 
#filePath2 = "xclara.txt"
#filePath2 = "xclara.csv"                                                              
#dataTesting1 = np.loadtxt(filePath1, delimiter=" ")
dataTesting2 = np.loadtxt(filePath2, delimiter=" ")    
#dataTesting1 = pd.read_csv('xclara.csv')
#dataTesting2 = pd.read_csv('xclara.csv')

#params
k = 2 #number of cluster
var = 5 #var in RFB kernel
iterationCounter = 0
input = dataTesting2
Centroid_initializationMethod = "byOriginDistance" #options = random, byCenterDistance, byOriginDistance

def initializeCluster(dataInput, nCluster, method):
    listOf_ClusterMember = [[] for i in range(nCluster)]
    if (method == "random"):
        shuffledDataIn = dataInput
        np.random.shuffle(shuffledDataIn)
        for i in range(0, dataInput.shape[0]):
            listOf_ClusterMember[i%nCluster].append(dataInput[i,:])
    if (method == "byCenterDistance"):
        center = np.matrix(np.mean(dataInput, axis=0))
        repeatedCentroid = np.repeat(center, dataInput.shape[0], axis=0)
        deltaMatrix = abs(np.subtract(dataInput, repeatedCentroid))
        euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))
        dataNew = np.array(np.concatenate((euclideanMatrix, dataInput), axis=1))
        dataNew = dataNew[np.argsort(dataNew[:, 0])]
        dataNew = np.delete(dataNew, 0, 1)
        divider = dataInput.shape[0]/nCluster
        for i in range(0, dataInput.shape[0]):
            listOf_ClusterMember[np.int(np.floor(i/divider))].append(dataNew[i,:])
    if (method == "byOriginDistance"):
        origin = np.matrix([[0,0]])
        repeatedCentroid = np.repeat(origin, dataInput.shape[0], axis=0)
        deltaMatrix = abs(np.subtract(dataInput, repeatedCentroid))
        euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))
        dataNew = np.array(np.concatenate((euclideanMatrix, dataInput), axis=1))
        dataNew = dataNew[np.argsort(dataNew[:, 0])]
        dataNew = np.delete(dataNew, 0, 1)
        divider = dataInput.shape[0]/nCluster
        for i in range(0, dataInput.shape[0]):
            listOf_ClusterMember[np.int(np.floor(i/divider))].append(dataNew[i,:])

    return listOf_ClusterMember

def RbfKernel(data1, data2, sigma):
    delta =abs(np.subtract(data1, data2))
    squaredEuclidean = (np.square(delta).sum(axis=1))
    result = np.exp(-(squaredEuclidean)/(2*sigma**2))
    return result

def thirdTerm(memberOf_Cluster):
    result = 0
    for i in range(0, memberOf_Cluster.shape[0]):
        for j in range(0, memberOf_Cluster.shape[0]):
            result = result + RbfKernel(memberOf_Cluster[i, :], memberOf_Cluster[j, :], var)
    result = result / (memberOf_Cluster.shape[0] ** 2)
    return result

def secondTerm(dataI, memberOf_Cluster):
    result = 0
    for i in range(0, memberOf_Cluster.shape[0]):
        result = result + RbfKernel(dataI, memberOf_Cluster[i,:], var)
    result = 2 * result / memberOf_Cluster.shape[0]
    return result

def plotResult(listOf_ClusterMembers, centroid, iteration, converged):
    n = listOf_ClusterMembers.__len__()
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    plt.figure("result")
    plt.clf()
    plt.title("iteration-" + iteration)
    for i in range(n):
        col = next(color)
        memberOf_Cluster = np.asmatrix(listOf_ClusterMembers[i])
        plt.scatter(np.ravel(memberOf_Cluster[:, 0]), np.ravel(memberOf_Cluster[:, 1]), marker=".", s=100, c=col)
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    for i in range(n):
        col = next(color)
        plt.scatter(np.ravel(centroid[i, 0]), np.ravel(centroid[i, 1]), marker="*", s=400, c=col, edgecolors="black")
    if (converged == 0):
        plt.ion()
        plt.show()
        plt.pause(0.1)
    if (converged == 1):
        plt.show(block=True)

def kMeansKernel(data, Centroid_initializationMethod):
    global iterationCounter
    memberInitilized_toCluster = initializeCluster(data, k, Centroid_initializationMethod)
    nCluster = memberInitilized_toCluster.__len__()
    #looping until converged
    while(True):
        # calculate centroid, only for visualization purpose
        centroid = np.ndarray(shape=(0, data.shape[1]))
        for i in range(0, nCluster):
            memberOf_Cluster = np.asmatrix(memberInitilized_toCluster[i])
            centroidof_Cluster = memberOf_Cluster.mean(axis=0)
            centroid = np.concatenate((centroid, centroidof_Cluster), axis=0)
        #plot result in every iteration
        plotResult(memberInitilized_toCluster, centroid, str(iterationCounter), 0)
        oldTime = np.around(time.time(), decimals=0)
        kernelResultClusterOf_AllCluster = np.ndarray(shape=(data.shape[0], 0))
        #assign data to cluster whose centroid is the closest one
        for i in range(0, nCluster):#repeat for all cluster
            term3 = thirdTerm(np.asmatrix(memberInitilized_toCluster[i]))
            matrixTerm3 = np.repeat(term3, data.shape[0], axis=0); matrixTerm3 = np.asmatrix(matrixTerm3)
            matrixTerm2 = np.ndarray(shape=(0,1))
            for j in range(0, data.shape[0]): #repeat for all data
                term2 = secondTerm(data[j,:], np.asmatrix(memberInitilized_toCluster[i]))
                matrixTerm2 = np.concatenate((matrixTerm2, term2), axis=0)
            matrixTerm2 = np.asmatrix(matrixTerm2)
            kernelResultClusterI = np.add(-1*matrixTerm2, matrixTerm3)
            kernelResultClusterOf_AllCluster =\
                np.concatenate((kernelResultClusterOf_AllCluster, kernelResultClusterI), axis=1)
        clusterMatrix = np.ravel(np.argmin(np.matrix(kernelResultClusterOf_AllCluster), axis=1))
        listOf_ClusterMember = [[] for l in range(k)]
        for i in range(0, data.shape[0]):#assign data to cluster regarding cluster matrix
            listOf_ClusterMember[np.asscalar(clusterMatrix[i])].append(data[i,:])
        for i in range(0, nCluster):
            print("Cluster member numbers-", i, ": ", listOf_ClusterMember[0].__len__())
        #break when converged
        boolAcc = True
        for m in range(0, nCluster):
            prev = np.asmatrix(memberInitilized_toCluster[m])
            current = np.asmatrix(listOf_ClusterMember[m])
            if (prev.shape[0] != current.shape[0]):
                boolAcc = False
                break
            if (prev.shape[0] == current.shape[0]):
                boolPerCluster = (prev == current).all()
            boolAcc = boolAcc and boolPerCluster
            if(boolAcc==False):
                break
        if(boolAcc==True):
            break
        iterationCounter += 1
        #update new cluster member
        memberInitilized_toCluster = listOf_ClusterMember
        newTime = np.around(time.time(), decimals=0)
        print("iteration-", iterationCounter, ": ", newTime - oldTime, " seconds")
    return listOf_ClusterMember, centroid

clusterResult, centroid = kMeansKernel(input, Centroid_initializationMethod)
plotResult(clusterResult, centroid, str(iterationCounter) + ' (converged)', 1)
print("converged!")
