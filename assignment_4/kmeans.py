import numpy as np
import sys
import matplotlib.pyplot as plt
import csv
import math

def _se(a, b):
        vLen = len(a)
        result = 0
        for i in range(vLen):
                result = result + pow(abs(float(a[i] - b[i])), 2)
        return math.sqrt(result)


def findCenter(clusterSet):
        u = []
        n = len(clusterSet)
        nn = len(clusterSet[0])
        for i in range(nn):
                u.append(0)
        for i in range(n):
                for ii in range(nn):
                        u[ii] = u[ii] + clusterSet[i][ii]
        for i in range(nn):
                u[i] = u[i] / n
        return u


def sse(cluster, center, k):
        sseValue = 0
        for i in range(k):
                subLen = len(cluster[i])
                for ii in range(subLen):
                        sseValue = sseValue + _se(cluster[i][ii], center[i])
        return sseValue

def minimum(data):
        min_value = data[0]
        min_position = 0
        for i in range(len(data)):
                if data[i] < min_value:
                        min_value = data[i]
                        min_position = i
        return int(min_position)




dataSet = []
k = 2
cluster = []
sseArray = []


#read data
with open("p4-data.txt") as file:
        data = csv.reader(file)
        for i in data:
            temp = i
            temp = list(map(int, temp))
            dataSet.append(temp)

print(len(dataSet), " - ", len(dataSet[1]))

#set k value / validate paramater
if len(sys.argv) == 2:
        k = int(sys.argv[1])
else:
        print("Miss K value, exiting...")
        exit(1)
print(k)

#initialize K cluster
#np.random.seed(200)
for i in range(k):
        cluster.append([])
        cluster[i].append(dataSet[np.random.randint(0, 5999)])

print(len(cluster), " - ", len(cluster[0]), " - ", len(cluster[0][0]))

dataSize = len(dataSet)

for i in range(10):
        print(i, flush=True)
        #find center
        center = []
        for ii in range(k):
                center.append(findCenter(cluster[ii]))
        #empty cluster
        cluster = []
        for ii in range(k):
                cluster.append([])
        #assignt into cluster
        for ii in range(dataSize):
                temp = []
                for iii in range(k):
                        temp.append(_se(dataSet[ii], center[iii]))
                cluster[minimum(temp)].append(dataSet[ii])
        sseArray.append(sse(cluster, center, k))

for i in range(10):
        print(sseArray[i])

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(x, sseArray,
            color = "green", s = 10, label = 'sse')


## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 12, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("sse vs iteration")

## function to show plot
plt.show()
