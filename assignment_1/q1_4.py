import numpy as np
import sys
import matplotlib.pyplot as plt
import random as rd

if(len(sys.argv)!=3):
    print("Entery data files\n")
    sys.exit(0)

#inport dataset from file
dataset = np.loadtxt(sys.argv[1])
datasett = np.loadtxt(sys.argv[2])

ase_y_t = []
ase_y_e = []

d = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
for x in d:
    x_t = []
    y_t = []
    x_e = []
    y_e = []
    #parse dataset into feature/target
    #train data
    for i in dataset:
        t = []
        for ii in range(x):
            t.append(rd.uniform(0.0, 10.5))
        for ii in range(13):
            t.append(i[ii])
        x_t.append(t)
        y_t.append(i[13])

    #test data
    for i in datasett:
        t = []
        for ii in range(x):
            t.append(rd.uniform(0.0, 10.5))
        for ii in range(13):
            t.append(i[ii])
        x_e.append(t)
        y_e.append(i[13])

    #transpose x_t
    x_t_t = [[x_t[j][i] for j in range(len(x_t))] for i in range(len(x_t[0]))]
    #comput w
    x_m = np.matmul(x_t_t, x_t)
    xy = np.matmul(x_t_t, y_t)
    xv = np.linalg.inv(x_m)
    w = np.matmul(xv, xy)

    #comput ASE for train/test data
    s = np.matmul(x_t, w)
    t = []
    for i in s:
        t.append(i)

    ase_t = 0.
    for i in range(len(y_t)):
        ase_t = ase_t + (y_t[i] - t[i]) * (y_t[i] - t[i])
    ase_t = ase_t / len(y_t)

    ase_y_t.append(ase_t)

    s = np.matmul(x_e, w)
    tt = []
    for i in s:
        tt.append(i)

    ase_e = 0.
    for i in range(len(y_e)):
        ase_e = ase_e + (y_e[i] - tt[i]) * (y_e[i] - tt[i])
    ase_e = ase_e / len(y_e)

    ase_y_e.append(ase_e)


## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(d, ase_y_t,
            color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(d, ase_y_e,
            color = "blue", s = 6, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 45, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("ASE vs D")

## function to show plot
plt.show()
