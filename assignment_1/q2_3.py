import numpy as np
import sys
import matplotlib.pyplot as plt
import csv

if(len(sys.argv)!=4):
    print("Entery data files: ", len(sys.argv), "\n")
    sys.exit(0)

#data
x_t = []
y_t = []
x_e = []
y_e = []
xa = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
yta = []
yea = []

#learn rate
n = 0.000002
l = float(sys.argv[3])

with open(sys.argv[1]) as file:
    data = csv.reader(file)
    for x in data:
        t = x[0:256]
        t = list(map(int, t))
        x_t.append(t)
        y_t.append(int(x[256]))

with open(sys.argv[2]) as file:
    data = csv.reader(file)
    for x in data:
        t = x[0:256]
        t = list(map(int, t))
        x_e.append(t)
        y_e.append(int(x[256]))


ps = 1. / len(x_t)
pss = 1. / len(x_e)

w = [0] * 256
for c in range(10):
    t = [0] * 256
    #iteration
    for x in range(len(x_t)):
        wtx = 0
        for i in range(len(w)):
            wtx = wtx + w[i] * x_t[x][i]
        #print(x, " -- ", wtx, "\n")
        ep_ = ( 1 + np.exp( -1 * wtx) )
        y_ =  1 / ep_
        y_ = y_ - y_t[x]
        for i in range(len(t)):
            t[i] = t[i] + y_ * x_t[x][i] + ps * l * w[i]
    nt = [x*n for x in t]
    for xx in range(len(w)):
        w[xx] = w[xx] - nt[xx]

#train error
t = [0] * 256
for x in range(len(x_t)):
    wtx = 0
    for i in range(len(w)):
        wtx = wtx + w[i] * x_t[x][i]
    ep_ = ( 1 + np.exp( -1 * wtx) )
    y_ =  1 / ep_
    y_ = y_ - y_t[x]
    for i in range(len(t)):
        t[i] = t[i] + y_ * x_t[x][i]
print("train error: ",np.linalg.norm(t), "\n")
#test error
t = [0] * 256
for x in range(len(x_e)):
    wtx = 0
    for i in range(len(w)):
        wtx = wtx + w[i] * x_e[x][i]
    ep_ = ( 1 + np.exp( -1 * wtx) )
    y_ =  1 / ep_
    y_ = y_ - y_e[x]
    for i in range(len(t)):
        t[i] = t[i] + y_ * x_e[x][i]
print("test error: ",np.linalg.norm(t), "\n")
