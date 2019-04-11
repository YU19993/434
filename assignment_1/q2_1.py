import numpy as np
import sys
import matplotlib.pyplot as plt
import csv

if(len(sys.argv)!=4):
    print("Entery data files: ", len(sys.argv), "\n")
    sys.exit(0)

x_t = []
y_t = []
x_e = []
y_e = []

with open(sys.argv[1]) as file:
    data = csv.reader(file)
    for x in data:
        x_t.append(x[0:256])
        y_t.append(x[256])

with open(sys.argv[2]) as file:
    data = csv.reader(file)
    for x in data:
        x_e.append(x[0:256])
        y_e.append(x[256])


w = [0] * 256

t = [0] * 256
for x in range(256):
    w_t =[[w[j][i] for j in range(len(w))] for i in range(len(w[0]))]
    wtx = np.matmul(w_t), x_t[x])
    y_ =  1 / 1+exp(-1 * wtx)
    t_ = (y_ - y_t[x]) * x_t[x]
    if(|t_| < e):
        break
    t = t + t_
w = w - nt
