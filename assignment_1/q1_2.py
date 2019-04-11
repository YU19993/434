import numpy as np
import sys
import matplotlib.pyplot as plt

if(len(sys.argv)!=3):
    print("Entery data files\n")
    sys.exit(0)

#inport dataset from file
dataset = np.loadtxt(sys.argv[1])
datasett = np.loadtxt(sys.argv[2])
x_t = []
y_t = []
x_e = []
y_e = []

#parse dataset into feature/target
#train data
for i in dataset:
    t = [1]
    for ii in range(13):
        t.append(i[ii])
    x_t.append(t)
    y_t.append(i[13])

#test data
for i in datasett:
    t = [1]
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


print(w, "\n")

#comput ASE for train/test data
s = np.matmul(x_t, w)
t = []
for i in s:
    t.append(i)

ase_t = 0.
for i in range(len(y_t)):
    ase_t = ase_t + (y_t[i] - t[i]) * (y_t[i] - t[i])
ase_t = ase_t / len(y_t)

print(ase_t, "\n")

s = np.matmul(x_e, w)
t = []
for i in s:
    t.append(i)

ase_e = 0.
for i in range(len(y_e)):
    ase_e = ase_e + (y_e[i] - t[i]) * (y_e[i] - t[i])
ase_e = ase_e / len(y_e)

print(ase_e, "\n")
