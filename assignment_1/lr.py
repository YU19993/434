import numpy as np
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt

#inport dataset from file
dataset = np.loadtxt('housing_train.txt')
datasett = np.loadtxt('housing_test.txt')
x_t = []
y_t = []
x_e = []
y_e = []

print(len(dataset))

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

reg = linear_model.LinearRegression()
reg.fit(x_t, y_t)
w = []
t = reg.coef_
for i in t:
    w.append(i)

print('Coefficients: \n', w, "\n")

y_t_c = np.matmul(x_t, w)
t = []
for i in y_t_c:
    t.append(i)
#print(t, "\n")

ase_t = 0.
for i in range(len(y_t)):
    ase_t = ase_t + (y_t[i] - y_t_c[i]) * (y_t[i] - y_t_c[i])
ase_t = ase_t / len(y_t)

print(ase_t, "\n")

y_e_c = np.matmul(x_e, w)

ase_e = 0.
for i in range(len(y_e)):
    ase_e = ase_e + (y_e[i] - y_e_c[i]) * (y_e[i] - y_e_c[i])
ase_e = ase_e / len(y_e)

print(ase_e, "\n")


#transpose x_t
x_t_t = [[x_t[j][i] for j in range(len(x_t))] for i in range(len(x_t[0]))]

x_m = np.matmul(x_t_t, x_t)
xy = np.matmul(x_t_t, y_t)
xv = np.linalg.inv(x_m)
w = np.matmul(xv, xy)

print(w)

s = np.matmul(x_t, w)
t = []
for i in s:
    t.append(i)
print(t, "\n")

ase_t = 0.
for i in range(len(y_t)):
    ase_t = ase_t + (y_t[i] - t[i]) * (y_t[i] - t[i])
ase_t = ase_t / len(y_t)

print(ase_t, "\n")

s = np.matmul(x_e, w)
t = []
for i in s:
    t.append(i)
#print(t, "\n")

ase_e = 0.
for i in range(len(y_e)):
    ase_e = ase_e + (y_e[i] - t[i]) * (y_e[i] - t[i])
ase_e = ase_e / len(y_e)

print(ase_e, "\n")
