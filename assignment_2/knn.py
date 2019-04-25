import numpy as np
import sys
import matplotlib.pyplot as plt
import csv
import math

#train data
y = []
x = []
k = int(sys.argv[3])
tx = []
ty = []

with open(sys.argv[2]) as file:
        data = csv.reader(file)
        for i in data:
            x_ = i[1:31]
            x_ = list(map(float, x_))
            t = 0
            for ii in range(len(x_)):
                    t = x_[ii] * x_[ii]
            norm = math.sqrt(t)
            for ii in range(len(x_)):
                    x_[ii] = x_[ii] / norm
            tx.append(x_)
            if int(i[0]) == -1:
                    ty.append(0)
            else:
                    ty.append(int(i[0]))


with open(sys.argv[1]) as file:
        data = csv.reader(file)
        for i in data:
            x_ = i[1:31]
            x_ = list(map(float, x_))
            t = 0
            for ii in range(len(x_)):
                    t = x_[ii] * x_[ii]
            norm = math.sqrt(t)
            for ii in range(len(x_)):
                    x_[ii] = x_[ii] / norm
            x.append(x_)
            if int(i[0]) == -1:
                    y.append(0)
            else:
                    y.append(int(i[0]))

def dis(a, b):
        d = 0
        for i in range(len(a)):
                d = d +  (a[i] - b[i]) * (a[i] - b[i])
        d = math.sqrt(d)
        return d

def skipCopy(b, i):
        a = []
        for x in range(len(b)):
            a.append(b[x])
        t = a.pop(i)
        return a

def _knn_(x, xx, yy, k):
        bx = []
        for i in range(len(xx)):
                bx.append((dis(x, xx[i]), yy[i]))
        bx = sorted(bx)[:k]
        fp = 0
        fn = 0
        for i in range(len(bx)):
                if bx[i][1] == 1:
                        fp = fp + 1
                else:
                        fn = fn + 1
        if fp > fn:
                return 1
        else:
                return 0

def knn_(x, y, k):
    bestX = []
    bestY = []
    for i in range(len(x)):
        xx = skipCopy(x, i)
        yy = skipCopy(y, i)
        p = _knn_(x[i], xx, yy, k)
        if y[i] == p:
                bestX.append(x[i])
                bestY.append(y[i])
    best = []
    best.append(bestX)
    best.append(bestY)
    return best

def knn(bx, by, k):
        n = len(bx[0])
        nn = len(bx)
        w = [0] * n
        for s in range(10000):
                t = [0] * n
                for x in range(nn):
                        wtx = 0
                        for xx in range(n):
                                wtx = wtx + w[xx] * bx[x][xx]
                        yt = 1 / (1 + np.exp(-1 * wtx))
                        yt = yt - by[x]
                        for xx in range(n):
                                t[xx] = t[xx] + yt * bx[x][xx]
                for x in range(n):
                        w[x] = w[x] - 0.002 * t[x]
        return w

def predict(x, w):
        p = 0
        for i in range(len(x)):
                p = p + w[i] * x[i]
        #print(p)
        if p > 0:
                return 1
        else:
                return 0

def errC(x, y, k, tx, ty):
        b = knn_(x, y, k)
        bx = b[0]
        by = b[1]
        w = knn(bx, by, k)
        werr = 0
        terr = 0

        for i in range(len(x)):
                py = predict(x[i], w)
                #print(i, ":  ", py, "  -  ", y[i])
                if(py != y[i]):
                        #print(i, ":  ", py, "  -  ", y[i])
                        werr = werr + 1
        werr = float(werr / len(x))

        for i in range(len(tx)):
                py = predict(tx[i], w)
                #print(i, ":  ", py, "  -  ", y[i])
                if(py != ty[i]):
                        #print(i, ":  ", py, "  -  ", y[i])
                        terr = terr + 1
        terr = float(terr / len(tx))

        return [werr, terr]

#print(errC(x, y, k, tx, ty))

def loo(x, y, k):
        dl = int(len(x) / 3)
        terr = []
        for i in range(3):
                print(i)
                train_x  = []
                train_y = []
                test_x = []
                test_y = []
                ii = 0
                #print(i)
                while ii < i * dl:
                        train_x.append(x[ii])
                        train_y.append(y[ii])
                        ii = ii + 1

                while ii < (i + 1) * dl:
                        test_x.append(x[ii])
                        test_y.append(y[ii])
                        ii = ii + 1
                        #print(ii, end=", ")
                #print('\n')
                while ii < 3 * dl:
                        train_x.append(x[ii])
                        train_y.append(y[ii])
                        ii = ii + 1
                #print(test_y)
                #print(train_y)
                e = errC(train_x, train_y, k, test_x, test_y)
                terr.append(e[1])
                #print(e[1])
        er = 0
        for i in range(len(terr)):
                er = er + terr[i]
        er = er/3.
        return er

def kNN(x, y, k, tx, ty):
        r = errC(x, y, k, tx, ty)
        o = loo(x, y, k)
        return [r[0], o, r[1]]

def rk(x, y, k, tx, ty):
            h = []
            for i in range(14):
                    kk = 1 + i * 4
                    print("K = ", kk)
                    h.append(kNN(x, y, kk, tx, ty))
            for i in range(len(h)):
                    print(h[i][0], end = ',')
            print("\n\n")
            for i in range(len(h)):
                    print(h[i][1], end = ',')
            print("\n\n")
            for i in range(len(h)):
                    print(h[i][2], end = ',')

rk(x, y, k, tx, ty)
