import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import pathlib
data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + "\ex2data1.txt")
t = [0,0,0]
data["x0"] = np.ones(len(data))
def plot():
    admitted = data[data["y"]==1]
    not_admitted = data[data["y"]==0]
    plt.scatter(admitted["x1"],admitted["x2"])
    plt.scatter(not_admitted["x1"],not_admitted["x2"])
    plt.legend(loc = "upper right",labels = ["Admitted","NotAdmitted"])
    plt.show()
def g(z):
    return 1/(1+np.exp(-z))
def h(i):
    return g(t[0] + t[1] * data["x1"][i] + t[2] * data["x2"][i])
def j():
    s = 0
    for i in range(len(data)):
        s+= (data["y"][i]*np.log(h(i)+0.0000001)) - (1-data["y"][i])*np.log(1-h(i)+0.0000001)
    return 1/len(data)*s
def gradientdescent(t,alpha):
    temp = [0,0,0]
    s = [0,0,0]
    for j in range(len(t)):
        for i in range(len(data)):
            s[j]+= (h(i) - data["y"][i])*data["x"+str(j)][i]
    for j in range(len(t)):
        temp[j] = t[j] - alpha/len(data)*s[j]
    return temp
epochs = 400
for i in range(epochs):
    t = gradientdescent(t,0.001)
correct = 0
for i in range(len(data)):
    if h(i)>0.5:
        if data["y"][i] == 1:
            correct+=1
    elif h(i)<=0.5:
        if data["y"][i] == 0:
            correct +=1
print(correct/len(data)*100)
