import pandas as pd
import random
import math
from sklearn.preprocessing import LabelEncoder

def neuron_weight(w,lr,unj,feature_count,ynj,vnj,x):
    c = lr*(ynj - vnj)*math.exp(unj)/math.pow(1+math.exp(unj),2)     #activation(unj)*(1-activation(unj))            
    #print("\nweight adjustment: ",c,"\n")
    for i in range(feature_count):
        w[i+1] = w[i+1] + c*x[i]
    w[0] = w[0] + c
    return w
    
def activation(u):
    v = 1/(1+math.exp(-u))
    return v

def u_nj(w,x,j,n,d,bias):
    sum = 0
    for i in range(0,d):
        #print(w[j][i+1]," ",x[n][i])
        sum = sum + w[j][i+1]*x[n][i]
    #print(w[j][0]," ",bias)
    sum = sum + w[j][0]*bias
    return sum

def output(y,classes):  #output vector y
    yout = [0]*classes
    yout[y]=1
    return yout

def train(w,l,y,bias,eta,classes,feature_count):
    E = 0
    #print("\nE",E,"\n")
    for k in range(len(l)):
        yout = output(y[k],classes)
        enj = 0
        for j in range(classes):
            unj = u_nj(w,l,j,k,feature_count,bias)
            vnj = activation(unj)
            #print(vnj)
            enj = math.pow(yout[j]-vnj,2)*1/2
            w[j] = neuron_weight(w[j],eta,unj,feature_count,yout[j],vnj,l[k])
            #print("predicted output:",vnj,"\tactual output: ",yout[j],"\terror: ",enj)
            E = E + enj
            #print("\nError at each step: ",E,"\tweight:",w,"\n")
    return E/(len(l)),w
#__main__
data = pd.read_csv("../Downloads/datasets/Iris.csv")
data = data.drop(columns=['Id'],axis =1 )
le = LabelEncoder()
bias = 1
eta = 0.1
col = data.columns
y = data[col[-1]]
x = data.drop(columns=[col[-1]],axis=1)
classes = len(y.unique())
feature_count = len(x.columns)
y = le.fit_transform(y)
y = list(y)
#initialising the weight
w = []
for j in range(classes):
    wi = []
    for i in range(feature_count + 1):
        wi.insert(i,random.random())
    w.insert(j,wi)

l = x.values

#u = u_nj(w,l,1,0,feature_count,bias)
#v = activation(u)
E = 10
count = 0
while(E>0.001):
    E,w = train(w,l,y,bias,eta,classes,feature_count)
    print("\n\nTotal error: ",E)#," weight: ",w)
    count = count + 1

