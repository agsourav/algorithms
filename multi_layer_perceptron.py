import pandas as pd
import random
import math
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
def weight_adjust(w,w1,lr,unj,feature_count,ynj,vnj,hn,bias):
    e = lr*(ynj - vnj)*activation(unj)*(1 - activation(unj))
    #print("\nWeight: ",w)
    for i in range(1,feature_count):
        w[i] = w[i] + e*w1*hn[i-1]
        #print(i,'\t',w[i])
    w[0] = w[0] + e*w1*bias
    #print(0,'\t',w[0],'\n')
    return w

def activation(u):                              #used only in output layer
    return 1/(1 + math.exp(-u))

def summation_weight(w,x,feature_count,bias):   #this function is irrespective of layer
    sum = 0
    for i in range(feature_count):
        sum = sum + w[i]*x[i]
    sum = sum + w[0]*bias
    return sum

def output(y,classes):                          #output vector y
    yout = [0]*classes
    yout[y]=1
    return yout

def neuron_layer(hn,no_of_nodes,bias):
    for layer in range(1,no_of_layers+1):
        for k in range(no_of_nodes[layer]):
            u = summation_weight(w[layer-1][k],hn[layer-1],no_of_nodes[layer-1],bias[layer-1])
            #print('layer: ',layer,'\tnode :',k,'\tu :',u)
            hn[layer][k] = activation(u)
    return hn
def train(no_of_nodes,w,hn,yn,lr,bias):
    e = 0
    layer = 1
    for nodes in range(1,no_of_nodes[layer]+1):
       for classes in range(no_of_nodes[layer+1]):
            #print('classes: ',classes,'\tnode :',nodes)
            vnj = hn[layer+1][classes]
            #print('Actual :',yn[classes],'\tpredicted',vnj)
            w[layer][classes][nodes] = w[layer][classes][nodes] + lr*(yn[classes] - vnj)*vnj*(1-vnj)*hn[layer][nodes-1]
   
    for classes in range(no_of_nodes[layer+1]):
        vnj = hn[layer+1][classes]
        w[layer][classes][0] = w[layer][classes][0] + lr*(yn[classes] - vnj)*vnj*(1-vnj)*bias[layer]
        #adjust the weights corresponding to bias and also for second layer
        e = e + 1/2*math.pow((yn[classes] - vnj),2)
    layer = 0
    for i in range(no_of_nodes[layer]):
        for j in range(no_of_nodes[layer+1]):
            s = 0
            for classes in range(no_of_nodes[layer+2]):
                vnk = hn[layer+2][classes]
                vnj = hn[layer+1][j]
                s = s + (yn[classes] - vnj)*vnj*(1-vnj)*vnk*(1-vnk)*w[layer+1][classes][j+1]
            w[layer][j][i+1] = w[layer][j][i+1] + lr*s*hn[layer][i]
                                                                
    return w,e 

def predict(x,bias):
    hn[0] = x
    h1 = neuron_layer(hn,no_of_nodes,bias)
    print(h1[-1])
#__main__

data = pd.read_csv("../Downloads/datasets/Iris.csv")
data.drop(columns=['Id'],axis=1)
bias = []
eta = 0.1
col = data.columns
target = data[col[-1]]
features = data.drop(columns=[col[-1]],axis=1)
feature_count = len(features.columns)
classes = len(target.unique())
no_of_layers = 2
no_of_nodes = [feature_count,5,classes,classes]
l = features.values
le = LabelEncoder()
target = list(le.fit_transform(target))
print(target)
hn = []     #this matrix will hold the value for each neuron
#initialising the weight
w = []
E = 1
for layer in range(0,no_of_layers+1):
    
    w1 = []
    bias.insert(layer-1,1)
    for j in  range(no_of_nodes[layer+1]+1):
        w2 = []
        for i in range(no_of_nodes[layer]+1):
            if(layer == no_of_layers):
                w2.insert(i,1)
            else:
                w2.insert(i,random.random())
        w1.insert(j,w2)
        print(w2,'\n')
    w.insert(layer,w1)
print('Weight matrix initialised\n')
#while(E>0.001):
#    for layer in range(no_of_layers):
count = 0
"""while(E>0.001):
    E = 0
    count = count+ 1
    E=train(no_of_layers,no_of_nodes,hn,w,bias,l,len(l),target,eta,classes)
    print('\ncount: ',count,'\terror: ',E,'\n')"""
hn.insert(0,l[0])
hn.insert(1,[0,0.5,0.6,0.4,1])
hn.insert(2,[0.3,0,0.56])
hn = neuron_layer(hn,no_of_nodes,bias)
#print(train(no_of_nodes,w,hn,output(target[n],classes),eta))
E = 1
out_error = []
count =0
while(E>0.001):
    count = count + 1
    print('Before iteration :',E,'\n')
    e_matrix = []
    for itr in range(len(l)):
        hn[0] = l[itr]
        hn = neuron_layer(hn,no_of_nodes,bias)
        act_out = output(target[itr],classes)
        w,e = train(no_of_nodes,w,hn,act_out,eta,bias)
        E = E + e
        e_matrix.append(e)
        #print('\n\nitr :',itr,'\t',e)
    
    #plt.plot([itr for itr in range(len(l))],e_matrix)
    #plt.show()
    k = E/len(l)
    E = k
    out_error.append(k)
    print(k)
    
    
