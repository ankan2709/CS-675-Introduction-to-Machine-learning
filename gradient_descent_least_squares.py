'''
Name: Ankan Dash
Course: CS 675
HomeWork 2, Implementing Gradient descent for Least Squares

'''

import sys
import math
import random

## reading the data and labels

file = sys.argv[1]
f = open(file)
data = []
i = 0
l = f.readline()


while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    l = f.readline()
    
#for i in range(len(data)):
    #data[i].append(float(1))

rows = len(data)
cols = len(data[0])
f.close()


label_file = sys.argv[2]
f = open(label_file)
labels = {}
n = []
n.append(0)
n.append(0)
l = f.readline()
while(l!= ''):
    a = l.split()
    labels[int(a[1])] = int(a[0])
    l = f.readline()
    n[int(a[0])] += 1
    
## converting labels with "0" to "-1"    
for key,values in labels.items():
    if labels[key] == 0:
        labels[key] = -1
    

## initializing W
val = 0.02*random.uniform(0,1) -1
W = [val for x in range(cols)]

# defining the Dot product
def dot(W, X):
    res = [x * y for x, y in zip(W, X)]
    return sum(res)



## defining the Cost function or the least squares function

def cost(w,x,y):
    cost = sum([(dot(w, x[i]) - y.get(i))**2 for i in range(0, len(x)) if y.get(i) != None])
    return cost


## Implementing Gradient Descent 
previous_cost = cost(W,data,labels)

learning_rate = 0.0001

stopping_condition = 0.001

max_iteration = 1000

condition = False

iterarion_no = 0

while (condition == False):
    
    df = []
    
    for i in range(len(data)):
        if labels.get(i) != None:
            dff = [(data[i][j]*(dot(W, data[i]) - labels.get(i))) for j in range(len(data[0]))]
            df.append(dff)


    
    deltaf = [sum(i) for i in zip(*df)]
    
    weight = [(W[i] - learning_rate*deltaf[i]) for i in range(0,len(W))]
    W = weight
    error = cost(W,data,labels)
    #print(error)
    #print('\n')
        
    if (abs(previous_cost - error) <= stopping_condition):
        condition = True
    previous_cost = error
    iterarion_no = iterarion_no + 1

    if iterarion_no == max_iteration:
        condition = True


#print(W[:-1])
#print('\n')
#normw = math.sqrt(sum([W[i]**2 for i in range(0, len(W)-1)]))
#W0 = W[len(W)-1]
#distance_to_origin = abs(W0/normw)
#print('Distance to origin ',distance_to_origin)
#print('\n')


## classifying on unlabeled data
for i in range(0,rows,1):
    if (labels.get(i) == None):
        activation = dot(W,data[i])        
        if activation > 0:
            print('1', i)
        else:
            print('0', i)