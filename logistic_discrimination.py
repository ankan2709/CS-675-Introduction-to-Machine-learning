'''

Name : Ankan Dash
Course : CS 675, NJIT
HomeWork 4
Python program for the logistic discrimination gradient
descent algorithm

'''


import math
import sys
import random

# reading the data and labels

data = sys.argv[1]
f = open(data)
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
    data[i].append(1)  # FOR THE BIAS
    i += 1
f.close()

label_data = sys.argv[2]
f = open(label_data)
rows = len(data)
cols = len(data[0])
labels = {}

n = []
n.append(0)
n.append(0)

l = f.readline()
while (l != ''):
    a = l.split()
    labels[int(a[1])] = int(a[0])
    l = f.readline()
    n[int(a[0])] += 1
    
f.close()


learning_rate = sys.argv[3]
learning_rate = float(learning_rate)
stopping_condition = sys.argv[4] 
stopping_condition = float(stopping_condition)


# defining the dot product
def dot(w,x):
    dot_product = 0
    for col in range(0,len(data[0]),1):
        dot_product += w[col]*x[col]
    return dot_product

# defining the dot product
def sigmoid(x, w):
    dp = dot(x, w)
    sigma = 1 / (1 + math.exp(-1 * dp))
    if (sigma >= 1):
        sigma = 0.999999
    return sigma

# initializing w
val = 0.02*random.uniform(0,1) - 0.01
w = [val for j in range(len(data[0]))]

# logistic regression
diff = 1
error = 0

while (diff > stopping_condition):
    deltaf = [0 for x in range(len(data[0]))]
    
    for i in range(len(data)):
        if labels.get(i) != None:
            sig = sigmoid(w,data[i]) - labels[i]
            for j in range(len(data[0])):
                deltaf[j] += sig*data[i][j]
                
    for j in range(len(data[0])):
        w[j] -= learning_rate*deltaf[j]
        
    previous_objective = error
    error = 0
    
    for i in range(len(data)):
        if labels.get(i) != None:
            error += -(labels[i] * math.log(sigmoid(w, data[i])) +
                           ((1 - labels[i]) * math.log(1 - sigmoid(w, data[i]))))
            
            
        diff = abs(previous_objective - error)
        
weights = w[:-1]
print(weights)
w0 = w[-1]
print('\n')

normw = math.sqrt(sum([w[i]**2 for i in range(0, len(w)-1)]))#print('||w|| = ',normw)
print('\n')
distance_to_origin = (w0/normw)
print('Distance to origin ',distance_to_origin)


# classifying unlabeled data

for i in range(len(data)):
    if (labels.get(i) == None):
        dp = dot(w, data[i])
        if dp > 0:
            print('1', i)
        else:
            print('0', i)