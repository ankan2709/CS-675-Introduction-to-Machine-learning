'''

Name : Ankan Dash
Course : CS 675, NJIT
HomeWork 3
Python program for optimizing the SVM hinge loss

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
rows = len(data)
cols = len(data[0])
f.close()

label_data = sys.argv[2]
f = open(label_data)
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
    
for key, value in labels.items():
    if labels[key] == 0:
        labels[key] = -1
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

# initializing w
val = 0.02*random.uniform(0,1) - 0.01
w = [val for j in range(len(data[0]))]


# implementing the hinge loss svm algorithm
diff = 1
error = 0


while(diff > stopping_condition):
    deltaf = [0 for j in range(len(data[0]))]
    
    for i in range(len(data)):
        if (labels.get(i) != None):
            dp = labels[i] * dot(w,data[i])
            for j in range(len(data[0])):
                if dp < 1:
                    deltaf[j] += -(labels[i]*data[i][j])
                else:
                    deltaf[j] += 0
                    
    for j in range(len(data[0])):
        w[j] -= learning_rate * deltaf[j]
 
    previous_objective = error
    error = 0
    
    for i in range(len(data)):
        if (labels.get(i) != None):
            error += max(0, 1 - (labels.get(i)) * dot(w, data[i]))
    
    diff = abs(previous_objective - error)

weights = w
print(weights)
w0 = w[-1]
print('\n')

normw = math.sqrt(sum([w[i]**2 for i in range(0, len(w)-1)]))
distance_to_origin = abs(w0/normw)
print('Distance to origin ',distance_to_origin)

# classifying unlabeled data

for i in range(len(data)):
    if (labels.get(i) == None):
        dp = dot(w, data[i])
        if dp > 0:
            print('1', i)
        else:
            print('0', i)