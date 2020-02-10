'''

Name : Ankan Dash
Course : CS 675
Assignment 1

'''
import sys

file = sys.argv[1]
f = open(file)
data = []
i = 0
l = f.readline()

## reading the data 

while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    l = f.readline()
rows = len(data)
cols = len(data[0])
f.close()

## reading the labels

label_file = sys.argv[2]
f = open(label_file)
train_labels = {}
n = []
n.append(0)
n.append(0)
l = f.readline()
while(l!= ''):
    a = l.split()
    train_labels[int(a[1])] = int(a[0])
    l = f.readline()
    n[int(a[0])] += 1

## computing means

m0 = []

for j in range(0,cols,1):
    m0.append(0.01)  # initilize the mean of each column to 0.001 to counter zero error
    
m1 = []
for j in range(0,cols,1):
    m1.append(0.01)
    
for i in range(0,rows,1):
    if (train_labels.get(i) != None and train_labels[i] == 0):
        for j in range(0,cols,1):
            m0[j] = m0[j] + data[i][j]
    if (train_labels.get(i) != None and train_labels[i] == 1):
        for j in range(0,cols,1):
            m1[j] = m1[j] + data[i][j]
        
for j in range(0,cols,1):
    m0[j] = m0[j]/n[0]
    m1[j] = m1[j]/n[1]


## computing variance

s0 = []

for j in range(0,cols,1):
    s0.append(0)    # initializing the SD of each of the columns to be 0 
    
s1 = []
for j in range(0,cols,1):
    s1.append(0)    # initializing the SD of each of the columns to be 0
    
for i in range(0,rows,1):
    if (train_labels.get(i) != None and train_labels[i] == 0):
        for j in range(0,cols,1):
            s0[j] = s0[j] + (data[i][j] - m0[j])**2
    if (train_labels.get(i) != None and train_labels[i] == 1):
        for j in range(0,cols,1):
            s1[j] = s1[j] + (data[i][j] - m1[j])**2
        
for j in range(0,cols,1):
    s0[j] = (s0[j]/n[0])**0.5
    s1[j] = (s1[j]/n[1])**0.5



# classify unlabeled points

for i in range(0,rows,1):
    if (train_labels.get(i) == None):
        d0 = 0
        d1 = 0
        for j in range(0,cols,1):
            d0 = d0 + (((data[i][j] - m0[j])/s0[j])**2)
            d1 = d1 + (((data[i][j] - m1[j])/s1[j])**2)
            
        if (d0<d1):
            print('0',i)
        else:
            print('1',i)