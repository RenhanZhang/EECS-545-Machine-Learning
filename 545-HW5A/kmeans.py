import numpy as np
import random
#improvement: distance can be euclidean, cosine, or user defined

def distance(a,b):
    return np.dot(a-b, a-b)

def find_nearest(v, u):
    '''
    find the elements in u that is closest to v
    '''
    nearest = 0  # index of the element in v that's nearest to v
    min_dist = 0
    for i in range(0,len(u)):
        dist = distance(v, u[i])
        #print str(i) + ': the dist is ' + str(dist)

















        if i == 0 or dist < min_dist:
            min_dist = dist
            nearest = i
    #print 'The nearst is ' + str(nearest)
    return nearest

def does_converge(m, n, thres, C):
    diff = 0.0
    d = m.shape[1] #dimention
    for i in range(0, len(m)):
        diff = diff + distance(m[i], n[i])

    #average over dimention and number of classes
    diff = diff/C/d
    print diff
    if diff > thres:
        return False
    else: return True

def kmeans(data, C, thres):
    # C is number of classes, thres is the convergence threshold
    N = data.shape[0]   #num of data points
    centroid = np.zeros(shape=[C, data.shape[1]])
    #randomly initialize mean vector
    for i in range(0, C):
        centroid[i] = data[random.randint(0, N-1)]
    label = np.zeros(N)

    while True:
        for i in range(0, N):
            label[i] = find_nearest(data[i], centroid)   #assign labels to the nearest centroids
            #print str(label[i])

        # update the centroids
        new_cent = np.zeros(shape=[C, data.shape[1]])
        for cl in range(0, C):
            index = label == cl
            new_cent[cl] = np.mean(data[index], axis=0)
        if does_converge(centroid, new_cent, thres, C):
            break
        centroid = new_cent
    return centroid, label

#use provided centroids to re-represent the original data
def predict(data, centroids):
    shape = data.shape
    new_data = np.zeros(shape=[shape[0], shape[1]])
    for i in range(0, shape[0]):
        label = find_nearest(data[i], centroids)
        new_data[i] = centroids[label]
    return new_data