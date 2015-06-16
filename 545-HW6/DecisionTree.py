import numpy as np
import math
import sys
from collections import deque
# find the optimal attribute that yields the max information gain
def optAttr(data, attrList, labels):
    minE = 2                 # initialize minE as a number greater than 1 since entropy is
                             # always equal or less than 1
    N = float(len(data))
    for attr in attrList:
        posIdx = data[:,attr] == 1
        posE = entropy(labels[posIdx])
        negIdx = data[:,attr] == 0
        negE = entropy(labels[negIdx])
        E = posE * sum(posIdx)/N + negE * sum(negIdx)/N
        if E <= minE:
            minE, optA = E, attr
    return optA

def entropy(labels):
    '''
    compute the binary entropy
    '''
    N = float(len(labels))
    E = 0
    for label in set(labels):
        n = sum(labels==label)
        if n == 0: continue
        E += n/N * (-math.log(n/N, 2))
    return E

class DecisionTree:

    def __init__(self, data, labels, attrList):
        self.root = DTreeNode(data, labels, attrList)

    def traverse(self):
        q = deque([self.root, None])
        while q:
            n = q.popleft()
            if n is None:
                if q:
                    print '\n----------'
                    q.append(None)
                    continue
                else:
                    break
            sys.stdout.write(str(n.layer))
            if n.left:
                q.append(n.left)
            else: sys.stdout.write('#')
            if n.right:
                q.append(n.right)
            else: sys.stdout.write('#')



class DTreeNode:
    def __init__(self, data, labels, attrList):
        '''
        attrList: list of all the feature attributes
        label: labels indicating the class
        '''
        if len(data) == 0:
            return None
        labels = np.array(labels)
        assert len(labels.shape) == 1                   # ensure the labels is one dim
        self.E = entropy(labels)
        self.layer = 5 - len(attrList)
        self.left = None
        self.right = None
        if self.E != 0:
            self.clas = -1
            self.attr = optAttr(data, attrList, labels)
            attr = self.attr
            attrList.remove(attr)
            self.left = DTreeNode(data[data[:,attr] == 1, :], labels[data[attr] == 1], attrList)
            self.right = DTreeNode(data[data[:,attr] == 0,:], labels[data[attr] == 0], attrList)
        else:
            self.clas = labels[0]


