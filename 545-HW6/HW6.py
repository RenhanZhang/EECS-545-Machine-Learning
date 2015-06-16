from DecisionTree import DecisionTree
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
import sys
def Q1():
    data = np.genfromtxt('data.csv', delimiter=',')
    dt = DecisionTree(data[1:,:-1], data[1:,-1], [0,1,2,3])
    dt.traverse()

def multinomial(l):
    l = np.array(l)
    cp = 0                      # cumulative prob
    rnd_pick = random.uniform(0,1)
    for i in range(len(l)):
        cp += l[i]
        if rnd_pick <= cp: return i
    return len(l) - 1

def Q2a():
    epsilon = [0.01, 0.1, 1]
    reward_mean = [.3, .5, .6]
    step = 15000
    avg_reward = np.zeros([len(epsilon),step])           # accumulative reward avged over 50 iterations
    ite = 50.0
    tau = 0.01
    for kk in range(len(epsilon)):
        e = epsilon[kk]
        for rep in range(int(ite)):
            Q = [10,10,10]                 # initialize the empirical mean as 1 to encourage exploration
            accu_reward = np.zeros(step)      # accumulative reward
            k = [0,0,0]                        # record the times a certain arm is picked
            dec = []
            for i in range(step):

                if e != -1:
                    # find the arm with the max Q value
                    max_idx = [a for a, b in enumerate(Q) if b == max(Q)]
                    # in case there are more than one maxima, pick one at random
                    iidx = int(random.uniform(0, len(max_idx)))
                    if iidx == len(max_idx):
                        iidx = iidx - 1
                    idx = max_idx[iidx]

                    # use epsilon-greedy policy to pick an arm
                    rand_pick = random.uniform(0, 1)
                    if rand_pick > 1-e:
                        idx = int(random.uniform(0, len(Q)))
                        if idx == len(Q):
                           idx = idx - 1
                else:
                    max_idx = [a for a, b in enumerate(Q) if b == max(Q)]
                    # in case there are more than one maxima, pick one at random
                    iidx = int(random.uniform(0, len(max_idx)))
                    if iidx == len(max_idx):
                        iidx = iidx - 1
                    idx = max_idx[iidx]

                    # use epsilon-greedy policy to pick an arm
                    rand_pick = random.uniform(0, 1)
                    if rand_pick > 0.999:
                       #power = [math.exp(h/tau) for h in Q]
                       #power = np.array(power)
                       #prob = power/power.sum()
                       prob = np.array(Q)/sum(Q)
                       idx = multinomial(prob)

                dec.append(idx)
                # generate the reward of the idx th arm
                reward_pick = random.uniform(0, 1)
                reward = 0
                if reward_pick <= reward_mean[idx]:
                    reward = 1

                # update the accumulative reward
                if i == 0:
                    accu_reward[i] = reward
                else:
                    accu_reward[i] = accu_reward[i-1] + reward

                # update Q and k
                k[idx] = k[idx] + 1
                Q[idx] = Q[idx] + 1.0/k[idx] * (reward - Q[idx])
            avg_reward[kk,:] += accu_reward/ite

    return avg_reward

def Q2b():

    random.seed()
    reward_mean = [.3, .5, .6]
    step = 15000
    avg_reward = np.zeros(step)           # accumulative reward avged over 50 iterations
    ite = 50.0
    tau = 0.01
    for rep in range(int(ite)):
        Q = [5,5,5]                 # initialize the empirical mean as 1 to encourage exploration
        accu_reward = np.zeros(step)      # accumulative reward
        k = [0,0,0]                        # record the times a certain arm is picked
        dec = []
        for i in range(step):
            if i < 100:
                e = 1
            elif i < 1000:
                e = 0.1
            else:
                e = 0.001
            max_idx = [a for a, b in enumerate(Q) if b == max(Q)]
            # in case there are more than one maxima, pick one at random
            iidx = int(random.uniform(0, len(max_idx)))
            if iidx == len(max_idx):
                iidx = iidx - 1
            idx = max_idx[iidx]
            rand_pick = random.uniform(0, 1)
            if rand_pick > 1-e:
               prob = [0.33,0.33,0.34]
               idx = multinomial(prob)
            dec.append(idx)

            # generate the reward of the idx th arm
            reward_pick = random.uniform(0, 1)
            reward = 0
            if reward_pick <= reward_mean[idx]:
                reward = 1

            # update the accumulative reward
            if i == 0:
                accu_reward[i] = reward
            else:
                accu_reward[i] = accu_reward[i-1] + reward

            # update Q and k
            k[idx] = k[idx] + 1
            Q[idx] = Q[idx] + 1.0/k[idx] * (reward - Q[idx])
        avg_reward += accu_reward/ite
    return avg_reward

def Q2():
    step = 15000
    eg_reward = Q2a()
    modified_reward = Q2b()
    n = np.arange(1, step+1)
    plt.plot(0.6-eg_reward[0,:]/n, label = '0.01')
    plt.plot(0.6-eg_reward[1,:]/n, label = '0.1')
    plt.plot(0.6-eg_reward[2,:]/n, label = '1')
    plt.plot(0.6-modified_reward/n, label = 'modified')
    plt.axis([0,15000, 0, 0.2])
    plt.legend()
    plt.savefig('1.png')

def Q3():
    gamma = 0.95
    alpha = 0.1

    offset = {'S':5, 'N':-5, 'W':-1,'E':1}
    pol = [['S','S','S','S','S'],
           ['N','S','S','S','S'],
           ['E','N','W','W','W'],
           ['N','N','N','N','W'],
           ['N','N','N','N','W'],
          ]
    pol = np.reshape(pol, [25,])

    p = np.diag(0.2*np.ones(25))
    for i in range(25):
        if i == 5 or i == 11:
            continue
        p[i, i+offset[pol[i]]] = 0.8
    # deals with A and B
    p[5,5] = 0
    p[5,8] = 1
    p[11,11] = 0
    p[11, 18] = 1
    a = 1
    R = np.zeros(25)
    R[5] = 3
    R[11] = 5

    # part a
    V = np.dot(inv(np.eye(25) - gamma * p), R)
    c = 1

    # part b
    v = np.zeros(25)
    bottom_left = []
    for i in range(500):
        v = R + gamma * np.dot(p, v)
        bottom_left.append(v[20])

    plt.plot(np.arange(1,501),bottom_left)
    plt.savefig('q3-2.png')
    plt.close()
    # part c
    mc_bl = []
    for i in range(1000):
        s = 20
        r = 0
        for k in range(100):
            next_s = multinomial(p[s,:])
            r += pow(gamma, k) * R[s]
            s = next_s
        mc_bl.append(r)
    for i in range(1,1000):
        mc_bl[i] = mc_bl[i-1] + mc_bl[i]
    n = np.arange(1,1001)
    plt.plot(n, mc_bl/n)
    plt.savefig('q3-3.png')
    plt.close()

    #part d
    td_bl = []
    N = 1000
    td_v = np.zeros(25)
    for i in range(N):
        s = 20
        for j in range(30):
            next_s = multinomial(p[s,:])
            td_v[s] = (1-alpha) * td_v[s] + alpha * (R[s] + gamma * td_v[next_s])
            s = next_s
        td_bl.append(td_v[20])
    plt.plot(n, td_bl)
    plt.axis([1,N, 0, 22])
    plt.savefig('q3-4.png')
    plt.close()

    # optimal control
    action = {'N':(-1, 0), 'S':(1,0), 'W':(0,-1), 'E':(0,1)}
    V = np.zeros([5,5])
    opt_a = np.ones([5,5])
    opt_a = opt_a.astype('str')
    oc_bl = []
    while True:
        next_V = np.zeros([5,5])
        for i in range(5):
            for j in range(5):
                if i == 1 and j == 0:
                    next_V[1,0] = 3 + gamma * V[1,3]
                    continue
                if i == 2 and j ==1:
                    next_V[2,1] = 5 + gamma * V[3,3]
                    continue
                this_opt_a = 'N'
                this_max_v = -sys.maxint
                for a in action.keys():
                    dir = action[a]
                    next_i = i + dir[0]
                    next_j = j + dir[1]
                    reward = 0
                    if next_i < 0 or next_i > 4 or next_j < 0 or next_j > 4:
                        next_i = i
                        next_j = j
                        reward = -1
                    value = 0.2 * gamma * V[i,j] + 0.8 * (reward + gamma * V[next_i, next_j])
                    if value > this_max_v:
                        this_max_v = value
                        this_opt_a = a
                opt_a[i,j] = this_opt_a
                next_V[i,j] = this_max_v
        oc_bl.append(next_V[4,0])
        if (next_V-V).max() < 0.001:
            break
        V = next_V

    plt.plot(np.arange(1, len(oc_bl)+1), oc_bl)
    plt.savefig('q3-5.png')
    plt.close()

    # q learning
    action = {'N':(-1, 0), 'S':(1,0), 'W':(0,-1), 'E':(0,1)}
    t = 30
    a_map = {0:'N', 1:'S', 2:'W', 3:'E'}
    epsilon = 0.4
    whole = np.zeros([t, 1000, 5,5,4])
    for trial in range(t):
        q = np.zeros([5,5,4])
        for episode in range(1000):
            s = (4,0)
            for l in range(30):
                i,j = s
                idxs = [a for a, b in enumerate(q[i, j]) if b == max(q[i, j])]              # epsilon greedy
                idx = idxs[int(random.uniform(0, len(idxs)))]
                if random.uniform(0,1) > 1-epsilon:
                    idx = int(random.uniform(0,4))
                dir = a_map[idx]
                a = action[dir]
                next_i = i + a[0]
                next_j = j + a[1]
                reward = 0
                if next_i < 0 or next_j < 0 or next_i > 4 or next_j > 4:
                    next_i = i
                    next_j = j
                    reward = -1
                if random.uniform(0,1) < 0.2:
                    next_i = i
                    next_j = j
                if s == (1,0):
                    next_i, next_j = (1,3)
                    reward = 3
                if s == (2,1):
                    next_i, next_j = (3,3)
                    reward = 5
                q[i,j,idx] = (1-alpha) * q[i,j,idx] + alpha * (reward + gamma * q[next_i,next_j,:].max())
                s = (next_i, next_j)
            if episode%100 == 0:
                xx = 1
            whole[trial, episode, :, :, :] = q
    e = whole.mean(axis = 0)
    value = e.max(axis = 3)
    ql_bl = value[:,4,0]
    '''
    for i in range(5):
        for j in range(5):
            print i, j, e[999,i,j,:]
    '''
    plt.plot(np.arange(1,1001), ql_bl)
    plt.savefig('q3-8.png')
    plt.close()











#Q1()
random.seed()
Q2()
Q3()
