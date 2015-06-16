import itertools
import numpy as np
from scipy.stats import rv_discrete
import time
import matplotlib.pyplot as plt

#transition matrix A[i,j]: prob from state i to state j. Note each row sums to 1
A = np.array([[0.5,0.2,0.3], [0.2,0.4,0.4],[0.4,0.1,0.5]])
fi = np.array([[0.8,0.2], [0.1,0.9],[0.5,0.5]])    #emission prob fi[i,j] = prob of see xj for state zi
pi = [0.5,0.3,0.2]                                #prior prob of state
x = (1,2,1,2)                                     #observations of x

def part1():
    Zs = []
    for row in itertools.product([1,2,3], repeat = 4):
        Zs.append(row)
    joints = []                                    #joint prob of (Z,x) for every possible Z
    priors = []
    lh = []
    for Z in Zs:
        joint = pi[Z[0]-1] * fi[Z[0]-1, x[0]-1]
        prior = pi[Z[0]-1]
        likelihood = fi[Z[0]-1, x[0]-1]
        for i in range(1,4):
            joint = joint * A[Z[i-1]-1, Z[i]-1] * fi[Z[i]-1, x[i]-1]
            prior = prior * A[Z[i-1]-1, Z[i]-1]
            likelihood = likelihood * fi[Z[i]-1, x[i]-1]
        joints.append(joint)
        priors.append(prior)
        lh.append(likelihood)
    Zs = np.array(Zs)
    priors = np.array(priors)
    lh = np.array(lh)
    posteri = np.array(joints)/np.sum(joints)
    idx = (-posteri).argsort()
    lh = lh[idx]
    Zs = Zs[idx,:]
    posteri = posteri[idx,:]
    priors = priors[idx]
    joints = np.array(joints)[idx]
    a = 1
    for i in range(0, 0):
        print Zs[i,:], priors[i], lh[i], joints[i], posteri[i]

def sample():
    samples = []
    zz = [1,2,3]
    xx = [1,2]
    prior_gen = rv_discrete(name = 'prior', values = (zz, pi))

    trans_gen = []
    trans_gen.append(rv_discrete(name = 'tran1', values = (zz, A[0,:])))
    trans_gen.append(rv_discrete(name = 'tran2', values = (zz, A[1,:])))
    trans_gen.append(rv_discrete(name = 'tran3', values = (zz, A[2,:])))

    x_gen = []
    x_gen.append(rv_discrete(name = 'emission1', values = (xx, fi[0,:])))
    x_gen.append(rv_discrete(name = 'emission2', values = (xx, fi[1,:])))
    x_gen.append(rv_discrete(name = 'emission3', values = (xx, fi[2,:])))

    for i in range(5000):
        X = []
        Z = []
        Z.append(prior_gen.rvs())
        X.append(x_gen[Z[0]-1].rvs())
        for j in range(1,4):
            Z.append(trans_gen[Z[j-1]-1].rvs())
            X.append(x_gen[Z[j]-1].rvs())
        samples.append(X)
    return np.array(samples)

# forward backward algorithm
def forbac(x,lA,lPi, lFi):
    alpha = np.zeros([4, 3])
    beta = np.zeros([4, 3])
    alpha[0,:] = lPi * lFi[:, x[0]-1]
    beta[3,:] = 1
    for i in range(1,4):
        alpha[i,:] = lFi[:,x[i]-1] * np.dot(alpha[i-1, :], lA)
        for k in range(0,3):
            beta[3-i,k] = np.dot(beta[4-i,:], lA[k,:]*lFi[:,x[4-i]-1])
    px = float(np.sum(alpha[3,:]))                     #jointly probability of p(x1,x2,x3,x4)
    epsilon = np.zeros([3,3,3])
    gamma = np.zeros([4, 3])

    for n in range(4):
        for i in range(3):
            gamma[n,i] = alpha[n, i] * beta[n, i]/px
    for n in range(1,4):
        for i in range(0,3):                      # z(n) = i
            for j in range(0,3):                  # z(n-1) = j
                epsilon[n-1,i,j] = alpha[n-1, j] * lFi[i, x[n]-1] * lA[j,i] * beta[n,i]/px
    return px,gamma, epsilon

def binary(l):
    result = 0
    for x in l:
        result = result * 2 + x
    return result

def EM(data, lA, lPi, lFi):
    true_px = {}
    dist = []
    for x in itertools.product([1,2], repeat = 4):
        px, g, e = forbac(x, A, pi, fi)
        true_px[binary(x)] = px

    for i in range(ite+1):
        gamma = []     # gamma ~ (R,L,p) (R,4,3), R: # of data points, L: len of observations p: number of possible states
        epsilon = []   # (R, L-1,p,p), (R,3,3,3)
        Px = {}        # list of joint probability on X
        exist_gamma = {}
        exist_epsilon = {}
        for x in data:
            bi_x = binary(x)
            if exist_gamma.get(bi_x,None) is None:
               px, g, e = forbac(x, lA, lPi, lFi)
               gamma.append(g)
               epsilon.append(e)
               Px[bi_x] = px
               exist_gamma[bi_x] = g
               exist_epsilon[bi_x] = e
            else:
                gamma.append(exist_gamma[bi_x])
                epsilon.append(exist_epsilon[bi_x])

        dist.append(0.5 * np.sum([abs(true_px[key] - Px[key]) for key in Px.keys()]))
        gamma = np.array(gamma)
        epsilon = np.array(epsilon)

        # update lPi
        lPi = np.sum(gamma[:,0,:], axis=0)
        lPi = lPi/np.sum(lPi)

        # update lA
        for j in range(3):
            for k in range(3):
                lA[j,k] = np.sum(epsilon[:, :, j, k])
            lA[j,:] = lA[j,:]/np.sum(lA[j,:])

        # update lFi
        temp_x = np.reshape(np.array(data), [len(data)*4,])
        for k in range(3):
            temp_gamma = np.reshape(gamma[:,:,k], [4*len(data),])
            for i in range(2):
                lFi[k,i] = np.sum(temp_gamma[temp_x == i+1])/np.sum(temp_gamma)
        #print np.abs(lA-A)
        #print np.abs(lPi-pi)
    #print np.abs(lA-A)
    #print np.abs(lPi-pi)
    return dist[1:]
    


def part2():
    samples = sample()
    N = [500, 1000, 2000, 5000]
    # randomly init lA, lpi
    init_A = np.random.rand(3,3)
    init_A = (init_A.T/init_A.sum(axis=1)).T

    init_Pi = np.random.rand(3)
    init_Pi = init_Pi/init_Pi.sum()

    init_fi = np.random.rand(3,2)
    init_fi = (init_fi.T/init_fi.sum(axis=1)).T

    dist_array = []
    for n in N:
        print '---------------N = %d-----------------' %n
        dist = EM(samples[:n, :], np.copy(init_A), np.copy(init_Pi), np.copy(init_fi))     # learned A, learned fi
        dist_array.append(dist)
    t = np.arange(1,ite +1)

    plt.plot(t, dist_array[0], ls = '-.', label='N=500')
    plt.plot(t, dist_array[1], ls = ':',label='N=1000')
    plt.plot(t, dist_array[2], ls = '--',label='N=2000')
    plt.plot(t, dist_array[3], label = 'N=5000')

    plt.legend()
    plt.savefig('1.png')
    #plt.show()

print 'Q1'
ite = 10
start_time = time.time()
#part1()
part2()
#print '----------%s seconds-------------' %(time.time()-start_time)
