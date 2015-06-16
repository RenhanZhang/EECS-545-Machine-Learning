import numpy as np
import math
import matplotlib.pylab as plt
import matplotlib
def sigmoid(l):
    return [1/(1.0+ math.exp(-a)) for a in l]

def softmax(l):
    result = [math.exp(-a) for a in l]
    return np.array(result)/np.sum(result)

# normalize an array 
def normalize(l):
    l = l - np.min(l)
    return l/float(np.max(l)) * 255

def part2(w):
    f = plt.figure()
    for i in range(16):
        f.add_subplot(4,4,i+1)
        plt.axis('off')
        plt.imshow(np.reshape(normalize(w[1:,i]), (20,20)), cmap = matplotlib.cm.Greys_r)
    plt.savefig('3.png')

# forward and backward propagation
def predict(data, w1, w2):
    prediction = []
    for x in data:
        S_in =  np.dot(x, w1)                         # S_in is the input of the hidden layer
        S_out = sigmoid(S_in)
        S_out.insert(0,1)
        S_out = np.array(S_out)                        # S_out ~ (1, 17)
        M_in = np.dot(S_out, w2)                       # M_in: input of the softmax layer
        M_out = softmax(M_in)
        prediction.append(M_out.argmax() + 1)
    return prediction

def forbac_pro(x, label, w1, w2):
	# x ~ (1, 401), w1 ~ (401, 16), w2 ~ (17, 10)
    S_in =  np.dot(x, w1)                         # S_in is the input of the hidden layer
    S_out = sigmoid(S_in)
    S_out.insert(0,1)
    S_out = np.array(S_out)                        # S_out ~ (1, 17)

    M_in = np.dot(S_out, w2)                       # M_in: input of the softmax layer
    M_out = softmax(M_in)                          # M_out ~(1,10)

    # gradient of log likelihood w.r.t w2
    gd_w2 = np.tile(S_out, (10,1)).T * (-M_out)
    gd_w2[:,label-1] = gd_w2[:,label-1] + S_out


    # gradient of log likelihood w.r.t S_out
    gd_sout = w2[:, label-1] - np.sum( w2 * M_out,axis = 1)

    # gradient of log likelihood w.r.t S
    gd_s = S_out * (1 - S_out) * gd_sout
    #gd_s = S_out[1:] * (1 - S_out[1:]) * gd_sout[1:]       # gd_s ~ (1,16)

    # gradient of log likelihood w.r.t w1
    #gd_w1 = np.tile(x, (10,1)).T * gd_s
    gd_w1 = np.zeros((401,16))
    for i in range(16):
        gd_w1[:,i] = x * gd_s[i+1]

    return gd_w1, gd_w2

def train(train_data, test_data):
	# randomly init w to be uniformly distributed in [-0.12,0.12]
    w1 = np.random.rand(401, 16) * 0.24 - 0.12
    w2 = np.random.rand(17, 10) * 0.24 - 0.12
    train_err = []
    test_err = []
    for i in range(600):
        print 'Iteration %d' %i
        grad1, grad2 = np.zeros((401, 16)), np.zeros((17, 10))
    	for dp in train_data:
    		g1, g2 = forbac_pro(dp[:-1], dp[-1], w1, w2)                   # compute gradient of w1 and w2 
    		grad1 += g1 + 0.5 * lmd * w1
    		grad2 += g2 + 0.5 * lmd * w2
    	w1 = w1 - grad1 * LR
    	w2 = w2 - grad2 * LR
        train_pre = predict(train_data[:,:-1], w1, w2)
        err = np.sum(train_pre != train_data[:,-1])
        train_err.append(err)

        test_pre = predict(test_data[:,:-1], w1, w2)
        err = np.sum(test_pre != test_data[:,-1])
        test_err.append(err)

    plt.plot(np.arange(1,601), train_err, label = 'Train Error')
    plt.plot(np.arange(1,601), test_err, ls = ':', label = 'Test Error')
    plt.legend()
    plt.savefig('2.png')
    
    return w1, w2
        


def part1():
    train_data = np.genfromtxt('train.csv', delimiter=',')
    train_data = np.hstack((np.ones((len(train_data),1)),train_data))
    
    test_data = np.genfromtxt('test.csv', delimiter = ',')
    test_data =np.hstack((np.ones((len(test_data),1)), test_data))

    w1, w2 = train(train_data, test_data)

    return w1

print 'Q2'
lmd = 0.001       # regularization term
LR = .0005
                  # learning rate

w1 = part1()
part2(w1)
