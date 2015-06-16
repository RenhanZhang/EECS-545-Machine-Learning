#import Image
import numpy as np
import kmeans as km
import os
import re
import PCA
import matplotlib.image as mpim
import matplotlib.pyplot as plt
import matplotlib
import scipy
'''
def load_data(filename):
    im = Image.open(filename)
    return np.array(im)

def Q2_1():
    im = Image.open('mandrill-small.tiff')
    arr = np.array(im)
    shape = arr.shape
    data = arr.reshape((shape[0] * shape[1], shape[2]))
    cent, label = km.kmeans(data, 16, 0.01)
    return cent

def Q2_2(centroids):
    im = Image.open('mandrill-large.tiff')
    arr = np.array(im)
    shape = arr.shape
    data = arr.reshape((shape[0] * shape[1], shape[2]))
    compressed_data = km.predict(data, centroids)
    compressed_data = compressed_data.reshape((shape[0], shape[1], shape[2]))
    im2 = Image.fromarray(compressed_data, 'RGB')
    im2.show()
'''
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    #print '-----------'
    #print len(buffer), len(header), width, height, maxval
    return np.frombuffer(buffer,dtype='u1',count=int(width)*int(height),offset=13).reshape((int(height), int(width)))
def Q1_1(amu, aeigenvec, smu, seigenvec):
    fig = plt.figure()
    fig.add_subplot(3,2,1)
    plt.axis('off')
    plt.title('an2i')
    plt.imshow(np.reshape(amu, (30, 32)), cmap = matplotlib.cm.Greys_r)
    fig.add_subplot(3,2,3)
    plt.axis('off')
    plt.imshow(np.reshape(aeigenvec[:,0], (30, 32)), cmap = matplotlib.cm.Greys_r)
    fig.add_subplot(3,2,5)
    plt.axis('off')
    plt.imshow(np.reshape(aeigenvec[:,1], (30, 32)), cmap = matplotlib.cm.Greys_r)
    fig.add_subplot(3,2,2)
    plt.axis('off')
    plt.title('straight')
    plt.imshow(np.reshape(smu, (30, 32)), cmap = matplotlib.cm.Greys_r)
    fig.add_subplot(3,2,4)
    plt.axis('off')
    plt.imshow(np.reshape(seigenvec[:,0], (30, 32)), cmap = matplotlib.cm.Greys_r)
    fig.add_subplot(3,2,6)
    plt.axis('off')
    plt.imshow(np.reshape(seigenvec[:,1], (30, 32)), cmap = matplotlib.cm.Greys_r)
    plt.savefig('1.png')

def Q1_2(aeigenval, acov, seigenval,scov):

    a_fraction = np.zeros((960,1))
    acov = np.diag(acov)
    acov = acov[(-acov).argsort()]
    s_fraction = np.zeros((960,1))
    scov = scov[(-scov).argsort()]
    for i in range(0, 960):
        if i == 0:
            a_fraction[i] = abs(aeigenval[i])
            s_fraction[i] = abs(seigenval[i])
        else:
            a_fraction[i] = a_fraction[i-1] + abs(aeigenval[i])
            s_fraction[i] = s_fraction[i-1] + abs(seigenval[i])
            acov[i] = acov[i]+acov[i-1]
            scov[i] = scov[i]+scov[i-1]
    a_fraction = np.divide(a_fraction, a_fraction[959])
    s_fraction = np.divide(s_fraction, s_fraction[959])
    acov = np.divide(acov, a_fraction[959])
    scov = scov/s_fraction[959]
    fig = plt.figure(figsize=(16,8))

    fig.add_subplot(1,2,1)
    plt.title('an2i')
    plt.plot(np.arange(1,961), a_fraction)
    plt.plot(np.arange(1,961), acov)
    fig.add_subplot(1,2,2)
    plt.title('straight')
    plt.plot(np.arange(1,961), s_fraction)
    plt.plot(np.arange(1,961), scov)
    plt.savefig('2.png')

def Q1_3(amu, aeigenvec, smu, seigenvec):
    target = read_pgm('faces_4/an2i/an2i_straight_neutral_open_4.pgm')
    target = np.reshape(target,[1,30*32])

    a_coef = []
    s_coef = []
    a_coef.append(np.dot(target-amu, aeigenvec[:,0]))
    a_coef.append(np.dot(target-amu, aeigenvec[:,1]))
    s_coef.append(np.dot(target-smu, seigenvec[:,0]))
    s_coef.append(np.dot(target-smu, seigenvec[:,1]))
    x = np.arange(1,3)
    fig = plt.figure(figsize=(16,8))
    fig.add_subplot(1,2,1)
    plt.title('an2i')
    #plt.bar(a_coef,2)
    plt.bar([1,2], a_coef, align='center')
    plt.xticks([1,2], ['1st','2nd'])

    fig.add_subplot(1,2,2)
    plt.title('straight')
    #plt.bar(x, s_coef)
    plt.bar([1,2], s_coef, align='center')
    plt.xticks([1,2], ['1st','2nd'])
    plt.savefig('3.png')


def Q1_4(amu, aeigenvec, smu, seigenvec):
    K = [5,50, 960]
    target = read_pgm('faces_4/an2i/an2i_straight_neutral_open_4.pgm')
    target = np.reshape(target,[1,30*32])
    a_recovered = np.zeros([960, 3])
    s_recovered = np.zeros([960, 3])
    for i in range(3):
        k = K[i]
        a_recovered[:,i] = PCA.recover(target, amu, aeigenvec[:,:k])
        s_recovered[:,i] = PCA.recover(target, smu, seigenvec[:,:k])
    fig = plt.figure()
    fig.add_subplot(3,2,1)
    plt.axis('off')
    plt.title('an2i')
    plt.imshow(np.reshape(a_recovered[:,0], (30, 32)), cmap = matplotlib.cm.Greys_r)
    fig.add_subplot(3,2,3)
    plt.axis('off')
    plt.imshow(np.reshape(a_recovered[:,1], (30, 32)), cmap = matplotlib.cm.Greys_r)
    fig.add_subplot(3,2,5)
    plt.axis('off')
    plt.imshow(np.reshape(a_recovered[:,2], (30, 32)), cmap = matplotlib.cm.Greys_r)
    fig.add_subplot(3,2,2)
    plt.axis('off')
    plt.title('straight')
    plt.imshow(np.reshape(s_recovered[:,0], (30, 32)), cmap = matplotlib.cm.Greys_r)
    fig.add_subplot(3,2,4)
    plt.axis('off')
    plt.imshow(np.reshape(s_recovered[:,1], (30, 32)), cmap = matplotlib.cm.Greys_r)
    fig.add_subplot(3,2,6)
    plt.axis('off')
    plt.imshow(np.reshape(s_recovered[:,2], (30, 32)), cmap = matplotlib.cm.Greys_r)
    plt.savefig('4.png')

def main():
    flat_an2i, flat_straight = load_data()
    x = np.reshape(flat_an2i, [1, 960 * 32])
    amu, acov, aeigenval, aeigenvec = PCA.PCA(flat_an2i, 960)
    smu, scov, seigenval, seigenvec = PCA.PCA(flat_straight, 960)
    Q1_1(amu, aeigenvec[:, :2], smu, seigenvec[:, :2])
    Q1_2(aeigenval,acov, seigenval,scov)
    Q1_3(amu, aeigenvec, smu, seigenvec)
    Q1_4(amu, aeigenvec, smu, seigenvec)

def load_data():
    an2i = []
    straight = []
    sz = (32, 30)
    # read data
    for folder in os.listdir('faces_4/'):
        if folder == '.DS_Store':
            continue
        for file in os.listdir('faces_4/'+folder):
            if 'an2i' in file:
                an2i.append(read_pgm('faces_4/'+folder+'/'+file))
            if 'straight' in file:
                straight.append(read_pgm('faces_4/'+folder+'/'+file))
    flat_an2i = np.reshape(np.array(an2i), (len(an2i), 30 * 32))
    flat_straight = np.reshape(np.array(straight), (len(straight), 30*32))
    return flat_an2i, flat_straight

if __name__ == '__main__':
    main()
