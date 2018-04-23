# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from numpy.linalg import eigh, det, inv, solve, norm
import scipy.fftpack
# import sounddevice as sd
from scipy.io import wavfile
# import features as ft
from glob import glob
from scipy.ndimage import imread
from mpl_toolkits.mplot3d import Axes3D
from ikrlib import *

def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        print "Processing file: ", f
        features[f] = imread(f, True).astype(np.float64).flatten()
    return features


def main():
    dir = ['data/target_train', 'data/non_target_train', 'data/target_dev', 'data/non_target_dev']
    res = []
    dat1 = png2fea(dir[0])
    dat2 = png2fea(dir[1])
    dat3 = png2fea(dir[2])
    dat4 = png2fea(dir[3])
    # dat1.update(dat3)

    dat1 = np.vstack(dat1.values())
    dat2 = np.vstack(dat2.values())
    dat3 = np.vstack(dat3.values())
    dat4 = np.vstack(dat4.values())
    
    new_dat = dat1
    for _ in range(6):
        new_dat = np.r_[new_dat, dat1]
    dat1 = new_dat

    print "Computing cov..."
    cov = np.cov(np.vstack([dat1, dat2]).T, bias=True)
    dim = cov.shape[1]

    print "Computing eigh..."
    d,e = scipy.linalg.eigh(cov, eigvals=(dim-2, dim-1), turbo=True)

    print "Computing dot..."
    target = dat1.dot(e)
    non_target = dat2.dot(e)
    test_target = dat3.dot(e)
    test_non_target = dat4.dot(e)

    x = np.r_[target, non_target]
    t = np.r_[np.ones(len(target)), np.zeros(len(non_target))]
    w, w0, _ = train_generative_linear_classifier(x, t)

    for _ in range(50):
        w, w0 = train_linear_logistic_regression(x, t, w, w0)

    #w0 *= dat2.shape[0] / float(dat1.shape[0])

    plt.plot(target[:,0], target[:,1], 'r.', label="Target_train")
    plt.plot(non_target[:,0], non_target[:,1], 'b.', label="Non_target_train")
    plt.plot(test_target[:,0], test_target[:,1], 'm.', label="Target_dev")
    plt.plot(test_non_target[:,0], test_non_target[:,1], 'c.', label="Non_target_dev")
    x1, x2 = plt.axis()[:2]
    y1 = (-w0 - (w[0] * x1)) / w[1]
    y2 = (-w0 - (w[0] * x2)) / w[1]
    plt.plot([x1, x2], [y1, y2], 'k', linewidth=2)
    plt.legend()
    plt.show()
    
    print "Test targets:"
    target_sum = 0
    for tgt in test_target:
        res = w.dot(tgt) + w0
        target_sum += res
        print res

    print "Test non_targets:"
    non_target_sum = 0
    for tgt in test_non_target:
        res = w.dot(tgt) + w0
        non_target_sum += res
        print res

    print "Target sum:", target_sum
    print "Non_target sum:", non_target_sum


if __name__ == '__main__':
    main()
