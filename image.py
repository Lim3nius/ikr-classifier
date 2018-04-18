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

def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        print('Processing file: ', f)
        features[f] = imread(f, True).astype(np.float64).flatten()
    return features

def main():
    dir = ['data/target_dev', 'data/non_target_dev']
    res = []
    dat1 = png2fea(dir[0])
    dat2 = png2fea(dir[1])

    dat1 = np.vstack(dat1.values())
    dat2 = np.vstack(dat2.values())

    cov = np.cov(np.vstack([dat1,dat2]).T, bias=True)
    dim = cov.shape[1]

    d,e = scipy.linalg.eigh(cov, eigvals=(dim-3, dim-1))

    target = dat1.dot(e)
    non_target = dat2.dot(e)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(target[:,2], target[:,1],target[:,0], c='r', marker='o')
    ax.scatter(non_target[:,2], non_target[:,1], non_target[:,0], c='b', marker='^')
    plt.show()


if __name__ == '__main__':
    main()
