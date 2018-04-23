import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from numpy.linalg import eigh, det, inv, solve, norm
import scipy.fftpack
import sounddevice as sd
from scipy.io import wavfile
import features as ft
from glob import glob
from ikrlib import logistic_sigmoid

def train_modelA(target_dirs, non_target_dirs):
    target_coef = []
    target_frequency = []
    for dir_name in target_dirs:
        for f in glob(dir_name + '/*.wav'):
            print('Processing file: ', f)
            fs, f = wavfile.read(f)
            mfcc, freq = ft.features(f, fs)
            target_coef.append(mfcc)
            target_frequency.append(freq)
    target_coef = np.vstack(target_coef)
    target_frequency = np.array(target_frequency)

    non_target_coef = []
    non_target_frequency = []
    for dir_name in non_target_dirs:
        for f in glob(dir_name + '/*.wav'):
            print('Processing file: ', f)
            fs, f = wavfile.read(f)
            mfcc, freq = ft.features(f, fs)
            non_target_coef.append(mfcc)
            non_target_frequency.append(freq)
    non_target_coef = np.vstack(non_target_coef)
    non_target_frequency = np.array(non_target_frequency)

    mu_freq1 = target_frequency.mean()
    mu_freq2 = non_target_frequency.mean()
    cov_freq1 = target_frequency.var()
    cov_freq2 = non_target_frequency.var()

    # Initialize mean vectors to randomly selected data points from corresponding class
    # Initialize all covariance matrices to the same covariance matrices computed using
    # all the data from the given class
    m1 = 2
    mus1 = target_coef[randint(1, target_coef.shape[1], m1)]
    cov1 = np.cov(target_coef.T, bias=True)
    covs1 = [cov1] * m1
    ws1 = np.ones(m1) / m1

    m2 = 2
    mus2 = non_target_coef[randint(1, non_target_coef.shape[1], m2)]
    cov2 = np.cov(non_target_coef.T, bias=True)    
    covs2 = [cov2] * m2
    ws2 = np.ones(m2) / m2
    
    for i in range(30):
        ws1, mus1, covs1, ttl1 = ft.train_gmm(target_coef, ws1, mus1, covs1)
        ws2, mus2, covs2, ttl2 = ft.train_gmm(non_target_coef, ws2, mus2, covs2)
        print(ttl1,ttl2)

    return (mu_freq1, mu_freq2), (cov_freq1, cov_freq2), (mus1, mus2), (covs1, covs2), (ws1, ws2)


def test_modelA(test_dirs, muf, covf, mug, covg, ws, fs = 16000):
    freq_posterior = []
    mfcc_posterior = []
    for dir_name in test_dirs:
        for f in glob(dir_name + '/*.wav'):
            print('Processing file: ', f)
            fs, f = wavfile.read(f)
            mfcc, freq = ft.features(f, fs)
            freq_posterior.append(mlab.normpdf(freq, muf[0], math.sqrt(covf[0]))*0.5/(mlab.normpdf(freq, muf[1], math.sqrt(covf[1]))*0.5))
            mfcc = np.vstack(mfcc)
            tmp =[]
            for coef in mfcc:
                tmp.append(ft.logpdf_gmm(coef, ws[0], mug[0], covg[0]) + np.log(ws[0]) - ft.logpdf_gmm(coef, ws[1], mug[1], covg[1]) - np.log(ws[1]))
            mfcc_posterior.append(np.mean(tmp))

    return np.log(freq_posterior), mfcc_posterior