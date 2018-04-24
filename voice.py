import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy.random import rand
from numpy.linalg import eigh, det, inv, solve, norm
import scipy.fftpack
import sounddevice as sd
from scipy.io import wavfile
import features as ft
from glob import glob
import math
from ikrlib import logistic_sigmoid, logpdf_gauss

def foo():
    print(1)

def train_modelA(target_dirs, non_target_dirs, tgauss = 4, ngauss = 11):
    target_coef = []
    target_frequency = []
    print("Target data:")
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
    print("Non target data:")
    for dir_name in non_target_dirs:
        for f in glob(dir_name + '/*.wav'):
            print('Processing file: ', f)
            fs, f = wavfile.read(f)
            mfcc, freq = ft.features(f, fs)
            non_target_coef.append(mfcc)
            non_target_frequency.append(freq)
    non_target_coef = np.vstack(non_target_coef)
    non_target_frequency = np.array(non_target_frequency)

    print("Training gaussian distribution for frequency")
    mu_freq1 = target_frequency.mean()
    mu_freq2 = non_target_frequency.mean()
    cov_freq1 = target_frequency.var()
    cov_freq2 = non_target_frequency.var()

    # Initialize mean vectors to randomly selected data points from corresponding class
    # Initialize all covariance matrices to the same covariance matrices computed using
    # all the data from the given class
    print("Training GMM for mfcc")
    m1 = tgauss
    mus1 = target_coef[np.random.randint(1, target_coef.shape[1], m1)]
    cov1 = np.cov(target_coef.T, bias=True)
    covs1 = [cov1] * m1
    ws1 = np.ones(m1) / m1

    m2 = ngauss
    mus2 = non_target_coef[np.random.randint(1, non_target_coef.shape[1], m2)]
    cov2 = np.cov(non_target_coef.T, bias=True)    
    covs2 = [cov2] * m2
    ws2 = np.ones(m2) / m2
    
    for i in range(30):
        ws1, mus1, covs1, ttl1 = ft.train_gmm(target_coef, ws1, mus1, covs1)
        ws2, mus2, covs2, ttl2 = ft.train_gmm(non_target_coef, ws2, mus2, covs2)
        print("target error:", ttl1, "non target error: ", ttl2)

    # target_freq_post = []
    # target_mfcc_post = []
    # print("Ploting training data")
    # for freq in target_frequency:
    #     target_freq_post.append(scipy.stats.norm.logpdf(freq, mu_freq1, cov_freq1) + np.log(0.5)-(scipy.stats.norm.logpdf(freq, mu_freq2, cov_freq2) - np.log(0.5)))
    #     tmp =[]
    #     for coef in target_coef:
    #         tmp.append(ft.logpdf_gmm(coef, ws1, mus1, covs1) + np.log(0.5) - ft.logpdf_gmm(coef, ws2, mus2, cov2) - np.log(0.5))
    #     target_mfcc_post.append(np.mean(tmp))
    # non_target_freq_post = []
    # non_target_mfcc_post = []
    # for freq in non_target_frequency:
    #     non_target_freq_post.append(scipy.stats.norm.logpdf(freq, mu_freq1, cov_freq1) + np.log(0.5)-(scipy.stats.norm.logpdf(freq, mu_freq2, cov_freq2) - np.log(0.5)))
    #     tmp =[]
    #     for coef in target_coef:
    #         tmp.append(ft.logpdf_gmm(coef, ws1, mus1, covs1) + np.log(0.5) - ft.logpdf_gmm(coef, ws2, mus2, cov2) - np.log(0.5))
    #     non_target_mfcc_post.append(np.mean(tmp))

    # scatter(non_target_mfcc_post, non_target_freq_post,label='non_target_train')
    # scatter(target_mfcc_post, target_freq_post, label='target_train')

    return (mu_freq1, mu_freq2), (cov_freq1, cov_freq2), (mus1, mus2), (covs1, covs2), (ws1, ws2)


def test_modelA(test_dirs, muf, covf, mug, covg, ws, soft = False, fs = 16000):
    freq_posterior = []
    mfcc_posterior = []
    for dir_name in test_dirs:
        for f in glob(dir_name + '/*.wav'):
            print('Processing file: ', f)
            fs, f = wavfile.read(f)
            mfcc, freq = ft.features(f, fs)
            freq_posterior.append(scipy.stats.norm.logpdf(freq, muf[0], covf[0]) + np.log(0.5) - (scipy.stats.norm.logpdf(freq, muf[1], covf[1]) - np.log(0.5)))
            mfcc = np.vstack(mfcc)
            tmp =[]
            for coef in mfcc:
                tmp.append(ft.logpdf_gmm(coef, ws[0], mug[0], covg[0]) + np.log(0.5) - ft.logpdf_gmm(coef, ws[1], mug[1], covg[1]) - np.log(0.5))
            print(np.mean(tmp)+freq_posterior[-1])
            mfcc_posterior.append(np.mean(tmp))
    # fin = scipy.stats.norm.logpdf(freq_posterior, mup[0], covp[0]) - scipy.stats.norm.logpdf(freq_posterior, mup[0], covp[0])
    res = np.array(freq_posterior)+np.array(mfcc_posterior)
    if (soft is True):
        return res

    return res > 8.3