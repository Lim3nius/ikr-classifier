from numpy import polyfit, arange
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from numpy.linalg import eigh, det, inv, solve, norm
import scipy.fftpack
import sounddevice as sd
from scipy.io import wavfile
import math
from numpy.fft import rfft
from numpy import argmax, mean, diff, log
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve
from time import time
from scipy.special import logsumexp
import sys


def prt_rand(original, filtered, n = 1):
    for i in range(n):
        # idx = np.random.randint(len(filts))
        idx = i+10
        x = scipy.fftpack.fft(filtered[idx])
        x = x[:x.size//2]
        x = abs(x)

        b, axes = subplots(3,1)#,sharex=True,sharey=True)
        axes[0].set_title(idx)
        axes[0].plot(original[idx][20000:],color='xkcd:light olive')
        axes[1].plot(filtered[idx],color='xkcd:light orange')
        axes[2].plot(x,color='xkcd:coral')
        # axes[2].plot(filts[idx],color='xkcd:coral')

def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    https://gist.github.com/endolith/255291
    """
    # Calculate autocorrelation (same thing as convolution, but with
    # one input reversed in time), and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d = diff(corr)
    start = find(d > 0)[0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px

def filter(s, eshift, ewin, treshold):
    """
    """
	shape = ((s.shape[0] - ewin) // eshift+1 , ewin)+ s.shape[1:]
	strides = (s.strides[0] * eshift, s.strides[0]) + s.strides[1:]
	x = np.lib.stride_tricks.as_strided(s, shape=shape, strides=strides)
	mask = np.mean(x**2, axis=1) > treshold
	mask[-1] = False
	x = x[np.invert(mask)]
	x = np.append(x[:,:eshift], x[-1,eshift:])

	return x

def features(f, fs, treshold_koef = 0.75, window = 250 , noverlap = 200):
	sd.default.samplerate=fs
	nfft = 256
	nbanks = 23
	nceps = 13
	snrdb = 40

	treshold = (f[15000:20000]**2).mean()*treshold_koef
	s = f[20000:]
	# x = filter(s, 200, 400, treshold)
	x=s
	freq = freq_from_autocorr(s, fs)

	window = np.hamming(window)

	shift = window.size - noverlap
	shape = ((x.shape[0] - window.size) // shift+1 , window.size)+ x.shape[1:]
	strides = (x.strides[0] * shift, x.strides[0]) + x.strides[1:]
	xw = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
	mask = np.mean(xw**2, axis=1) > treshold
	mask[-1] = False
	# xw[mask,:shift] = 0
	xw = xw[np.invert(mask)]

	fend = 0.5 * fs
	fstart = 32
	melstart = 1127.*np.log(1. + fstart/ 700.)
	melend = 1127.*np.log(1. + fend/ 700.)
	ls = np.linspace(melstart, melend, nbanks +2)
	mel_inv = (np.exp(ls/1127.) - 1.)*700
	cbin = np.round(mel_inv/ fs * nfft).astype(int)
	mfb = np.zeros((nfft// 2 + 1, nbanks))

	for ii in range(nbanks):
		mfb[cbin[ii]:  cbin[ii+1]+1, ii] = np.linspace(0., 1., cbin[ii+1] - cbin[ii]   + 1)
		mfb[cbin[ii+1]:cbin[ii+2]+1, ii] = np.linspace(1., 0., cbin[ii+2] - cbin[ii+1] + 1)

	dct_mx = scipy.fftpack.idct(np.eye(nceps, nbanks), norm='ortho') # the same DCT as in matlab

	#if np.isscalar(window):

	xw = scipy.fftpack.fft(xw*window, nfft)
	S = xw[:,:xw.shape[1]//2+1]
	return dct_mx.dot(np.log(mfb.T.dot(np.abs(S.T)))).T, freq#, f, x, treshold


def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def train_gmm(x, ws, mus, covs):
    """
    TRAIN_GMM Single iteration of EM algorithm for training Gaussian Mixture Model
    [Ws_new,MUs_new, COVs_new, TLL]= TRAIN_GMM(X,Ws,NUs,COVs) performs single
    iteration of EM algorithm (Maximum Likelihood estimation of GMM parameters)
    using training data X and current model parameters Ws, MUs, COVs and returns
    updated model parameters Ws_new, MUs_new, COVs_new and total log likelihood
    TLL evaluated using the current (old) model parameters. The model
    parameters are mixture component mean vectors given by columns of M-by-D
    matrix MUs, covariance matrices given by M-by-D-by-D matrix COVs and vector
    of weights Ws.
    """   
    gamma = np.vstack([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)])
    logevidence = logsumexp(gamma, axis=0)
    gamma = np.exp(gamma - logevidence)
    tll = logevidence.sum()
    gammasum = gamma.sum(axis=1)
    ws = gammasum / len(x)
    mus = gamma.dot(x)/gammasum[:,np.newaxis]
    
    if covs[0].ndim == 1: # diagonal covariance matrices
      covs = gamma.dot(x**2)/gammasum[:,np.newaxis] - mus**2
    else:
      covs = np.array([(gamma[i]*x.T).dot(x)/gammasum[i] - mus[i][:, np.newaxis].dot(mus[[i]]) for i in range(len(ws))])        
    return ws, mus, covs, tll

def logpdf_gauss(x, mu, cov):
    assert(mu.ndim == 1 and len(mu) == len(cov) and (cov.ndim == 1 or cov.shape[0] == cov.shape[1]))
    x = np.atleast_2d(x) - mu
    if cov.ndim == 1:
        return -0.5*(len(mu)*np.log(2 * np.pi) + np.sum(np.log(cov)) + np.sum((x**2)/cov, axis=1))
    else:
        return -0.5*(len(mu)*np.log(2 * np.pi) + np.linalg.slogdet(cov)[1] + np.sum(x.dot(inv(cov)) * x, axis=1))

def logpdf_gmm(x, ws, mus, covs):
    return logsumexp([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)], axis=0)        

# def freq_from_HPS(sig, fs):
#     """
#     Estimate frequency using harmonic product spectrum (HPS)
#     """
#     windowed = sig * blackmanharris(len(sig))

#     from pylab import subplot, plot, log, copy, show

#     # harmonic product spectrum:
#     c = abs(rfft(windowed))
#     maxharms = 8
#     subplot(maxharms, 1, 1)
#     plot(log(c))
#     for x in range(2, maxharms):
#         a = copy(c[::x])  # Should average or maximum instead of decimating
#         # max(c[::x],c[1::x],c[2::x],...)
#         c = c[:len(a)]
#         i = argmax(abs(c))
#         true_i = parabolic(abs(c), i)[0]
#         print ('Pass %d: %f Hz' % (x, fs * true_i / len(windowed)))
#         c *= a
#         subplot(maxharms, 1, x)
#         plot(log(c))
#     show()
