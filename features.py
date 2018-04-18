import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from numpy.linalg import eigh, det, inv, solve, norm
import scipy.fftpack
import sounddevice as sd
from scipy.io import wavfile

def foo(a):
	print('11')

def features(f, fs):
	# fs, f = wavfile.read(file)
	sd.default.samplerate=fs
	window = 200
	noverlap = 120
	nfft = 256
	nbanks = 23
	nceps = 13
	snrdb = 40
	ewin = 400
	eshift = 200
	treshold = (f[15000:20000]**2).mean()*0.5
	s = f[20000:].copy()

	shape = ((s.shape[0] - ewin) // eshift+1 , ewin)+ s.shape[1:]
	strides = (s.strides[0] * eshift, s.strides[0]) + s.strides[1:]
	x = np.lib.stride_tricks.as_strided(s, shape=shape, strides=strides)
	#shape = window.size * x.shape[0] + window.size
	mask = np.mean(x**2, axis=1) > treshold
	# mask[-1] = False
	# x[mask,:eshift] = 0
	x = x[np.invert(mask)]
	x = np.append(x[:,:eshift], x[-1,eshift:])
	# odkomentovat ak bude padat
	# noise = rand(x.shape[0])
	# x = x + noise.dot(norm(x,2)) / norm(noise, 2)/ (10 **(snrdb/20))
	filt = x.copy()

	window = np.hamming(window)

	shift = window.size - noverlap
	shape = ((x.shape[0] - window.size) // shift+1 , window.size)+ x.shape[1:]
	strides = (x.strides[0] * shift, x.strides[0]) + x.strides[1:]
	x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

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

	x = scipy.fftpack.fft(x*window, nfft)
	S = x[:,:x.shape[1]//2+1]
	return dct_mx.dot(np.log(mfb.T.dot(np.abs(S.T)))).T, s, f, filt, treshold
