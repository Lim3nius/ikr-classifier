%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from numpy.linalg import eigh, det, inv, solve, norm
import scipy.fftpack
import sounddevice as sd
from scipy.io import wavfile
import features as ft
from glob import glob
dir_name = 'non_target_dev'

c = []
ss = []
origs = []
filts = []
masks = []

for f in glob(dir_name + '/*.wav'):
    print('Processing file: ', f)
    fs, f = wavfile.read(f)
    # x = ft.features(f, fs)
    # break
    mfcc, s, f, filt, mask = ft.features(f, fs)
    c.append(mfcc)
    ss.append(s)
    origs.append(f)
    filts.append(filt)
    masks.append(mask)

c = np.vstack(c)

for i in range(3):
	# idx = np.random.randint(len(filts))
	idx = i+10
	b, axes = subplots(3,1,sharex=True,sharey=True)
	axes[0].set_title(idx)
	axes[0].plot(origs[idx][20000:],color='xkcd:light olive')
	axes[1].plot(ss[idx],color='xkcd:light orange')
	axes[2].plot(filts[idx],color='xkcd:coral')