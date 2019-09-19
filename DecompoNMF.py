from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from scipy import signal
import math
#from scipy import signal

import librosa
import librosa.display

STEP = 512

title = 'Palestrina'
#Palestrina, Cadence4VMaj
y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/code/DescripteursHarmoniquesAudio/'+title+'.wav')

S = librosa.stft(y)
X, X_phase = librosa.magphase(S)
n_components = 3
W, H = librosa.decompose.decompose(X, n_components=n_components, sort=True)

print(S.shape)
print(W.shape)
print(H.shape)
