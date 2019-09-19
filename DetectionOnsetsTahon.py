# Beat tracking example
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

import params

BINS_PER_OCTAVE = params.BINS_PER_OCTAVE
N_OCTAVES = params.N_OCTAVES
NFFT = int(params.NFFT)
STEP = int(params.STEP)

title = 'Palestrina'
#Palestrina, Cadence4VMaj
y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/code/DescripteursHarmoniquesAudio/'+title+'.wav', duration = 6)


def detect_onsets(y, sr, M):
	#detect onsets
	oenv = librosa.onset.onset_strength(S=M, sr=sr)
	# Detect events without backtracking
	onset_raw = librosa.onset.onset_detect(onset_envelope=oenv, backtrack=False)
	# we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
	onset_frames = librosa.util.fix_frames(onset_raw, x_min=0, x_max=M.shape[1]-1)
	onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length = STEP)
    #onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length = STEP)
	# To reduce dimensionality, we'll beat-synchronous the CQT
	Msync = librosa.util.sync(M, onset_raw, aggregate=np.median)

	plt.figure(figsize=(12, 4))
	plt.plot(oenv, label='Onset strength')
	plt.vlines(onset_raw, 0, oenv.max(), label='Raw onsets')
	plt.legend(frameon=True, framealpha=0.75)
	plt.tight_layout()

	plt.figure(figsize=(12, 4))
	plt.subplot(2,1,1)
	plt.title('CQT spectrogram')
	librosa.display.specshow(M, y_axis='cqt_hz', sr=sr, hop_length= STEP, bins_per_octave=BINS_PER_OCTAVE, x_axis='time')
	plt.tight_layout()

	plt.subplot(2,1,2)
	plt.title('CQT spectrogram synchronized on onsets')
	librosa.display.specshow(Msync, bins_per_octave=BINS_PER_OCTAVE, y_axis='cqt_hz', x_axis='time', x_coords=onset_times)
	plt.tight_layout()





	return onset_raw, onset_times, Msync

C = librosa.amplitude_to_db(librosa.core.magphase(librosa.cqt(y=y, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE, hop_length = STEP))[0], ref=np.max)
onset_raw, onset_times, Msync =  detect_onsets(y, sr, C)


plt.show()
